# -*- coding: utf-8 -*-
"""
会话上下文管理器 —— 多轮对话的窗口截断与摘要压缩。

解决长对话的 Token 爆炸问题：
  1. 保留最近 N 条消息作为短期上下文；
  2. Token 超限时，将较早历史压缩为 LLM 摘要；
  3. 构建最终 Prompt 时拼接「摘要 + 最近对话 + 当前问题」。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.utils.config import get_config

logger = logging.getLogger(__name__)


class ContextManager:
    """多轮会话上下文管理器。

    Usage::
        ctx = ContextManager()
        history = [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
        ]
        context, summary = ctx.build(history, "当前问题")
    """

    def __init__(self) -> None:
        cfg = get_config().session_context
        self._max_history = cfg.max_history_messages
        self._max_tokens = cfg.max_context_tokens
        self._summary_enabled = cfg.summary_enabled
        self._llm = None

    @property
    def llm(self):
        """懒加载 LLM，用于历史摘要压缩。"""
        if self._llm is None:
            from src.llms.models import ModelFactory
            self._llm = ModelFactory.get_llm(temperature=0)
        return self._llm

    def estimate_tokens(self, text: str) -> int:
        """粗略估算文本 Token 数（中文按字符/1.5，英文按词*1.3）。"""
        if not text:
            return 0
        from src.evaluation.metrics import calculate_tokens
        try:
            return calculate_tokens(text)
        except Exception:
            chinese = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
            if chinese > len(text) * 0.2:
                return max(1, int(len(text) / 1.5))
            return max(1, int(len(text.split()) * 1.3))

    def summarize_history(self, history: List[Dict[str, Any]]) -> str:
        """使用 LLM 将历史消息压缩为简短摘要。

        Args:
            history: [{"role", "content"}, ...] 消息列表。

        Returns:
            摘要文本，LLM 失败时返回空字符串。
        """
        if not history:
            return ""

        dialog = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}: {m.get('content', '')}"
            for m in history
        )
        prompt = (
            "请将以下客服对话压缩为简洁摘要，保留关键信息（用户诉求、"
            "已提供的信息、未解决的问题），不超过 200 字：\n\n" + dialog[-4000:]
        )
        try:
            from langchain_core.messages import HumanMessage
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            summary = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            logger.info("历史摘要压缩完成: %d 条消息 → %d 字", len(history), len(summary))
            return summary
        except Exception as exc:
            logger.warning("历史摘要压缩失败: %s", exc)
            return ""

    def build(
        self,
        history: List[Dict[str, Any]],
        current_question: str,
    ) -> str:
        """构建拼入 Prompt 的多轮上下文。

        Args:
            history:         完整历史消息（按时间正序）。
            current_question: 用户当前问题。

        Returns:
            上下文文本，可直接作为 RAG Prompt 的 history 部分。
        """
        if not history:
            return ""

        # 1. 截断到最近 N 条
        recent = history[-self._max_history:]

        # 2. 估算 Token，超限则摘要压缩较早部分
        total_tokens = sum(
            self.estimate_tokens(str(m.get("content", ""))) for m in recent
        )

        if total_tokens <= self._max_tokens:
            return self._format(recent)

        if not self._summary_enabled:
            # 不启用摘要时，只保留能容纳的最近消息
            kept: List[Dict[str, Any]] = []
            acc = 0
            for m in reversed(recent):
                t = self.estimate_tokens(str(m.get("content", "")))
                if acc + t > self._max_tokens:
                    break
                kept.insert(0, m)
                acc += t
            return self._format(kept)

        # 3. 摘要压缩较早历史 + 保留最近消息
        split_point = max(1, len(recent) // 2)
        old_part = recent[:split_point]
        new_part = recent[split_point:]
        summary = self.summarize_history(old_part)

        if summary:
            head = f"[历史摘要] {summary}"
        else:
            head = self._format(old_part)
        return head + "\n\n" + self._format(new_part)

    @staticmethod
    def _format(messages: List[Dict[str, Any]]) -> str:
        """将消息列表格式化为对话文本。"""
        lines = []
        for m in messages:
            role = "用户" if m.get("role") == "user" else "助手"
            lines.append(f"{role}: {m.get('content', '')}")
        return "\n".join(lines)
