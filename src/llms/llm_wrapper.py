# -*- coding: utf-8 -*-
"""
LLM 调用封装 —— 统一 invoke() 与 stream() 接口。

支持流式输出（用于首字延迟 TTFT 计算），自动从 ModelFactory 获取后端实例。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Generator, Optional

from langchain_core.messages import HumanMessage

from src.llms.models import ModelFactory
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class LLMWrapper:
    """大语言模型调用封装器。

    Usage::

        llm = LLMWrapper(temperature=0.1)
        answer = llm.invoke("你是谁？")
        for token in llm.stream("讲个笑话"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """
        Args:
            provider:    提供商标识，None 则使用 config.yaml 默认。
            model_name:  模型名称，None 则使用 config.yaml 默认。
            temperature: 生成温度，None 则使用 config.yaml 默认。
            max_tokens:  最大生成 Token 数，None 则使用 config.yaml 默认。
        """
        cfg = get_config()
        self._provider = provider or cfg.models.llm.provider
        self._model_name = model_name or cfg.models.llm.model_name
        self._temperature = (
            temperature if temperature is not None else cfg.models.llm.temperature
        )
        self._max_tokens = max_tokens or cfg.models.llm.max_tokens
        self._model: Any = None  # 懒加载

    @property
    def model(self):
        """懒加载底层 LLM 实例。"""
        if self._model is None:
            self._model = ModelFactory.get_llm(
                provider=self._provider,
                model_name=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        return self._model

    # ── 同步调用 ───────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """同步生成回答。

        Args:
            prompt:        用户提示词。
            system_prompt: 可选的系统角色设定。

        Returns:
            {"content": <文本>, "model": <模型名>, "usage": <token 用量>}
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(HumanMessage(content=prompt))

        t0 = time.perf_counter()
        resp = self.model.invoke(messages)
        elapsed = time.perf_counter() - t0

        content = resp.content if hasattr(resp, "content") else str(resp)

        usage = {}
        if hasattr(resp, "response_metadata"):
            usage = resp.response_metadata.get("token_usage", {})

        result = {
            "content": content,
            "model": f"{self._provider}/{self._model_name}",
            "latency_seconds": round(elapsed, 4),
            "usage": usage,
        }
        logger.debug("invoke 完成: latency=%.3fs, chars=%d", elapsed, len(content))
        return result

    # ── 流式调用 ───────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[dict, None, None]:
        """流式生成，逐 Token 产出。

        每个产出为 {"token": <文本片段>, "is_first": <bool>}。
        首个 token 之前的耗时即为 TTFT（首字延迟）。

        Args:
            prompt:        用户提示词。
            system_prompt: 可选的系统角色设定。

        Yields:
            包含 token 文本和 is_first 标记的字典。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(HumanMessage(content=prompt))

        t0 = time.perf_counter()
        is_first = True

        try:
            for chunk in self.model.stream(messages):
                chunk_content = ""
                if hasattr(chunk, "content"):
                    chunk_content = chunk.content or ""
                elif isinstance(chunk, str):
                    chunk_content = chunk

                if chunk_content:
                    info = {"token": chunk_content, "is_first": is_first}
                    if is_first:
                        ttft = time.perf_counter() - t0
                        info["ttft_seconds"] = round(ttft, 4)
                        logger.debug("stream TTFT: %.3fs", ttft)
                        is_first = False
                    yield info
        except Exception as exc:
            logger.error("流式生成异常: %s", exc)
            raise

    # ── 带 TTFT 测量的流式封装 ────────────────────────────────────────

    def stream_with_metrics(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """流式生成并返回完整指标（TTFT + 总耗时 + 完整文本）。

        Args:
            prompt:        用户提示词。
            system_prompt: 可选的系统角色设定。

        Returns:
            {
                "content": <完整文本>,
                "ttft_seconds": <首字延迟>,
                "total_latency_seconds": <总耗时>,
                "token_count": <token 数量（流式累计）>,
            }
        """
        full_text = ""
        ttft = 0.0
        token_count = 0
        t_start = time.perf_counter()

        for info in self.stream(prompt, system_prompt=system_prompt):
            full_text += info["token"]
            token_count += 1
            if info.get("is_first"):
                ttft = info.get("ttft_seconds", 0.0)

        total_latency = time.perf_counter() - t_start

        return {
            "content": full_text,
            "ttft_seconds": round(ttft, 4),
            "total_latency_seconds": round(total_latency, 4),
            "token_count": token_count,
        }
