# -*- coding: utf-8 -*-
"""
RAG 问答链 —— 串联检索、Prompt 模板与 LLM 生成。

流程::

    用户问题
        │
        ▼
    retrieve(query)  ← 混合检索 + 重排序
        │
        ▼
    构建 Context Prompt（文档片段 + 问题 + 约束）
        │
        ▼
    LLMWrapper.invoke()  ← 调用 LLM 生成
        │
        ▼
    {
        "answer": "...",
        "retrieved_docs": [...],
        "source_metadata": [...],
    }
"""
from __future__ import annotations

import logging
from typing import Any, Dict,Generator, List, Optional

from langchain_core.documents import Document

from src.llms.llm_wrapper import LLMWrapper
from src.retrievers.pipeline import retrieve as pipeline_retrieve
from src.utils.config import get_config

logger = logging.getLogger(__name__)

_monitor_pipeline = None


def _get_monitor():
    global _monitor_pipeline
    if _monitor_pipeline is None:
        from src.evaluation.monitor import MonitorPipeline
        _monitor_pipeline = MonitorPipeline()
    return _monitor_pipeline

# ── 默认 Prompt 模板 ──────────────────────────────────────────────────────
# DEFAULT_RAG_PROMPT = (
#     "你是一个专业的知识库问答助手。请严格基于以下检索到的文档内容回答问题。\n"
#     "\n"
#     "要求：\n"
#     "  - 如果文档内容足以回答问题，请给出准确、简洁的回答\n"
#     "  - 如果文档内容不足，请明确回答\"根据现有资料无法回答\"\n"
#     "  - 不要编造或推测文档中没有的信息\n"
#     "  - 在回答末尾列出引用的文档来源（文件名）\n"
#     "\n"
#     "--- 检索到的文档内容 ---\n"
#     "{context}\n"
#     "--- 文档内容结束 ---\n"
#     "\n"
#     "用户问题：{question}\n"
# )
DEFAULT_RAG_PROMPT = (
    "你是一个专业的知识库问答助手，同时也乐于与用户进行友好的日常交流。\n"
    "\n"
    "## 回答策略\n"
    "请根据用户问题的性质，选择对应的回答方式：\n"
    "\n"
    "**情况一：用户的问题与检索到的文档内容相关**\n"
    "  - 严格基于文档内容给出准确、简洁的回答\n"
    "  - 不要编造或推测文档中没有的信息\n"
    "  - 如果文档内容不足以回答问题，请明确告知用户\"当前知识库中暂无相关资料\"\n"
    "  - 在回答末尾列出引用的文档来源（文件名）\n"
    "\n"
    "**情况二：文档内容为\"未检索到相关文档\"**\n"
    "  - 请明确告知用户\"当前知识库中暂无相关资料，请补充后再提问\"\n"
    "\n"
    "**情况三：用户的问题与文档内容无关（如问候、闲聊、通用知识问题等）**\n"
    "  - 以友好、自然的方式直接回答，无需引用文档\n"
    "  - 保持专业且亲切的语气\n"
    "\n"
    "--- 检索到的文档内容 ---\n"
    "{context}\n"
    "--- 文档内容结束 ---\n"
    "\n"
    "--- 对话历史 ---\n"
    "{history}\n"
    "--- 历史结束 ---\n"
    "\n"
    "用户问题：{question}\n"
)


class RAGChain:
    """端到端 RAG 问答链。

    Usage::

        chain = RAGChain()
        result = chain.answer("Qdrant 的分片策略是什么？")
        print(result["answer"])
    """

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> None:
        """
        Args:
            prompt_template: 自定义 Prompt 模板，需包含 {context} 和 {question} 占位符。
                             不指定时使用 DEFAULT_RAG_PROMPT。
            top_k:          传给检索器的最终返回文档数。
            use_cache:      是否启用缓存（若为 True，从 CacheService 读取命中结果）。
        """
        self._prompt_template = prompt_template or DEFAULT_RAG_PROMPT
        self._top_k = top_k or get_config().retrieval.top_k
        self._use_cache = use_cache
        self._llm: Optional[LLMWrapper] = None
        self._cache: Any = None

    @property
    def llm(self) -> LLMWrapper:
        """懒加载 LLM 封装器。"""
        if self._llm is None:
            self._llm = LLMWrapper()
        return self._llm

    def _get_cache(self):
        """懒加载缓存服务。"""
        if self._cache is None:
            from src.cache.cache_service import CacheService
            self._cache = CacheService()
        return self._cache

    @staticmethod
    def _build_history(history: Optional[List[Dict[str, Any]]]) -> str:
        """构建多轮会话上下文（窗口截断 + 摘要压缩）。"""
        if not history:
            return "（无历史对话）"
        try:
            from src.sessions.context_manager import ContextManager
            ctx = ContextManager()
            text = ctx.build(history, history[-1].get("content", "") if history else "")
            return text or "（无历史对话）"
        except Exception as exc:
            logger.warning("会话上下文构建失败，忽略历史: %s", exc)
            return "（无历史对话）"

    # ── 主入口 ────────────────────────────────────────────────────────

    def answer_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        regenerate: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式 RAG 问答：先检索，再流式生成。

        Yields:
            {"type": "status", "content": "检索中..."}
            {"type": "status", "content": "思考中..."}
            {"type": "token", "content": "<文本片段>"}
            {"type": "sources", "content": [...]}
            {"type": "done", "content": "<完整回答>"}
        """
        if not question.strip():
            yield {"type": "error", "content": "问题为空"}
            return

        # 1. 检查缓存
        if self._use_cache and not regenerate:
            try:
                cache = self._get_cache()
                cached = cache.get(question, user_id=user_id)
                if cached:
                    logger.info("缓存命中(流式): %s...", question[:40])
                    yield {"type": "cached", "content": True}
                    yield {"type": "token", "content": cached.get("answer", "")}
                    yield {"type": "sources", "content": cached.get("source_metadata", [])}
                    yield {"type": "done", "content": cached.get("answer", "")}
                    return
            except Exception as exc:
                logger.warning("缓存读取失败: %s", exc)

        # 2. 检索
        yield {"type": "status", "content": "🔍 正在检索相关知识库..."}
        try:
            retrieved_docs = pipeline_retrieve(
                query=question,
                candidate_top_n=self._top_k,
                use_reranker=True,
            )
        except Exception as exc:
            logger.error("检索失败: %s", exc)
            yield {"type": "error", "content": f"检索失败: {exc}"}
            return

        if not retrieved_docs:
            yield {"type": "status", "content": "⚠️ 未检索到相关文档，尝试基于自身知识回答..."}
            retrieved_docs = []

        # 3. 构建 Prompt
        context = self._build_context(retrieved_docs) if retrieved_docs else "未检索到相关文档"
        history_text = self._build_history(history)
        prompt = self._prompt_template.format(
            context=context,
            history=history_text,
            question=question,
        )

        # 4. 流式生成
        yield {"type": "status", "content": "💭 正在思考..."}
        source_metadata = self._extract_source_metadata(retrieved_docs) if retrieved_docs else []
        full_text = ""

        try:
            for info in self.llm.stream(prompt):
                token = info["token"]
                full_text += token
                yield {"type": "token", "content": token}
        except Exception as exc:
            logger.error("LLM 流式生成失败: %s", exc)
            yield {"type": "error", "content": f"生成失败: {exc}"}
            return

        # 5. 写入缓存
        if self._use_cache and full_text:
            try:
                cache_entry = {
                    "question": question,
                    "answer": full_text,
                    "source_metadata": source_metadata,
                }
                self._get_cache().set(
                    question,
                    cache_entry,
                    user_id=user_id,
                )
            except Exception:
                pass

        yield {"type": "sources", "content": source_metadata}

        # ── 6. MonitorPipeline 评估（流式）──
        try:
            from src.evaluation.metrics import calculate_tokens
            contexts_text = [doc.page_content for doc in retrieved_docs] if retrieved_docs else []
            eval_result = _get_monitor().evaluate(
                query=question,
                answer=full_text,
                contexts=contexts_text,
                model=getattr(self._llm, '_provider', '') + '/' + getattr(self._llm, '_model_name',
                                                                          '') if self._llm else "",
                input_tokens=calculate_tokens(question + context),
                output_tokens=calculate_tokens(full_text),
                latency=0.0,
            )
            logger.info("流式评估完成: cost=%.6f", eval_result.get("cost_usd", 0))
        except Exception as exc:
            logger.warning("MonitorPipeline 流式评估失败(不影响主流程): %s", exc)

        yield {"type": "done", "content": full_text}

    def answer(
        self,
        question: str,
        history: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        regenerate: bool = False,
    ) -> Dict[str, Any]:
        """对用户问题执行完整的 RAG 问答流程。

        Args:
            question: 用户自然语言问题。
            history:  多轮对话历史（可选），[{"role", "content"}, ...]。

        Returns:
            {
                "question":       原始问题,
                "answer":         LLM 生成的回答文本,
                "retrieved_docs": 检索到的 LangChain Document 列表,
                "source_metadata": [
                    {"source": "doc.md", "section_title": "...", "page": 1, "score": 0.92},
                    ...
                ],
                "cached":         是否从缓存命中,
            }
        """
        if not question.strip():
            return self._empty_result(question, "问题为空")

        # ── 1. 检查缓存 ──────────────────────────────────────────────
        if self._use_cache and not regenerate:
            try:
                cache = self._get_cache()
                cached = cache.get(question, user_id=user_id)
                if cached:
                    logger.info("缓存命中: %s...", question[:40])
                    result = dict(cached)
                    result["cached"] = True
                    result["retrieved_docs"] = []
                    result["source_metadata"] = result.get("source_metadata", [])
                    return result
            except Exception as exc:
                logger.warning("缓存读取失败: %s", exc)

        # ── 2. 检索 ──────────────────────────────────────────────────
        try:
            retrieved_docs = pipeline_retrieve(
                query=question,
                top_k=self._top_k,
                use_reranker=True,
            )
        except Exception as exc:
            logger.error("检索失败: %s", exc)
            return self._empty_result(question, f"检索失败: {exc}")

        if not retrieved_docs:
            logger.warning("检索结果为空: %s...", question[:40])
            return self._empty_result(question, "未检索到相关文档")

        # ── 3. 构建 Prompt ───────────────────────────────────────────
        context = self._build_context(retrieved_docs)
        history_text = self._build_history(history)
        prompt = self._prompt_template.format(
            context=context,
            history=history_text,
            question=question,
        )

        # ── 4. LLM 生成 ──────────────────────────────────────────────
        try:
            llm_result = self.llm.invoke(prompt)
            logger.info("LLM 生成完成: %s", llm_result)
            answer_text = llm_result.get("content", "")
        except Exception as exc:
            logger.error("LLM 生成失败: %s", exc)
            return self._empty_result(question, f"生成失败: {exc}")

        # ── 5. 收集来源元数据 ────────────────────────────────────────
        source_metadata = self._extract_source_metadata(retrieved_docs)

        result = {
            "question": question,
            "answer": answer_text,
            "retrieved_docs": retrieved_docs,
            "source_metadata": source_metadata,
            "cached": False,
            "model": llm_result.get("model", ""),
            "latency_seconds": llm_result.get("latency_seconds", 0),
        }

        # ── 6. MonitorPipeline 评估（成本 + Ragas + Langfuse 追踪）──
        try:
            usage = llm_result.get("usage", {})
            contexts_text = [doc.page_content for doc in retrieved_docs]
            eval_result = _get_monitor().evaluate(
                query=question,
                answer=answer_text,
                contexts=contexts_text,
                model=result["model"],
                input_tokens=usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
                latency=result["latency_seconds"],
            )
            result["evaluation"] = eval_result
            logger.info("评估完成: cost=%.6f, ragas=%s", eval_result.get("cost_usd", 0),
                         eval_result.get("ragas_scores", {}))
        except Exception as exc:
            logger.warning("MonitorPipeline 评估失败(不影响主流程): %s", exc)

        # ── 7. 写入缓存 ──────────────────────────────────────────────
        if self._use_cache:
            try:
                cache_entry = {
                    "question": question,
                    "answer": answer_text,
                    "source_metadata": source_metadata,
                    "model": result["model"],
                }
                self._get_cache().set(
                    question,
                    cache_entry,
                    user_id=user_id,
                )
            except Exception as exc:
                logger.warning("缓存写入失败: %s", exc)

        logger.info(
            "RAG 问答完成: '%s...' → %d 文档, %d chars",
            question[:40],
            len(retrieved_docs),
            len(answer_text),
        )
        return result

    # ── 内部方法 ─────────────────────────────────────────────────────

    @staticmethod
    def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
        """将检索到的文档拼接为上下文文本，控制总长度。"""
        parts: List[str] = []
        total = 0
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", f"doc_{i}")
            text = doc.page_content
            header = f"[文档 {i+1}: {source}]"
            block = f"{header}\n{text}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > len(header) + 20:
                    block = block[:remaining] + "..."
                    parts.append(block)
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)

    @staticmethod
    def _extract_source_metadata(docs: List[Document]) -> List[Dict]:
        """从检索文档中提取简化的来源信息。"""
        sources: List[Dict] = []
        seen = set()
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            if src in seen:
                continue
            seen.add(src)
            sources.append({
                "source": src,
                "section_title": doc.metadata.get("section_title", ""),
                "page": doc.metadata.get("page", 1),
                "file_type": doc.metadata.get("file_type", ""),
                "score": doc.metadata.get("rerank_score", doc.metadata.get("qdrant_score", 0)),
            })
        return sources

    @staticmethod
    def _empty_result(question: str, reason: str) -> Dict[str, Any]:
        """返回空结果模板。"""
        return {
            "question": question,
            "answer": "",
            "retrieved_docs": [],
            "source_metadata": [],
            "cached": False,
            "error": reason,
        }
