# -*- coding: utf-8 -*-
"""
评估指标计算 —— Recall / Precision / Faithfulness / Relevancy / Latency / Tokens。

包含 MetricsTracker 单例，记录每次请求的指标到内存日志，支持历史查询和统计汇总。
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ============================================================================
# 单次请求指标快照
# ============================================================================


@dataclass
class MetricSnapshot:
    """单次 RAG 请求的评估指标快照。"""
    timestamp: str = ""
    query: str = ""
    latency_seconds: float = 0.0
    num_retrieved: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    recall: Optional[float] = None
    precision: Optional[float] = None
    faithfulness_score: Optional[float] = None
    relevancy_score: Optional[float] = None


# ============================================================================
# 指标计算函数
# ============================================================================


# ── 语言检测辅助 ──────────────────────────────────────────────────────────
def _is_chinese(text: str) -> bool:
    """判断文本是否主要为中文。"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars > len(text) * 0.2 if text else False


# ── Token 计数 ────────────────────────────────────────────────────────────

def calculate_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """使用 tiktoken 计算文本的 Token 数量。

    如果模型不在 tiktoken 支持的列表中，回退到启发式估算：
    - 中文：字符数 / 1.5
    - 英文：单词数 * 1.3

    Args:
        text:       待计数的文本。
        model_name: 模型名称（用于选 tiktoken 编码器）。

    Returns:
        Token 估算值。
    """
    if not text:
        return 0

    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        pass

    # 回退：启发式
    if _is_chinese(text):
        return max(1, int(len(text) / 1.5))
    else:
        return max(1, int(len(text.split()) * 1.3))


# ── Recall ────────────────────────────────────────────────────────────────

def calculate_recall(
    retrieved_docs: List[Document],
    relevant_doc_ids: Set[str],
) -> float:
    """计算召回率。

    Args:
        retrieved_docs:   检索返回的文档列表。
        relevant_doc_ids: 真实相关文档的唯一标识集合（如 {source:page}）。

    Returns:
        召回率 [0.0, 1.0]。
    """
    if not relevant_doc_ids:
        return 0.0

    retrieved_ids: Set[str] = set()
    for doc in retrieved_docs:
        qdrant_id = doc.metadata.get("qdrant_id", "")
        source = doc.metadata.get("source", "")
        page = str(doc.metadata.get("page", 1))
        doc_id = qdrant_id or f"{source}#{page}"
        retrieved_ids.add(doc_id)

    hits = len(retrieved_ids & relevant_doc_ids)
    return hits / len(relevant_doc_ids)


# ── Precision ──────────────────────────────────────────────────────────────

def calculate_precision(
    retrieved_docs: List[Document],
    relevant_doc_ids: Set[str],
) -> float:
    """计算精确率。

    Args:
        retrieved_docs:   检索返回的文档列表。
        relevant_doc_ids: 真实相关文档的唯一标识集合。

    Returns:
        精确率 [0.0, 1.0]。
    """
    if not retrieved_docs:
        return 0.0

    hits = 0
    for doc in retrieved_docs:
        qdrant_id = doc.metadata.get("qdrant_id", "")
        source = doc.metadata.get("source", "")
        page = str(doc.metadata.get("page", 1))
        doc_id = qdrant_id or f"{source}#{page}"
        if doc_id in relevant_doc_ids:
            hits += 1

    return hits / len(retrieved_docs)


# ── Faithfulness（LLM 评判） ──────────────────────────────────────────────

FAITHFULNESS_PROMPT = (
    "请判断以下回答是否**忠实于**给定的上下文文档。\n"
    "\n"
    "评分标准：\n"
    "  1.0 — 所有陈述都能在上下文中找到直接依据\n"
    "  0.5 — 部分陈述有依据，部分为推测\n"
    "  0.0 — 全部为编造或与上下文矛盾\n"
    "\n"
    "上下文文档：\n"
    "{context}\n"
    "\n"
    "回答：\n"
    "{answer}\n"
    "\n"
    "请**只输出一个数字**（如 1.0 / 0.5 / 0.0），不要任何解释。"
)


def calculate_faithfulness(
    answer: str,
    context_docs: List[Document],
    llm: Optional[Any] = None,
) -> Optional[float]:
    """使用 LLM 评判回答是否忠实于上下文。

    Args:
        answer:       RAG 系统生成的回答。
        context_docs: 用作上下文的检索文档列表。
        llm:          LangChain ChatModel 实例，None 时通过 ModelFactory 获取。

    Returns:
        忠实度分数 [0.0, 1.0]，LLM 不可用时返回 None。
    """
    if not answer or not context_docs:
        return None

    context = "\n\n".join(doc.page_content[:1000] for doc in context_docs)
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    try:
        if llm is None:
            from src.llms.models import ModelFactory
            llm = ModelFactory.get_llm(temperature=0)

        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

        # 提取数字
        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            # 尝试从文本中提取
            import re
            match = re.search(r"(\d+(?:\.\d+)?)", raw)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
            return None
    except Exception as exc:
        logger.warning("Faithfulness 评判失败: %s", exc)
        return None


# ── Relevancy（LLM 评判） ─────────────────────────────────────────────────

RELEVANCY_PROMPT = (
    "请判断以下回答是否**切题**，即是否直接回应了用户的问题。\n"
    "\n"
    "评分标准：\n"
    "  1.0 — 回答完全切题，直接回应问题的核心\n"
    "  0.5 — 回答部分切题，包含一些无关内容\n"
    "  0.0 — 完全不切题或答非所问\n"
    "\n"
    "用户问题：\n"
    "{question}\n"
    "\n"
    "回答：\n"
    "{answer}\n"
    "\n"
    "请**只输出一个数字**（如 1.0 / 0.5 / 0.0），不要任何解释。"
)


def calculate_relevancy(
    question: str,
    answer: str,
    llm: Optional[Any] = None,
) -> Optional[float]:
    """使用 LLM 评判回答是否切题。

    Args:
        question: 用户原始问题。
        answer:   RAG 系统生成的回答。
        llm:      LangChain ChatModel 实例。

    Returns:
        切题度分数 [0.0, 1.0]。
    """
    if not question or not answer:
        return None

    prompt = RELEVANCY_PROMPT.format(question=question, answer=answer)

    try:
        if llm is None:
            from src.llms.models import ModelFactory
            llm = ModelFactory.get_llm(temperature=0)

        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

        try:
            score = float(raw)
            return max(0.0, min(1.0, score))
        except ValueError:
            import re
            match = re.search(r"(\d+(?:\.\d+)?)", raw)
            if match:
                return max(0.0, min(1.0, float(match.group(1))))
            return None
    except Exception as exc:
        logger.warning("Relevancy 评判失败: %s", exc)
        return None


# ── Latency 装饰器 ────────────────────────────────────────────────────────


def record_latency(func: Callable) -> Callable:
    """装饰器：记录函数执行耗时。

    装饰后的函数返回 (result, latency_seconds) 元组，
    并通过 MetricsTracker 记录耗时。

    Usage::

        @record_latency
        def my_search(query):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        latency = round(elapsed, 6)
        logger.debug("%s 耗时: %.4fs", func.__name__, latency)

        # 如果结果已经是 (value, latency) 元组则不覆盖
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], float):
            return result

        return result, latency

    return wrapper


# ============================================================================
# MetricsTracker 单例
# ============================================================================


class MetricsTracker:
    """指标追踪器（线程安全单例）。

    记录每次 RAG 请求的关键指标到内存日志，用于离线分析和监控。

    Usage::

        tracker = MetricsTracker()
        tracker.start_request("Qdrant 分片策略")
        # ... 检索 & 生成 ...
        tracker.end_request(
            num_retrieved=5,
            tokens_input=1200,
            tokens_output=150,
            recall=0.8,
            precision=0.6,
        )
        print(tracker.summary())
    """

    _instance: Optional["MetricsTracker"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricsTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._history: List[MetricSnapshot] = []
        self._pending: Dict[int, MetricSnapshot] = {}  # thread_id → 当前请求快照
        self._max_history = 10000
        self._write_lock = threading.Lock()
        self._initialized = True

    # ── 请求生命周期 ──────────────────────────────────────────────────

    def start_request(self, query: str) -> None:
        """标记一个新请求的开始。

        Args:
            query: 用户查询。
        """
        tid = threading.get_ident()
        snap = MetricSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
        )
        snap.latency_seconds = time.perf_counter()  # 暂存开始时间戳
        self._pending[tid] = snap

    def end_request(
        self,
        num_retrieved: int = 0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        recall: Optional[float] = None,
        precision: Optional[float] = None,
        faithfulness_score: Optional[float] = None,
        relevancy_score: Optional[float] = None,
    ) -> Optional[MetricSnapshot]:
        """结束当前请求，计算最终耗时并记录所有指标。

        Args:
            num_retrieved:      检索返回的文档数。
            tokens_input:       输入 Token 数。
            tokens_output:      输出 Token 数。
            recall:             召回率。
            precision:          精确率。
            faithfulness_score: 忠实度分数。
            relevancy_score:    切题度分数。

        Returns:
            完整的 MetricSnapshot。
        """
        tid = threading.get_ident()
        snap = self._pending.pop(tid, None)
        if snap is None:
            logger.warning("end_request 调用无对应 start_request")
            return None

        # 计算实际耗时
        start_ts = snap.latency_seconds
        snap.latency_seconds = round(time.perf_counter() - start_ts, 6)

        snap.num_retrieved = num_retrieved
        snap.tokens_input = tokens_input
        snap.tokens_output = tokens_output
        snap.recall = recall
        snap.precision = precision
        snap.faithfulness_score = faithfulness_score
        snap.relevancy_score = relevancy_score

        with self._write_lock:
            self._history.append(snap)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        logger.debug(
            "指标记录: query='%s', latency=%.4fs, recall=%s, precision=%s",
            snap.query[:40],
            snap.latency_seconds,
            str(snap.recall),
            str(snap.precision),
        )
        return snap

    # ── 查询接口 ──────────────────────────────────────────────────────

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近 N 条请求记录。

        Args:
            limit: 返回条数。

        Returns:
            字典列表（可序列化）。
        """
        with self._write_lock:
            recent = self._history[-limit:]
        return [
            {
                "timestamp": s.timestamp,
                "query": s.query[:80],
                "latency_seconds": s.latency_seconds,
                "num_retrieved": s.num_retrieved,
                "tokens_input": s.tokens_input,
                "tokens_output": s.tokens_output,
                "recall": s.recall,
                "precision": s.precision,
                "faithfulness": s.faithfulness_score,
                "relevancy": s.relevancy_score,
            }
            for s in recent
        ]

    def summary(self) -> Dict[str, Any]:
        """返回汇总统计。

        Returns:
            {
                "total_requests": 总请求数,
                "avg_latency": 平均耗时,
                "p50_latency": 中位耗时,
                "p99_latency": P99 耗时,
                "avg_recall": 平均召回,
                "avg_precision": 平均精确率,
                "avg_faithfulness": 平均忠实度,
                "avg_relevancy": 平均切题度,
            }
        """
        with self._write_lock:
            history = list(self._history)

        if not history:
            return {"total_requests": 0}

        latencies = sorted(s.latency_seconds for s in history)
        recalls = [s.recall for s in history if s.recall is not None]
        precisions = [s.precision for s in history if s.precision is not None]
        faiths = [s.faithfulness_score for s in history if s.faithfulness_score is not None]
        relevs = [s.relevancy_score for s in history if s.relevancy_score is not None]

        def _avg(vals: List[float]) -> Optional[float]:
            return round(sum(vals) / len(vals), 4) if vals else None

        def _p(p: float) -> Optional[float]:
            if not latencies:
                return None
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return round(latencies[idx], 4)

        return {
            "total_requests": len(history),
            "avg_latency": _avg(latencies),
            "p50_latency": _p(0.50),
            "p99_latency": _p(0.99),
            "avg_recall": _avg(recalls),
            "avg_precision": _avg(precisions),
            "avg_faithfulness": _avg(faiths),
            "avg_relevancy": _avg(relevs),
        }

    def clear(self) -> None:
        """清空所有历史记录。"""
        with self._write_lock:
            count = len(self._history)
            self._history.clear()
        logger.info("指标历史已清空: %d 条", count)
