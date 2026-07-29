# -*- coding: utf-8 -*-
"""
检索管线编排 —— 对外暴露统一的 retrieve() 接口。

流程::

    用户查询
        │
        ▼
    HybridRetriever（稠密 + 稀疏 + 自查询 → RRF 融合 Top-N 候选）
        │
        ▼
    Reranker（Cross-Encoder 精排 → Top-K 结果）
        │
        ▼
    List[Document]（携带 qdrant_score / rrf_score / rerank_score）
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.embeddings.vector_store import QdrantVectorStore
from src.retrievers.hybrid_retriever import HybridRetriever
from src.rerankers.reranker import Reranker
from src.utils.config import get_config

logger = logging.getLogger(__name__)

# ── 模块级单例缓存 ─────────────────────────────────────────────────────────
_store: Optional[QdrantVectorStore] = None
_retriever: Optional[HybridRetriever] = None
_reranker: Optional[Reranker] = None


def _get_store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore()
    return _store


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        cfg = get_config().retrieval
        _retriever = HybridRetriever(
            vector_store=_get_store(),
            top_k=cfg.top_k * 10,  # 候选池 = 最终 top_k 的 10 倍
        )
    return _retriever


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


# ── 主入口 ─────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: Optional[int] = None,
    candidate_top_n: Optional[int] = None,
    use_reranker: bool = True,
) -> List[Document]:
    """端到端检索接口。

    Args:
        query:            用户自然语言查询。
        top_k:            最终返回结果数，None 则使用 config.yaml 默认值。
        candidate_top_n:  混合检索候选集上限。
        use_reranker:     是否启用 Cross-Encoder 重排序。

    Returns:
        携带完整分值的 Document 列表。

    Example:
        >>> docs = retrieve("Qdrant 分片策略")
        >>> for d in docs:
        ...     print(d.metadata.get("rerank_score"), d.page_content[:50])
    """
    cfg = get_config().retrieval
    top_k = top_k or get_config().models.reranker.top_k
    candidate_top_n = candidate_top_n or cfg.top_k

    # ── 1. 多路混合检索 ────────────────────────────────────────────────
    retriever = HybridRetriever(
        vector_store=_get_store(),
        top_k=candidate_top_n,
    )
    candidates = retriever.retrieve(query)

    if not candidates:
        logger.warning("未检索到任何结果: %s", query)
        return []

    logger.info(
        "混合检索: 候选 %d 条 → 准备精排",
        len(candidates),
    )

    # ── 2. 重排序 ──────────────────────────────────────────────────────
    if use_reranker:
        reranker = _get_reranker()
        results = reranker.rerank(query, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    # ── 3. 清理内部标记字段 ────────────────────────────────────────────
    for doc in results:
        doc.metadata.pop("rrf_score", None)
        doc.metadata.pop("rrf_rank", None)

    logger.info(
        "检索管线完成: query='%s' → %d 条结果：%s (rerank=%s)",
        query[:60],
        len(results),
        results,
        use_reranker,
    )

    return results


# ── 便捷接口 ────────────────────────────────────────────────────────────────

def retrieve_with_metadata(
    query: str,
    top_k: Optional[int] = None,
    **kwargs,
) -> List[Document]:
    """检索并保留完整元数据（含 rrf_score / rerank_score）。"""
    docs = retrieve(query, top_k=top_k, **kwargs)
    return docs


def batch_retrieve(
    queries: List[str],
    top_k: Optional[int] = None,
    **kwargs,
) -> List[List[Document]]:
    """批量检索（串行）。"""
    return [retrieve(q, top_k=top_k, **kwargs) for q in queries]
