# -*- coding: utf-8 -*-
"""
混合检索器 —— 多路并行召回 + RRF 融合。

三路并行检索：
1. 稠密检索（语义相似度）
2. 稀疏检索（BM25 关键词）
3. 自查询检索（LLM 将自然语言转为 Qdrant filter 后检索）

去重后使用 RRF（Reciprocal Rank Fusion）算法融合排序，输出 Top-N 候选集。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.embeddings.vector_store import QdrantVectorStore
from src.llms.models import ModelFactory
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """多路混合检索器。

    Usage::

        store = QdrantVectorStore()
        retriever = HybridRetriever(store)
        candidates = retriever.retrieve("性能优化方案")
    """

    # ── RRF 常量 ────────────────────────────────────────────────────────
    RRF_K = 60  # RRF 平滑常数

    # ── 自查询 LLM 提示词 ─────────────────────────────────────────────
    SELF_QUERY_PROMPT = (
        "你是一个查询解析器。根据用户问题，生成 Qdrant 的元数据过滤条件。\n"
        "已知的可过滤字段包括：\n"
        "  - source: 文件名，如 \"search_architecture.md\"\n"
        "  - permission_tags: 权限标签列表，如 [\"public\", \"internal\", \"confidential\"]\n"
        "  - section_title: 章节标题，如 \"核心选型\"\n"
        "  - file_type: 文件类型，如 \"md\", \"pdf\", \"docx\"\n"
        "  - connector_type: 连接器类型，如 \"FileConnector\", \"MessageConnector\"\n"
        "  - page: 页码（整数）\n\n"
        "请输出**纯 JSON**，格式如下（不需要任何解释或 markdown 标记）：\n"
        "  {{\"must\": [{{\"key\": \"字段名\", \"match\": {{\"value\": \"精确值\"}}}}]}}\n"
        "  {{\"must\": [{{\"key\": \"字段名\", \"match\": {{\"any\": [\"值1\", \"值2\"]}}}}]}}\n"
        "  {{\"must\": [{{\"key\": \"page\", \"range\": {{\"gte\": 1, \"lte\": 10}}}}]}}\n\n"
        "如果查询**不包含任何过滤意图**，请输出：{{}}\n\n"
        "用户查询: {query}\n"
    )

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        top_k: int = 100,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            vector_store: QdrantVectorStore 实例。
            top_k:        最终候选集的数量上限。
            weights:      三路检索的融合权重 {"dense": 1.0, "sparse": 1.0, "self_query": 0.8}
        """
        self._store = vector_store
        self._top_k = top_k
        self._weights = weights or {"dense": 1.0, "sparse": 1.0, "self_query": 0.8}
        self._llm = None  # 懒加载

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ModelFactory.get_llm(temperature=0)
        return self._llm

    # ── 主入口 ────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """执行多路检索并返回融合后的候选集。

        Args:
            query: 用户自然语言查询。

        Returns:
            去重 + RRF 融合后的 Top-N Document 列表。
        """
        if not query.strip():
            return []

        # ── 1. 并行三路检索 ──
        dense_docs = self._safe_retrieve("dense", query)
        sparse_docs = self._safe_retrieve("sparse", query)
        self_query_docs = self._safe_retrieve_self_query(query)

        logger.info(
            "多路检索完成: dense=%d, sparse=%d, self_query=%d",
            len(dense_docs),
            len(sparse_docs),
            len(self_query_docs),
        )

        # ── 2. RRF 融合 ──
        merged = self._rrf_fusion(
            routes={
                "dense": dense_docs,
                "sparse": sparse_docs,
                "self_query": self_query_docs,
            },
            weights=self._weights,
            top_k=self._top_k,
        )

        logger.info("RRF 融合: 候选集 %d 条", len(merged))
        return merged

    # ── 安全检索（单个路由） ──────────────────────────────────────────

    def _safe_retrieve(self, mode: str, query: str) -> List[Document]:
        """安全调用 QdrantVectorStore 的单路检索，异常时返回空列表。"""
        try:
            if mode == "dense":
                return self._store.retrieve_dense(query, top_k=self._top_k)
            elif mode == "sparse":
                return self._store.retrieve_sparse(query, top_k=self._top_k)
            else:
                return []
        except Exception as exc:
            logger.warning("%s 检索异常: %s", mode, exc)
            return []

    def _safe_retrieve_self_query(self, query: str) -> List[Document]:
        """自查询检索：LLM 生成 filter → 带过滤 dense 检索。"""
        try:
            filter_cond = self._generate_filter(query)
            if not filter_cond:
                return []
            return self._store.retrieve_by_filter(
                query,
                filter_conditions=filter_cond,
                top_k=self._top_k,
                search_mode="dense",
            )
        except Exception as exc:
            logger.warning("自查询检索异常: %s", exc)
            return []

    # ── 自查询 filter 生成 ─────────────────────────────────────────────

    def _generate_filter(self, query: str) -> Optional[dict]:
        """使用 LLM 将自然语言查询转化为 Qdrant 过滤条件。"""
        try:
            prompt = self.SELF_QUERY_PROMPT.format(query=query)
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        except Exception as exc:
            logger.warning("自查询 LLM 调用失败: %s", exc)
            return None

        # 清理 LLM 可能返回的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("自查询 JSON 解析失败: %s", raw[:200])
            return None

        # 空对象 = 无过滤意图
        if not parsed or not parsed.get("must"):
            logger.debug("自查询：无过滤意图")
            return None

        logger.debug("自查询 filter: %s", parsed)
        return parsed

    # ── RRF 融合 ───────────────────────────────────────────────────────

    @classmethod
    def _rrf_fusion(
        cls,
        routes: Dict[str, List[Document]],
        weights: Dict[str, float],
        top_k: int,
    ) -> List[Document]:
        """Reciprocal Rank Fusion 多路结果融合。

        算法：
            RRF(d) = sum( w_i / (k + rank_i(d)) )

        其中 k=RRF_K 是平滑常数，w_i 是第 i 路检索的权重。
        """
        # 构建 doc_id → (doc, rrf_score) 映射
        doc_map: Dict[str, Tuple[Document, float]] = {}

        for route_name, docs in routes.items():
            weight = weights.get(route_name, 1.0)
            for rank, doc in enumerate(docs):
                qdrant_id = doc.metadata.get("qdrant_id", "")
                if not qdrant_id:
                    # 无 qdrant_id 时用 page_content hash 做去重
                    qdrant_id = str(hash(doc.page_content))

                rrf_score = weight / (cls.RRF_K + rank + 1)

                if qdrant_id in doc_map:
                    _, existing_score = doc_map[qdrant_id]
                    doc_map[qdrant_id] = (doc, existing_score + rrf_score)
                else:
                    doc_map[qdrant_id] = (doc, rrf_score)

        # 按 RRF 分数降序，取 top_k
        sorted_docs = sorted(
            doc_map.values(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        result = []
        for doc, rrf_score in sorted_docs:
            doc.metadata["rrf_score"] = round(rrf_score, 6)
            doc.metadata["rrf_rank"] = len(result)
            result.append(doc)

        return result
