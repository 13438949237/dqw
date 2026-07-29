# -*- coding: utf-8 -*-
"""
Qdrant 混合向量存储。

支持每个 chunk 同时持有四种向量表示：
- dense（稠密语义向量，通过 ModelFactory 嵌入生成）
- sparse（稀疏 BM25 向量，通过 FastEmbed Qdrant/bm25 模型生成）
- summary（长文本摘要向量，chunk > 1000 tokens 时由 LLM 生成摘要后嵌入）
- sentence（句子级细粒度向量，存入独立 collection，关联 parent_chunk_id）
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    SparseVectorParams,
    VectorParams,
)

from src.llms.models import ModelFactory
from src.utils.config import get_config

logger = logging.getLogger(__name__)

# ── 句子拆分正则 ──────────────────────────────────────────────────────────
_SENTENCE_PATTERN = re.compile(
    r"(?<=[。！？.!?\n])\s*"
)


class SparseEncoder:
    """稀疏向量编码器，基于 FastEmbed BM25 模型。

    FastEmbed 内置模型::

        "Qdrant/bm25"               - 内置 BM25 分词器（无需下载，中英文通用）
        "prithivida/Splade_PP_en_v1"  - 英文 SPLADE 神经稀疏模型（~300MB）

    使用示例::

        # 神经网络稀疏编码
        enc = SparseEncoder()
        enc.fit(corpus_texts)
        indices, values = enc.encode("查询文本")
    """

    # ── 内置模型列表 ──────────────────────────────────────────────────
    _BUILTIN_MODELS: ClassVar[Dict[str, str]] = {
        "bm25": "Qdrant/bm25",
        "splade": "prithivida/Splade_PP_en_v1",
    }

    # ── 构造函数 ─────────────────────────────────────────────────────

    def __init__(
        self,
        backend: str = "auto",
        model_name: Optional[str] = None,
        max_features: int = 10000,
    ) -> None:
        """初始化稀疏编码器。

        Args:
            model_name: FastEmbed 稀疏模型名，None 时使用 "Qdrant/bm25"。
        """
        self._model_name = model_name or self._BUILTIN_MODELS["bm25"]
        self._fast_model: Any = None
        self._fitted = False

    # ── 公开方法 ─────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "SparseEncoder":
        """拟合编码器（BM25 模式下无需拟合，保留接口以兼容调用方）。"""
        self._init_fastembed()
        self._fitted = True
        logger.info("稀疏编码器就绪: model=%s", self._model_name)
        return self

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        """将单条文本编码为 Qdrant 兼容的稀疏向量。

        Returns:
            (indices, values) —— 非零维度的索引列表和值列表。
        """
        if not self._fitted:
            logger.warning("编码器尚未拟合，返回空向量")
            return [], []

        return self._encode_fastembed(text)

    @property
    def backend(self) -> str:
        """返回当前实际使用的后端名称。"""
        return "fastembed"

    # ── 内部：FastEmbed 实现 ──────────────────────────────────────────

    def _init_fastembed(self) -> Any:
        if self._fast_model is None:
            try:
                from fastembed import SparseTextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "FastEmbed 后端需要安装 fastembed 包，请执行: pip install fastembed"
                ) from exc

            logger.info("加载 FastEmbed 稀疏模型: %s", self._model_name)
            self._fast_model = SparseTextEmbedding(model_name=self._model_name)
        return self._fast_model

    def _encode_fastembed(self, text: str) -> Tuple[List[int], List[float]]:
        if self._fast_model is None:
            self._init_fastembed()

        # FastEmbed 的 embed() 返回 List[SparseEmbedding]，每个元素有 .indices 和 .values
        results = list(self._fast_model.embed([text]))  # type: ignore[union-attr]
        if not results:
            return [], []

        result = results[0]
        # FastEmbed SparseEmbedding 有 .indices 和 .values 属性（numpy 数组）
        indices = result.indices.tolist() if hasattr(result.indices, "tolist") else list(result.indices)
        values = result.values.tolist() if hasattr(result.values, "tolist") else list(result.values)
        logger.info("FastEmbed 最终编码结果: indices=%s, values=%s", indices, values)
        return indices, values


# ============================================================================
# QdrantVectorStore
# ============================================================================


class QdrantVectorStore:
    """Qdrant 多向量存储。

    主 collection（chunks）同时持有密集向量和稀疏向量 即 dense + sparse 两种命名向量；
    子 collection（sentences）存储句子级嵌入。
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        sparse_model: Optional[str] = None,
    ) -> None:
        """
        Args:
            collection_name: 主 collection 名称，默认从 config 读取。
            host:            Qdrant 主机。
            port:            Qdrant HTTP 端口。
            sparse_model:    FastEmbed 稀疏模型名，None 则使用 "Qdrant/bm25"。
        """
        cfg = get_config()

        self._chunk_collection = collection_name or cfg.qdrant.collection_name
        self._sent_collection = f"{self._chunk_collection}_sentences"
        self._vector_size = cfg.models.embedding.dimension

        self._client = QdrantClient(
            host=host or cfg.qdrant.host,
            port=port or cfg.qdrant.port,
            # api_key=cfg.qdrant.api_key,
        )

        self._embedding_model = None  # 懒加载
        self._llm = None              # 懒加载
        self._sparse_encoder = SparseEncoder(
            model_name=sparse_model,
        )

        # 初始化 / 重建 collection
        self._init_collections()

        logger.info(
            "QdrantVectorStore 就绪: chunks=%s, sentences=%s, host=%s:%s",
            self._chunk_collection,
            self._sent_collection,
            cfg.qdrant.host,
            cfg.qdrant.port,
        )

    # ── Collection 管理 ─────────────────────────────────────────────────

    def _init_collections(self) -> None:
        """创建主 collection（dense + sparse）和句子 collection。

        如果 collection 已存在则直接复用，仅在不存在时创建。
        """
        collections = {c.name for c in self._client.get_collections().collections}

        # 主 collection
        if self._chunk_collection not in collections:
            self._client.create_collection(
                collection_name=self._chunk_collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    ),
                },
            )
            logger.info("已创建 collection: %s", self._chunk_collection)

        # 句子 collection（仅 dense）
        if self._sent_collection not in collections:
            self._client.create_collection(
                collection_name=self._sent_collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("已创建 collection: %s", self._sent_collection)

    # ── 懒加载子模型 ────────────────────────────────────────────────────

    @property
    def embedding(self):
        if self._embedding_model is None:
            self._embedding_model = ModelFactory.get_embedding()
        return self._embedding_model

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ModelFactory.get_llm()
        return self._llm

    # ── 添加文档 ───────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 32,
        generate_sentences: bool = True,
    ) -> List[str]:
        """批量入库文档，自动生成全部四种向量。

        Args:
            documents:          LangChain Document 列表。
            batch_size:         嵌入生成 / Qdrant upsert 批次大小。
            generate_sentences: 是否生成句子级向量（耗时操作）。

        Returns:
            所有入库 point 的 ID 列表。
        """
        if not documents:
            return []

        # 1. 拟合稀疏编码器
        texts = [doc.page_content for doc in documents]
        self._sparse_encoder.fit(texts)

        # 2. 逐文档生成向量与 point
        points: List[PointStruct] = []
        sentence_points: List[PointStruct] = []

        for idx, doc in enumerate(documents):
            point_id = str(uuid.uuid4())
            text = doc.page_content

            # ── dense 向量 ──
            dense_vec = self._get_dense_vector(text)

            # ── sparse 向量 ──
            sparse_indices, sparse_values = self._sparse_encoder.encode(text)

            # ── 摘要向量（长文本时） ──
            summary_text = ""
            if self._is_long(text):
                summary_text = self._generate_summary(text)
                # 如果成功生成摘要，用摘要向量替代原 dense 向量
                if summary_text:
                    dense_vec = self._get_dense_vector(summary_text)

            # ── 构建 payload ──
            payload = {
                **{k: v for k, v in doc.metadata.items()},
                "text": text,
                "has_summary": bool(summary_text),
                "summary_text": summary_text if summary_text else "",
                "char_count": len(text),
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    },
                    payload=payload,
                )
            )

            # ── 句子向量 ──
            if generate_sentences and text.strip():
                sps = self._build_sentence_points(point_id, text, doc.metadata)
                sentence_points.extend(sps)

            if (idx + 1) % 20 == 0:
                logger.info("向量生成进度: %d/%d", idx + 1, len(documents))

        # 3. 批量写入 Qdrant
        all_ids: List[str] = []
        for i in range(0, len(points), batch_size):
            batch = points[i: i + batch_size]
            self._client.upsert(
                collection_name=self._chunk_collection,
                points=batch,
                wait=True,
            )
            all_ids.extend([str(p.id) for p in batch])

        if sentence_points:
            for i in range(0, len(sentence_points), batch_size):
                batch = sentence_points[i: i + batch_size]
                self._client.upsert(
                    collection_name=self._sent_collection,
                    points=batch,
                    wait=True,
                )

        logger.info(
            "入库完成: chunks=%d, sentences=%d",
            len(points),
            len(sentence_points),
        )
        return all_ids

    # ── 检索方法 ───────────────────────────────────────────────────────

    def _ensure_sparse_fitted(self) -> None:
        """确保稀疏编码器已拟合。

        FastEmbed BM25 无需真实语料拟合，仅标记 _fitted=True。
        """
        if self._sparse_encoder._fitted:
            return

        self._sparse_encoder._fitted = True
        logger.info("稀疏编码器（fastembed BM25）已标记就绪")

    def retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """稠密语义检索。

        Args:
            query:          查询文本。
            top_k:          返回数量。
            score_threshold: 最低相似度阈值（None 则不过滤）。

        Returns:
            Document 列表，meta 中额外包含 qdrant_score 和 qdrant_id。
        """
        query_vec = self._get_dense_vector(query)

        results = self._client.query_points(
            collection_name=self._chunk_collection,
            query=query_vec,
            using="dense",
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True
        )
        logger.info("稠密检索完成，检索到：%d 条", len(results.points))

        return self._hits_to_documents(results.points)

    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """稀疏（BM25）关键词检索。

        Args:
            query: 查询文本。
            top_k: 返回数量。

        Returns:
            Document 列表。
        """
        self._ensure_sparse_fitted()
        indices, values = self._sparse_encoder.encode(query)
        if not indices:
            logger.warning("稀疏编码返回空向量，跳过检索: query='%s'", query[:50])
            return []

        logger.info(
            "发起稀疏检索: query='%s', indices_count=%d",
            query[:50],
            len(indices),
        )

        results = self._client.query_points(
            collection_name=self._chunk_collection,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        logger.info("稀疏检索完成，检索到：%d 条", len(results.points))

        return self._hits_to_documents(results.points)

    def retrieve_by_filter(
        self,
        query: str,
        filter_conditions: Dict[str, Any],
        top_k: int = 10,
        search_mode: str = "dense",
    ) -> List[Document]:
        """带元数据过滤的检索。

        Args:
            query:             查询文本。
            filter_conditions: 过滤条件字典。支持:
                {"source": "doc.txt"}                         → 精确匹配
                {"permission_tags": ["public", "internal"]}   → 任意匹配
                {"page": {"gte": 1, "lte": 10}}              → 范围过滤
            top_k:             返回数量。
            search_mode:       检索模式 "dense" | "sparse"。

        Returns:
            Document 列表。
        """
        logger.info("自查询发起检索: query='%s', filter=%s", query[:50], filter_conditions)
        qdrant_filter = self._build_filter(filter_conditions)
        logger.info("Qdrant 过滤条件: %s", qdrant_filter)

        if search_mode == "dense":
            query_vec = self._get_dense_vector(query)
            results = self._client.query_points(
                collection_name=self._chunk_collection,
                query=query_vec,
                using="dense",
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            indices, values = self._sparse_encoder.encode(query)
            if not indices:
                return []
            results = self._client.query_points(
                collection_name=self._chunk_collection,
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )

        return self._hits_to_documents(results.points)

    def retrieve_by_sentences(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Document]:
        """句子级检索 —— 从句子 collection 中找到最佳匹配句子，回父 chunk。

        Args:
            query: 查询文本。
            top_k: 返回的父 chunk 数量（会对同一父 chunk 的句子去重）。

        Returns:
            父 chunk 的 Document 列表。
        """
        query_vec = self._get_dense_vector(query)

        results = self._client.query_points(
            collection_name=self._sent_collection,
            query=query_vec,
            limit=top_k * 3,  # 多取一些避免去重后不足
            with_payload=True,
        )

        seen: set = set()
        parent_ids: List[str] = []
        for r in results.points:
            pid = r.payload.get("parent_chunk_id", "") if r.payload else ""
            if pid and pid not in seen:
                seen.add(pid)
                parent_ids.append(pid)
            if len(parent_ids) >= top_k:
                break

        if not parent_ids:
            return []

        # 回主 collection 按 id 获取完整 chunk
        retrieved = self._client.retrieve(
            collection_name=self._chunk_collection,
            ids=parent_ids,
            with_payload=True,
        )

        return self._hits_to_documents(retrieved)

    # ── 辅助方法 ───────────────────────────────────────────────────────

    def _get_dense_vector(self, text: str) -> List[float]:
        result = self.embedding.embed_query(text)
        if isinstance(result, np.ndarray):
            result = result.tolist()
        return result

    def _is_long(self, text: str) -> bool:
        """粗略估算 token 数 > 1000。中文 ≈ 字符数 / 1.5，英文 ≈ 字符数 / 4。"""
        # 简单启发式：超过 2000 字符即视为长文本
        return len(text) > 2000

    def _generate_summary(self, text: str) -> str:
        """调用 LLM 生成单句摘要。"""
        prompt = (
            "请用一句话总结以下文本的核心内容，不超过 80 个字。"
            "只输出摘要内容，不要任何前缀：\n\n"
            + text[:4000]  # 截断保护
        )
        try:
            from langchain_core.messages import HumanMessage
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            summary = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            logger.debug("摘要生成: %s...", summary[:60])
            return summary
        except Exception as exc:
            logger.warning("摘要生成失败: %s", exc)
            return ""

    def _build_sentence_points(
        self,
        parent_id: str,
        text: str,
        meta: dict,
    ) -> List[PointStruct]:
        """将文本拆分为句子并生成独立的 sentence points。"""
        raw_sentences = _SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]

        if len(sentences) < 2:
            return []  # 只有一句时没必要

        points: List[PointStruct] = []
        for i, sent in enumerate(sentences):
            # sid = f"{parent_id}_s{i}"
            sid = str(uuid.uuid4())
            vec = self._get_dense_vector(sent)
            points.append(
                PointStruct(
                    id=sid,
                    vector=vec,
                    payload={
                        "parent_chunk_id": parent_id,
                        "sentence_index": i,
                        "text": sent,
                        "source": meta.get("source", ""),
                    },
                )
            )

        return points

    @staticmethod
    def _build_filter(conditions: Dict[str, Any]) -> Optional[Filter]:
        """将 dict 条件转换为 Qdrant Filter 对象。
        支持两种格式：
        1. Qdrant 原生格式（LLM 自查询生成）：
           {"must": [{"key": "field", "match": {"value": "x"}}, ...]}
        2. 简化字典格式：
           {"field": "value", "tags": ["a", "b"], "page": {"gte": 1, "lte": 10}}
        """
        if not conditions:
            return None

        must_clauses: List[FieldCondition] = []

        if "must" in conditions and isinstance(conditions["must"], list):
            for clause in conditions["must"]:
                key = clause.get("key")
                match_info = clause.get("match")
                range_info = clause.get("range")

                if not key:
                    continue

                if match_info:
                    if "value" in match_info:
                        must_clauses.append(
                            FieldCondition(key=key, match=MatchValue(value=match_info["value"]))
                        )
                    elif "any" in match_info:
                        must_clauses.append(
                            FieldCondition(key=key, match=MatchAny(any=match_info["any"]))
                        )
                elif range_info:
                    must_clauses.append(
                        FieldCondition(key=key, range=Range(**range_info))
                    )
        else:
            for key, value in conditions.items():
                if isinstance(value, list):
                    must_clauses.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                    must_clauses.append(
                        FieldCondition(key=key, range=Range(**value))
                    )
                else:
                    must_clauses.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        return Filter(must=must_clauses) if must_clauses else None

    @staticmethod
    def _hits_to_documents(
        hits: list,
    ) -> List[Document]:
        """将 Qdrant 返回的 hit 转换为 LangChain Document。"""
        docs: List[Document] = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            meta = {k: v for k, v in payload.items()}
            if hasattr(hit, "score"):
                meta["qdrant_score"] = hit.score
            if hasattr(hit, "id"):
                meta["qdrant_id"] = str(hit.id)
            docs.append(Document(page_content=text, metadata=meta))
        return docs

    # ── 管理方法 ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """清空所有 collection 数据并重建。"""
        collections = {c.name for c in self._client.get_collections().collections}
        if self._chunk_collection in collections:
            self._client.delete_collection(self._chunk_collection)
            logger.info("已删除 collection: %s", self._chunk_collection)
        if self._sent_collection in collections:
            self._client.delete_collection(self._sent_collection)
            logger.info("已删除 collection: %s", self._sent_collection)
        import time
        for _ in range(30):
            remaining = {c.name for c in self._client.get_collections().collections}
            if self._chunk_collection not in remaining and self._sent_collection not in remaining:
                break
            time.sleep(0.5)
        self._init_collections()
        logger.info("文档库已清空并重建: %s, %s", self._chunk_collection, self._sent_collection)
# ... existing code ...

    def count(self) -> int:
        """返回主 collection 中的总向量数。"""
        info = self._client.get_collection(self._chunk_collection)
        return info.points_count

    def close(self) -> None:
        """关闭客户端连接。"""
        self._client.close()
