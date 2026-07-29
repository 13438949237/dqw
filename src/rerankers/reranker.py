# -*- coding: utf-8 -*-
"""
重排序器封装 —— 对候选集逐条计算 query-document 相关性分数。

支持本地 CrossEncoder（如 bge-reranker-v2-m3）和云端 API（Cohere / Jina），
通过 ModelFactory.get_reranker() 统一获取后端实例。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.llms.models import BaseReranker, ModelFactory
from src.utils.config import get_config

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器。

    Usage::

        reranker = Reranker()
        ranked = reranker.rerank("性能优化", candidate_docs, top_k=10)
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            provider:   重排序后端，None 则使用 config.yaml 默认值。
            model_name: 模型名称，None 则使用 config.yaml 默认值。
            top_k:      默认返回数量，None 则使用 config.yaml 默认值。
            score_threshold: 最低相关度阈值，低于此值的文档将被过滤。
        """
        cfg = get_config()
        self._provider = provider or cfg.models.reranker.provider
        self._model_name = model_name or cfg.models.reranker.model_name
        self._default_top_k = top_k or cfg.models.reranker.top_k
        self._score_threshold = (
            score_threshold
            if score_threshold is not None
            else cfg.models.reranker.score_threshold
        )
        self._model: Optional[BaseReranker] = None

    @property
    def model(self) -> BaseReranker:
        """懒加载重排序模型实例。"""
        if self._model is None:
            self._model = ModelFactory.get_reranker(
                provider=self._provider,
                model_name=self._model_name,
                top_k=self._default_top_k,
            )
        return self._model

    # ── 主入口 ────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """对候选文档集重新排序。

        Args:
            query:     用户查询字符串。
            documents: 候选 Document 列表。
            top_k:     最终返回数量，None 则使用默认值。

        Returns:
            按相关性分数降序排列的 Document 列表，meta 中额外包含 rerank_score。
            低于 score_threshold 的文档会被过滤。
        """
        if not documents:
            return []

        top_k = top_k or self._default_top_k
        texts = [doc.page_content for doc in documents]

        try:
            ranked_pairs: List[tuple] = self.model.rerank(
                query=query,
                documents=texts,
                top_k=min(top_k, len(documents)),
            )
        except Exception as exc:
            logger.error("重排序失败: %s，返回原始排序", exc)
            return documents[:top_k]

        # 按秩重新排列并注入分值
        results: List[Document] = []
        filtered_count = 0
        for idx, score in ranked_pairs:
            doc = documents[idx]
            score_val = round(float(score), 6)

            if score_val < self._score_threshold:
                filtered_count += 1
                logger.info("重排序得分低于阈值的检索结果: %s", doc.page_content)
                continue

            doc.metadata["rerank_score"] = score_val
            doc.metadata["rerank_rank"] = len(results)
            results.append(doc)


        if filtered_count > 0:
            logger.info(
                "阈值过滤: %d 条文档低于阈值 %.4f 被丢弃",
                filtered_count,
                self._score_threshold,
            )

        logger.info(
            "重排序完成: %d 条候选 → %d 条精排 (model=%s/%s)",
            len(documents),
            len(results),
            self._provider,
            self._model_name,
        )

        return results

    # ── 便捷批处理 ────────────────────────────────────────────────────

    def batch_rerank(
        self,
        queries: List[str],
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[List[Document]]:
        """对多个查询分别执行重排序（串行）。

        Args:
            queries:   用户查询列表。
            documents: 共享的候选 Document 列表。
            top_k:     每个查询的返回数量。

        Returns:
            每个查询对应的重排序结果列表。
        """
        return [self.rerank(q, documents, top_k) for q in queries]
