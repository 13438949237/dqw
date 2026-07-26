# -*- coding: utf-8 -*-
"""
多源数据连接器抽象基类。

所有连接器均需实现 load_documents() 方法，返回 langchain_core.documents.Document 列表。
每个 Document 须注入统一的来源元数据。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import List

from langchain_core.documents import Document


class BaseConnector(ABC):
    """数据源连接器抽象基类。

    子类必须实现 load_documents()，返回携带来源元数据的 Document 列表。
    """

    def __init__(self, source_label: str) -> None:
        """
        Args:
            source_label: 人类可读的数据源标识（如文件名、表名、API 端点）。
        """
        self.source_label = source_label

    @abstractmethod
    def load_documents(self) -> List[Document]:
        """从数据源加载文档。

        Returns:
            携带元数据的 LangChain Document 列表。
        """
        ...

    def _base_metadata(self) -> dict:
        """生成所有连接器共享的基础元数据。"""
        return {
            "source": self.source_label,
            "connector_type": self.__class__.__name__,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        }
