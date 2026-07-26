# -*- coding: utf-8 -*-
"""
数据库连接器 —— 通过 SQLAlchemy 读取关系型数据库中的文本字段。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.documents import Document

from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class DatabaseConnector(BaseConnector):
    """从关系型数据库表中读取文本列作为文档。

    Usage::

        connector = DatabaseConnector(
            connection_string="sqlite:///app.db",
            table_name="articles",
            text_columns=["title", "body"],
            id_column="id",
        )
        docs = connector.load_documents()
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        text_columns: List[str],
        id_column: str = "id",
        where_clause: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> None:
        """
        Args:
            connection_string: SQLAlchemy 连接串（如 sqlite:///db.sqlite3）。
            table_name:        目标表名。
            text_columns:      需要作为文档内容的文本列名列表。
            id_column:          用作 doc_id 的主键列。
            where_clause:      可选 SQL WHERE 条件（不含 WHERE 关键字）。
            limit:              最大返回行数。
        """
        self._conn_str = connection_string
        self._table = table_name
        self._text_cols = text_columns
        self._id_col = id_column
        self._where = where_clause
        self._limit = limit
        super().__init__(source_label=f"db://{table_name}")

    def load_documents(self) -> List[Document]:
        try:
            from sqlalchemy import create_engine, text
        except ImportError as exc:
            raise ImportError(
                "DatabaseConnector 需要安装 sqlalchemy，请执行: pip install sqlalchemy"
            ) from exc

        engine = create_engine(self._conn_str)
        documents: List[Document] = []

        try:
            with engine.connect() as conn:
                sql = self._build_query()
                logger.info("执行查询: %s", sql)
                result = conn.execute(text(sql))

                columns = list(result.keys())
                for row in result:
                    row_dict = dict(zip(columns, row))

                    # 拼接多列文本为单文档
                    text_parts = [
                        str(row_dict.get(c, ""))
                        for c in self._text_cols
                        if row_dict.get(c)
                    ]
                    page_content = "\n".join(text_parts)
                    if not page_content.strip():
                        continue

                    meta = self._base_metadata()
                    meta["source"] = f"{self._table}:{row_dict.get(self._id_col, '')}"
                    meta["table_name"] = self._table
                    meta["row_id"] = str(row_dict.get(self._id_col, ""))
                    meta["ingestion_timestamp"] = datetime.now(
                        timezone.utc
                    ).isoformat()

                    documents.append(
                        Document(page_content=page_content, metadata=meta)
                    )
        finally:
            engine.dispose()

        logger.info(
            "DatabaseConnector: 从表 %s 加载了 %d 个文档",
            self._table,
            len(documents),
        )
        return documents

    def _build_query(self) -> str:
        cols = ", ".join([self._id_col] + self._text_cols)
        sql = f"SELECT {cols} FROM {self._table}"
        if self._where:
            sql += f" WHERE {self._where}"
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql
