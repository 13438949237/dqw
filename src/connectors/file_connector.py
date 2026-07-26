# -*- coding: utf-8 -*-
"""
文件系统连接器 —— 支持本地目录、单文件、文件路径列表。
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class FileConnector(BaseConnector):
    """从本地文件系统加载文档。

    Usage::

        connector = FileConnector("data/samples", glob="**/*.pdf")
        docs = connector.load_documents()
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        glob: str = "**/*.txt",
        encoding: str = "utf-8",
        recursive: bool = True,
    ) -> None:
        """
        Args:
            path:      本地文件或目录路径（可传入列表）。
            glob:      文件匹配模式。
            encoding:  文本编码。
            recursive: 是否递归子目录。
        """
        self._paths = [path] if isinstance(path, str) else path
        self._glob = glob
        self._encoding = encoding
        self._recursive = recursive
        super().__init__(source_label=", ".join(self._paths))

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []

        for raw_path in self._paths:
            p = Path(raw_path).resolve()

            if p.is_file():
                documents.extend(self._load_single_file(p))
            elif p.is_dir():
                documents.extend(self._load_directory(p))
            else:
                logger.warning("路径不存在，跳过: %s", raw_path)

        logger.info("FileConnector: 共加载 %d 个文档", len(documents))
        return documents

    # ── 内部 ────────────────────────────────────────────────────────────

    def _load_single_file(self, file_path: Path) -> List[Document]:
        loader = TextLoader(
            str(file_path),
            encoding=self._encoding,
            autodetect_encoding=True,
        )
        try:
            docs = loader.load()
        except Exception:
            logger.exception("读取文件失败: %s", file_path)
            return []

        for doc in docs:
            doc.metadata.update(self._base_metadata())
            doc.metadata["source"] = file_path.name
            doc.metadata["file_path"] = str(file_path)
            doc.metadata["file_size"] = file_path.stat().st_size
            doc.metadata["last_modified"] = datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        return docs

    def _load_directory(self, dir_path: Path) -> List[Document]:
        loader = DirectoryLoader(
            path=str(dir_path),
            glob=self._glob if self._recursive else self._glob.lstrip("**/"),
            loader_cls=TextLoader,
            loader_kwargs={"encoding": self._encoding, "autodetect_encoding": True},
            show_progress=True,
            silent_errors=True,
        )
        docs = loader.load()

        for doc in docs:
            raw_source = doc.metadata.get("source", "")
            doc.metadata.update(self._base_metadata())
            doc.metadata["source"] = Path(raw_source).name
            doc.metadata["file_path"] = str(raw_source)
            try:
                st = Path(raw_source).stat()
                doc.metadata["file_size"] = st.st_size
                doc.metadata["last_modified"] = datetime.fromtimestamp(
                    st.st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                pass

        return docs
