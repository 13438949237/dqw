# -*- coding: utf-8 -*-
"""
文件系统连接器 —— 支持本地目录、单文件、文件路径列表。
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from docx import Document as DocxDocument
from openpyxl import load_workbook


from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS: set = {".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv"}

# 文本类扩展名（直接用 TextLoader）
_TEXT_EXTENSIONS: set = {".txt", ".md", ".csv"}


# ── 各格式加载函数 ────────────────────────────────────────────────────────


def _load_text(file_path: Path, encoding: str = "utf-8") -> List[Document]:
    loader = TextLoader(str(file_path), encoding=encoding, autodetect_encoding=True)
    return loader.load()


def _load_pdf(file_path: Path) -> List[Document]:
    return PyPDFLoader(str(file_path)).load()


def _load_docx(file_path: Path) -> List[Document]:
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []
    return [Document(page_content="\n".join(paragraphs))]


def _load_xlsx(file_path: Path) -> List[Document]:
    wb = load_workbook(str(file_path), read_only=True, data_only=True)
    all_text: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                rows.append("\t".join(cells))
        if rows:
            all_text.append(f"--- Sheet: {sheet_name} ---\n" + "\n".join(rows))
    wb.close()
    if not all_text:
        return []
    return [Document(page_content="\n\n".join(all_text))]


def _load_csv(file_path: Path, encoding: str = "utf-8") -> List[Document]:
    rows: list[str] = []
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if any(cell.strip() for cell in row):
                    rows.append("\t".join(row))
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if any(cell.strip() for cell in row):
                    rows.append("\t".join(row))
    if not rows:
        return []
    return [Document(page_content="\n".join(rows))]


# 扩展名 → 加载函数映射
_LOADER_MAP: Dict[str, callable] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".xlsx": _load_xlsx,
}


def _load_file_by_extension(file_path: Path, encoding: str = "utf-8") -> List[Document]:
    """根据文件扩展名选择合适的加载器。"""
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(file_path, encoding=encoding)

    if suffix in _LOADER_MAP:
        return _LOADER_MAP[suffix](file_path)

    return _load_text(file_path, encoding=encoding)


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
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.warning("不支持的文件类型 %s，跳过: %s", suffix, file_path)
            return []

        try:
            docs = _load_file_by_extension(file_path, encoding=self._encoding)
        except Exception:
            logger.exception("读取文件失败: %s", file_path)
            return []

        for doc in docs:
            doc.metadata.update(self._base_metadata())
            doc.metadata["source"] = file_path.name
            doc.metadata["file_path"] = str(file_path)
            doc.metadata["file_size"] = file_path.stat().st_size
            doc.metadata["file_type"] = suffix.lstrip(".")
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
