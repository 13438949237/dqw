# -*- coding: utf-8 -*-
"""
Document processing pipeline orchestrator.

load_and_chunk()            — 单连接器：加载 → 解析 → 智能分块
parallel_load_and_chunk()   — 多文件并行：线程池解析 → 去重 → 质量校验
"""
from __future__ import annotations

import hashlib
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document

from src.connectors.base import BaseConnector
from src.parsers.semantic_chunker import SemanticChunker
from src.parsers.smart_parser import parse_file, filter_chunks

logger = logging.getLogger(__name__)

_STRUCTURED_METADATA_KEYS = frozenset({
    "page",
    "start_page",
    "end_page",
    "page_range",
    "total_pages",
    "section_title",
    "heading_stack",
})


def _merge_connector_metadata(
    chunks: List[Document],
    connector_metadata: Dict,
) -> None:
    """Merge loader metadata without overwriting parser page structure."""
    connector_meta = {
        key: value
        for key, value in connector_metadata.items()
        if key != "page_content"
    }
    for chunk in chunks:
        structured = {
            key: chunk.metadata[key]
            for key in _STRUCTURED_METADATA_KEYS
            if key in chunk.metadata
        }
        chunk.metadata.update(connector_meta)
        chunk.metadata.update(structured)


def compute_file_hash(file_path: str) -> str:
    """计算文件 SHA256 指纹（取前 16 位，足够去重）。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()[:16]

def _parse_and_chunk_single(
    file_path: str,
    source_name: str,
) -> Tuple[str, List[Document], str]:
    """解析并分块单个文件（线程安全，供并行调用）。

    Returns:
        (source_name, chunks, file_hash)
    """
    file_hash = compute_file_hash(file_path)
    chunker = SemanticChunker(file_path=file_path)

    connector = _SimpleFileConnector(file_path)
    docs = connector.load_documents()
    if not docs:
        return source_name, [], file_hash

    doc = docs[0]
    try:
        parsed = parse_file(file_path)
        parsed.extra["file_path"] = file_path
        parsed.extra["last_modified"] = doc.metadata.get("last_modified", "")
        chunks = chunker.chunk_parsed_doc(parsed)
    except (ValueError, ImportError) as exc:
        logger.warning("Smart parse failed (%s), fallback: %s", file_path, exc)
        chunks = chunker.chunk_document(doc)

    chunks = filter_chunks(chunks)

    _merge_connector_metadata(chunks, doc.metadata)
    for c in chunks:
        c.metadata["file_hash"] = file_hash

    return source_name, chunks, file_hash

class _SimpleFileConnector:
    """轻量级单文件加载器，避免引入完整 FileConnector 的目录扫描开销。"""
    def __init__(self, file_path: str):
        self._path = Path(file_path)

    def load_documents(self) -> List[Document]:
        from src.connectors.file_connector import FileConnector
        fc = FileConnector(str(self._path))
        return fc.load_documents()



# ── 单连接器管线（保持向后兼容） ─────────────────────────────────────────

def load_and_chunk(
    connector: BaseConnector,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    smart_parse: bool = True,
    min_chars: int = 20,
    max_ws_ratio: float = 0.6,
) -> List[Document]:
    """Load documents via connector, parse, and chunk with quality filtering.

    Args:
        connector:     Connector with load_documents().
        chunk_size:    Override config default chunk size.
        chunk_overlap: Override config default chunk overlap.
        smart_parse:   Use smart_parser for structured parsing.
        min_chars:      Min chars per chunk (shorter chunks discarded).
        max_ws_ratio:   Max whitespace ratio (noisy chunks discarded).
    """
    docs = connector.load_documents()
    if not docs:
        logger.warning("Connector returned no documents")
        return []
    logger.info("Pipeline start: %d documents", len(docs))

    def _get_chunker(file_path: str) -> SemanticChunker:
        return SemanticChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path=file_path,
        )

    all_chunks: List[Document] = []
    processed_files: Set[str] = set()

    for doc in docs:
        file_path = doc.metadata.get("file_path", "")

        if smart_parse and file_path:
            if file_path in processed_files:
                continue
            processed_files.add(file_path)
            chunker = _get_chunker(file_path)
            try:
                parsed = parse_file(file_path)
                parsed.extra["file_path"] = file_path
                parsed.extra["last_modified"] = doc.metadata.get("last_modified", "")
                chunks = chunker.chunk_parsed_doc(parsed)
            except (ValueError, ImportError) as exc:
                logger.warning("Smart parse failed (%s), fallback: %s", file_path, exc)
                chunks = chunker.chunk_document(doc)
        else:
            chunker = _get_chunker(file_path)
            chunks = chunker.chunk_document(doc)

        chunks = filter_chunks(chunks, min_chars=min_chars, max_whitespace_ratio=max_ws_ratio)

        _merge_connector_metadata(chunks, doc.metadata)

        all_chunks.extend(chunks)

    total = len(all_chunks)
    for idx, c in enumerate(all_chunks):
        c.metadata["global_chunk_index"] = idx
        c.metadata["global_total_chunks"] = total

    logger.info("Pipeline done: %d docs -> %d chunks", len(docs), total)
    return all_chunks



# ── 并行批量管线（后端线程池） ─────────────────────────────────────────────

def parallel_load_and_chunk(
    file_paths: List[str],
    source_names: Optional[List[str]] = None,
    max_workers: int = 4,
    progress_callback: Optional[Callable] = None,
    skip_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Document], List[Dict[str, str]]]:
    """后端并行解析管线：线程池并行解析 + 文件指纹去重 + 质量校验。

    Args:
        file_paths:        待处理的本地文件路径列表。
        source_names:      对应的源文件名（展示用），None 时从路径提取。
        max_workers:       并行线程数（建议 2~4）。
        progress_callback: 进度回调 callback(current, total, filename)。
        skip_hashes:       已入库文件的哈希集合，命中则跳过。

    Returns:
        (chunks, file_reports)
        chunks       — 合并后的 Document 列表。
        file_reports — 每个文件的处理结果：
            [{"source": "xx.pdf", "status": "success", "chunks": 23, "hash": "a1b2..."},
             {"source": "yy.pdf", "status": "skipped", "reason": "文件内容未变化"}]
    """
    if not file_paths:
        return [], []

    if source_names is None:
        source_names = [Path(p).name for p in file_paths]

    file_reports: List[Dict[str, str]] = []
    pending: List[Tuple[str, str]] = []

    for fp, sn in zip(file_paths, source_names):
        if skip_hashes:
            try:
                fh = compute_file_hash(fp)
                if fh in skip_hashes:
                    file_reports.append({
                        "source": sn, "status": "skipped",
                        "reason": "文件内容未变化（SHA256 匹配）", "hash": fh,
                    })
                    continue
            except OSError:
                pass
        pending.append((fp, sn))

    if not pending:
        logger.info("所有文件均已入库，跳过解析")
        return [], file_reports

    total = len(pending)
    all_chunks: List[Document] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as executor:
        futures = {
            executor.submit(_parse_and_chunk_single, fp, sn): (fp, sn)
            for fp, sn in pending
        }

        for future in as_completed(futures):
            fp, sn = futures[future]
            try:
                source_name, chunks, file_hash = future.result()
                all_chunks.extend(chunks)
                file_reports.append({
                    "source": source_name,
                    "status": "success",
                    "chunks": str(len(chunks)),
                    "hash": file_hash,
                })
                logger.info("解析完成: %s → %d chunks", source_name, len(chunks))
            except Exception as exc:
                logger.error("解析失败: %s — %s", sn, exc)
                file_reports.append({
                    "source": sn, "status": "failed",
                    "reason": str(exc), "hash": "",
                })

            completed += 1
            if progress_callback:
                progress_callback(completed, total, sn)

    chunk_total = len(all_chunks)
    for idx, c in enumerate(all_chunks):
        c.metadata["global_chunk_index"] = idx
        c.metadata["global_total_chunks"] = chunk_total

    logger.info(
        "=== 并行管线完成: %d 文件 → %d chunks (workers=%d) ===",
        total, chunk_total, max_workers,
    )
    return all_chunks, file_reports


def batch_load_and_chunk(connectors: List[BaseConnector], **kwargs) -> List[Document]:
    """Process multiple connectors and merge results."""
    result: List[Document] = []
    for conn in connectors:
        result.extend(load_and_chunk(conn, **kwargs))
    return result
