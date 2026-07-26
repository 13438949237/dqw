# -*- coding: utf-8 -*-
"""
文档处理管线编排。

load_and_chunk() 一键完成 加载 → 解析 → 智能分块 全流程。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.connectors.base import BaseConnector
from src.parsers.semantic_chunker import SemanticChunker, chunk_with_config
from src.parsers.smart_parser import parse_file
from src.utils.config import get_config

logger = logging.getLogger(__name__)


def load_and_chunk(
    connector: BaseConnector,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    smart_parse: bool = True,
) -> List[Document]:
    """
    端到端加载与分块编排函数。

    流程::

        连接器加载文档
            │
            ▼
        [可选] 多格式智能解析（PDF/DOCX/MD/XLSX）
            │
            ▼
        语义感知分块（保留标题层级、表格/列表完整性）

    Args:
        connector:     实现了 load_documents() 的连接器实例。
        chunk_size:    分块大小，None 则使用 config.yaml 默认值。
        chunk_overlap: 分块重叠，None 则使用 config.yaml 默认值。
        smart_parse:   是否先通过 smart_parser 解析文件结构后再分块。
                       True 时适用于本地文件（FileConnector），
                       False 时适用于 DB/API/Message 连接器。

    Returns:
        携带完整元数据的 LangChain Document 列表。
    """
    # ── 1. 加载 ─────────────────────────────────────────────────────────
    logger.info("=== 管线启动 ===")
    docs = connector.load_documents()
    if not docs:
        logger.warning("连接器未返回任何文档")
        return []

    logger.info("连接器返回 %d 个文档", len(docs))

    # ── 2. 参数 ─────────────────────────────────────────────────────────
    cfg = get_config().document
    chunk_size = chunk_size or cfg.chunk_size
    chunk_overlap = chunk_overlap or cfg.chunk_overlap

    chunker = SemanticChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # ── 3. 解析 + 分块 ──────────────────────────────────────────────────
    all_chunks: List[Document] = []

    for doc in docs:
        file_path = doc.metadata.get("file_path", "")

        if smart_parse and file_path:
            # 经过 smart_parser 的结构化解析
            try:
                parsed = parse_file(file_path)
                parsed.extra["file_path"] = file_path
                parsed.extra["last_modified"] = doc.metadata.get(
                    "last_modified", ""
                )
                chunks = chunker.chunk_parsed_doc(parsed)
            except (ValueError, ImportError) as exc:
                logger.warning(
                    "智能解析失败 (%s)，回退到自动检测分块: %s",
                    file_path,
                    exc,
                )
                chunks = chunker.chunk_document(doc)
        else:
            # 直接自动检测结构分块
            chunks = chunker.chunk_document(doc)

        # 合并连接器元数据
        connector_meta = {
            k: v
            for k, v in doc.metadata.items()
            if k not in ("page_content",)
        }
        for chunk in chunks:
            chunk.metadata.update(connector_meta)

        all_chunks.extend(chunks)

    # ── 4. 全局索引 ─────────────────────────────────────────────────────
    total = len(all_chunks)
    for idx, chunk in enumerate(all_chunks):
        chunk.metadata["global_chunk_index"] = idx
        chunk.metadata["global_total_chunks"] = total

    logger.info(
        "=== 管线完成: %d 个文档 → %d 个 chunk (size=%d, overlap=%d) ===",
        len(docs),
        total,
        chunk_size,
        chunk_overlap,
    )
    return all_chunks


def batch_load_and_chunk(
    connectors: List[BaseConnector],
    **kwargs,
) -> List[Document]:
    """批量处理多个连接器，合并返回。

    Args:
        connectors: 连接器实例列表。
        **kwargs:   传递给 load_and_chunk() 的额外参数。

    Returns:
        合并后的 Document 列表。
    """
    all_chunks: List[Document] = []
    for connector in connectors:
        all_chunks.extend(load_and_chunk(connector, **kwargs))
    return all_chunks
