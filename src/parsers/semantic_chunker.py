# -*- coding: utf-8 -*-
"""
语义感知文档分块器。

在常规字符切分的基础上，识别并保留文档语义结构：
- 标题层级（H1 / H2 / H3 …），每个 chunk 记录当前所属章节。
- 表格与列表视为不可拆分原子，单独成块或附着于前一语义块。
- 段落优先在自然边界处切分。
- 注入丰富元数据：doc_id、页码、章节标题、文件修改时间、权限标签。
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.parsers.smart_parser import ParsedBlock, ParsedDocument, parse_file
from src.utils.config import get_config

logger = logging.getLogger(__name__)

# ── 权限标签虚构池（用于 demo 场景）──────────────────────────────────────
_PERMISSION_POOL = [
    ["public"],
    ["internal"],
    ["internal", "restricted"],
    ["confidential"],
    ["confidential", "legal"],
    ["confidential", "hr"],
    ["public", "marketing"],
]


class SemanticChunker:
    """语义感知分块器。

    支持两种使用模式：

    1. chunk_parsed_doc():  输入已解析的 ParsedDocument，利用解析器提供的
                             block_type / heading_level / page 信息。
    2. chunk_document():    输入原始 langchain Document，内部自动检测
                            Markdown 类标题、表格、列表结构。

    chunk_size/chunk_overlap 未显式传入时，会按 file_path 的文件类型读取
    config.yaml 中 document.per_type 的配置。
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            chunk_size:    每个分块的最大字符数。None 时按文件类型从
                           config.yaml 的 document.per_type 获取。
            chunk_overlap: 相邻分块之间的重叠字符数。None 时按文件类型从
                           config.yaml 的 document.per_type 获取。
            file_path:     用于确定文件类型并读取 per_type 配置的路径。
        """
        resolved_size, resolved_overlap = self._resolve_chunk_params(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            file_path=file_path,
        )
        self._chunk_size = resolved_size
        self._chunk_overlap = resolved_overlap

    @staticmethod
    def _resolve_chunk_params(
        chunk_size: Optional[int],
        chunk_overlap: Optional[int],
        file_path: Optional[str],
    ) -> Tuple[int, int]:
        """Resolve explicit overrides or config-driven per-type defaults."""
        cfg = get_config().document
        suffix = Path(file_path).suffix.lower() if file_path else ""
        configured_size, configured_overlap = cfg.get_chunk_params(suffix)
        return (
            chunk_size if chunk_size is not None else configured_size,
            chunk_overlap if chunk_overlap is not None else configured_overlap,
        )

    @staticmethod
    def _table_rows(content: str) -> List[str]:
        return [line.strip() for line in content.splitlines() if line.strip()]

    @staticmethod
    def _table_column_count(content: str) -> int:
        rows = SemanticChunker._table_rows(content)
        if not rows:
            return 0
        return max(row.count(" | ") + 1 for row in rows)

    @staticmethod
    def _block_end_page(block: ParsedBlock) -> int:
        extra = block.extra or {}
        end_page = extra.get("table_end_page", extra.get("end_page", block.page))
        return int(end_page) if end_page else block.page

    @staticmethod
    def _merge_cross_page_tables(blocks: List[ParsedBlock]) -> List[ParsedBlock]:
        """Merge consecutive same-column table fragments on adjacent pages."""
        merged: List[ParsedBlock] = []
        for block in blocks:
            if (
                block.block_type == "table"
                and merged
                and merged[-1].block_type == "table"
                and block.page == SemanticChunker._block_end_page(merged[-1]) + 1
                and SemanticChunker._table_column_count(block.content)
                == SemanticChunker._table_column_count(merged[-1].content)
            ):
                previous = merged.pop()
                previous_rows = SemanticChunker._table_rows(previous.content)
                current_rows = SemanticChunker._table_rows(block.content)
                if (
                    previous_rows
                    and current_rows
                    and previous_rows[0] == current_rows[0]
                ):
                    current_rows = current_rows[1:]

                start_page = previous.page
                end_page = block.page
                merged.append(
                    ParsedBlock(
                        block_type="table",
                        content="\n".join(previous_rows + current_rows),
                        page=start_page,
                        extra={
                            **previous.extra,
                            "table_start_page": start_page,
                            "table_end_page": end_page,
                        },
                    )
                )
            else:
                merged.append(block)
        return merged

    # ── 模式一：输入 ParsedDocument ────────────────────────────────────

    def chunk_parsed_doc(self, parsed: ParsedDocument) -> List[Document]:
        """对已解析的结构化文档进行语义分块。

        Args:
            parsed: 从 smart_parser.parse_file() 获得的中间表示。

        Returns:
            携带完整元数据的 LangChain Document 列表。
        """
        if not parsed.blocks:
            logger.warning("文档无内容块: %s", parsed.doc_id)
            return []

        # 构建标题栈，追踪当前所在的章节
        heading_stack: List[Tuple[int, str]] = []  # [(level, title), ...]
        current_chunk: List[ParsedBlock] = []
        current_len = 0
        current_page: Optional[int] = None
        chunks: List[Document] = []

        blocks = self._merge_cross_page_tables(parsed.blocks)

        for block in blocks:
            if current_chunk and block.page != current_page:
                chunks.append(self._finalize_chunk(current_chunk, parsed, heading_stack))
                current_chunk = []
                current_len = 0
            if not current_chunk:
                current_page = block.page

            if block.block_type == "heading":
                # 标题是天然的切分点 —— 输出当前累积的 chunk
                if current_chunk:
                    chunks.append(self._finalize_chunk(current_chunk, parsed, heading_stack))
                    current_chunk = []
                    current_len = 0

                # 更新标题栈：弹出同级或更深的标题
                while heading_stack and heading_stack[-1][0] >= block.heading_level:
                    heading_stack.pop()
                heading_stack.append((block.heading_level, block.content))

                # 标题自身作为块内容
                current_chunk.append(block)
                current_len += len(block.content)
                continue

            if block.block_type in ("table", "list_item"):
                # 表格和列表不可拆分：如果当前 chunk 装不下，先输出旧 chunk，
                # 然后将表格/列表作为独立原子块
                block_len = len(block.content)
                if current_chunk and current_len + block_len > self._chunk_size * 1.5:
                    chunks.append(self._finalize_chunk(current_chunk, parsed, heading_stack))
                    current_chunk = []
                    current_len = 0
                current_chunk.append(block)
                current_len += block_len
                continue

            # 普通段落：按 chunk_size 切分
            needed = len(block.content)
            if current_chunk and current_len + needed > self._chunk_size:
                chunks.append(self._finalize_chunk(current_chunk, parsed, heading_stack))
                # 重叠：保留最后一个段落作为下一块的上下文
                if self._chunk_overlap > 0 and current_chunk:
                    overlap_block = current_chunk[-1]
                    current_chunk = [overlap_block]
                    current_len = len(overlap_block.content)
                else:
                    current_chunk = []
                    current_len = 0

            current_chunk.append(block)
            current_len += needed

        # 最后剩余的块
        if current_chunk:
            chunks.append(self._finalize_chunk(current_chunk, parsed, heading_stack))

        # 分配 chunk_index
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = total

        logger.info(
            "语义分块完成: %s → %d 个 chunk", parsed.doc_id, total
        )
        return chunks

    def _finalize_chunk(
        self,
        blocks: List[ParsedBlock],
        parsed: ParsedDocument,
        heading_stack: List[Tuple[int, str]],
    ) -> Document:
        """将一个块列表合并为最终 Document。"""
        text = "\n\n".join(b.content for b in blocks)

        # 当前章节标题：最近的 H1 + H2
        h1 = ""
        h2 = ""
        for level, title in heading_stack:
            if level == 1:
                h1 = title
            elif level == 2:
                h2 = title
        section_title = f"{h1} > {h2}" if h2 else h1 or ""

        # 页码：记录块的首末页，兼容未来跨页 chunk 的过滤
        page_starts = [b.page for b in blocks if b.page > 0]
        page_ends = [self._block_end_page(b) for b in blocks]
        start_page = min(page_starts) if page_starts else 1
        end_page = max(page_ends) if page_ends else start_page
        page_range = (
            str(start_page)
            if start_page == end_page
            else f"{start_page}-{end_page}"
        )

        # 权限标签：基于内容哈希虚构
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        perm = _PERMISSION_POOL[h % len(_PERMISSION_POOL)]

        # 文件修改时间
        file_path = parsed.extra.get("file_path", "")
        last_modified = ""
        if file_path:
            try:
                last_modified = datetime.fromtimestamp(
                    Path(file_path).stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except OSError:
                pass

        metadata = {
            "doc_id": parsed.doc_id,
            "source": parsed.source,
            "file_type": parsed.file_type,
            "page": start_page,
            "start_page": start_page,
            "end_page": end_page,
            "page_range": page_range,
            "total_pages": parsed.total_pages,
            "section_title": section_title,
            "heading_stack": "/".join(t for _, t in heading_stack),
            "last_modified": last_modified,
            "permission_tags": perm,
            "chunk_id": str(uuid.uuid4())[:8],
        }

        return Document(page_content=text, metadata=metadata)

    # ── 模式二：输入原始 Document（自动检测结构）───────────────────────

    def chunk_document(self, doc: Document) -> List[Document]:
        """对原始 langchain Document 自动检测 Markdown 结构后分块。

        适用于连接器直接返回的 Document，无需经过 smart_parser。
        """
        source = doc.metadata.get("source", "unknown")
        file_path = doc.metadata.get("file_path", "")
        text = doc.page_content

        # 自动检测文档中的结构元素
        blocks = self._detect_structure(text)

        # 构造 ParsedDocument 以便复用 chunk_parsed_doc
        parsed = ParsedDocument(
            doc_id=doc.metadata.get("doc_id", f"doc:{source}"),
            source=source,
            file_type=Path(source).suffix.lstrip(".") or "txt",
            blocks=blocks,
            extra={
                "file_path": file_path,
                "last_modified": doc.metadata.get("last_modified", ""),
            },
        )

        return self.chunk_parsed_doc(parsed)

    @staticmethod
    def _detect_structure(text: str) -> List[ParsedBlock]:
        """从纯文本中自动检测标题、表格、列表、段落结构。"""
        lines = text.split("\n")
        blocks: List[ParsedBlock] = []
        buffer: List[str] = []
        in_code_block = False

        def flush():
            nonlocal buffer
            txt = "\n".join(buffer).strip()
            buffer = []
            if txt:
                blocks.append(ParsedBlock(block_type="paragraph", content=txt))

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                flush()
                in_code_block = not in_code_block
                continue
            if in_code_block:
                buffer.append(line)
                continue

            # 标题检测
            m = re.match(r"^(#{1,6})\s+(.+)", stripped)
            if m:
                flush()
                blocks.append(
                    ParsedBlock(
                        block_type="heading",
                        content=m.group(2),
                        heading_level=len(m.group(1)),
                    )
                )
                continue

            # 表格行检测
            if stripped.startswith("|") and "|" in stripped[1:]:
                flush()
                blocks.append(ParsedBlock(block_type="table", content=stripped))
                continue

            # 列表检测
            if re.match(r"^(\s*[-*+]|\s*\d+[.)])\s+", stripped):
                flush()
                blocks.append(ParsedBlock(block_type="list_item", content=stripped))
                continue

            if stripped == "":
                flush()
                continue

            buffer.append(line)

        flush()
        return blocks


# ============================================================================
# 工具函数
# ============================================================================


def chunk_with_config(documents: List[Document]) -> List[Document]:
    """使用 config.yaml 中的分块参数进行语义分块。

    Args:
        documents: 待分块的 LangChain Document 列表。

    Returns:
        语义分块后的 Document 列表。
    """
    all_chunks: List[Document] = []
    for doc in documents:
        chunker = SemanticChunker(
            file_path=doc.metadata.get("file_path", ""),
        )
        all_chunks.extend(chunker.chunk_document(doc))
    return all_chunks
