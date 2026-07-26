# -*- coding: utf-8 -*-
"""
多格式智能文档解析器。

根据文件扩展名自动分派到对应的解析引擎（pypdf / python-docx / markdown / openpyxl），
返回统一结构的 ParsedDocument，包含页面级/段落级结构信息。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# 解析结果数据结构
# ============================================================================


@dataclass
class ParsedBlock:
    """单个解析块（段落、标题、表格行等）。"""

    block_type: str          # "heading" | "paragraph" | "table" | "list_item"
    content: str
    heading_level: int = 0   # 仅 heading 类型有效，1=H1, 2=H2, ...
    page: int = 1            # 页码（PDF 解析时填充）
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """经过解析器处理后的文档中间表示。"""

    doc_id: str
    source: str
    file_type: str
    blocks: List[ParsedBlock] = field(default_factory=list)
    total_pages: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 解析引擎
# ============================================================================


def parse_pdf(file_path: str) -> ParsedDocument:
    """使用 pypdf 解析 PDF 文档，按页提取文本。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF 解析需要安装 pypdf，请执行: pip install pypdf"
        ) from exc

    path = Path(file_path)
    reader = PdfReader(file_path)
    blocks: List[ParsedBlock] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        # 按双换行切为段落
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            blocks.append(
                ParsedBlock(
                    block_type="paragraph",
                    content=para,
                    page=page_idx,
                )
            )

    doc = ParsedDocument(
        doc_id=f"pdf:{path.stem}",
        source=path.name,
        file_type="pdf",
        blocks=blocks,
        total_pages=len(reader.pages),
    )
    logger.info("PDF 解析完成: %s, %d 页, %d 段落", path.name, doc.total_pages, len(blocks))
    return doc


def parse_docx(file_path: str) -> ParsedDocument:
    """使用 python-docx 解析 Word 文档。"""
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ImportError(
            "DOCX 解析需要安装 python-docx，请执行: pip install python-docx"
        ) from exc

    path = Path(file_path)
    docx = DocxDocument(file_path)
    blocks: List[ParsedBlock] = []

    for para in docx.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # 根据样式名判断是否为标题
        style_name = para.style.name if para.style else ""
        if style_name.startswith("Heading") or style_name.startswith("标题"):
            try:
                level = int(re.search(r"\d+", style_name).group())  # type: ignore[union-attr]
            except (AttributeError, ValueError):
                level = 1
            blocks.append(
                ParsedBlock(block_type="heading", content=text, heading_level=level)
            )
        else:
            blocks.append(ParsedBlock(block_type="paragraph", content=text))

    # 提取表格
    for table in docx.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            blocks.append(
                ParsedBlock(
                    block_type="table",
                    content="\n".join(rows),
                )
            )

    doc = ParsedDocument(
        doc_id=f"docx:{path.stem}",
        source=path.name,
        file_type="docx",
        blocks=blocks,
    )
    logger.info("DOCX 解析完成: %s, %d 个块", path.name, len(blocks))
    return doc


def parse_markdown(file_path: str) -> ParsedDocument:
    """使用正则解析 Markdown 文档，保留标题层级。"""
    path = Path(file_path)
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.split("\n")
    blocks: List[ParsedBlock] = []
    buffer: List[str] = []
    in_code_block = False

    def flush_buffer():
        nonlocal buffer
        text = "\n".join(buffer).strip()
        buffer = []
        if text:
            blocks.append(ParsedBlock(block_type="paragraph", content=text))

    for line in lines:
        # 代码块边界
        if line.strip().startswith("```"):
            flush_buffer()
            in_code_block = not in_code_block
            continue

        if in_code_block:
            buffer.append(line)
            continue

        # 标题
        heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading_match:
            flush_buffer()
            level = len(heading_match.group(1))
            blocks.append(
                ParsedBlock(
                    block_type="heading",
                    content=heading_match.group(2).strip(),
                    heading_level=level,
                )
            )
            continue

        # 表格行（以 | 开头或以 | 为分隔）
        if line.strip().startswith("|"):
            flush_buffer()
            blocks.append(
                ParsedBlock(block_type="table", content=line.strip())
            )
            continue

        # 列表项
        if re.match(r"^(\s*[-*+]|\s*\d+[.)])\s+", line):
            flush_buffer()
            blocks.append(
                ParsedBlock(block_type="list_item", content=line.strip())
            )
            continue

        # 空行 = 段落边界
        if line.strip() == "":
            flush_buffer()
            continue

        buffer.append(line)

    flush_buffer()

    doc = ParsedDocument(
        doc_id=f"md:{path.stem}",
        source=path.name,
        file_type="md",
        blocks=blocks,
    )
    logger.info("Markdown 解析完成: %s, %d 个块", path.name, len(blocks))
    return doc


def parse_xlsx(file_path: str) -> ParsedDocument:
    """使用 openpyxl 解析 Excel 表格，每行作为一个块。

    如 unstructured 可用，优先使用 unstructured 以获取更好的表格语义理解。
    """
    path = Path(file_path)
    blocks: List[ParsedBlock] = []

    # 优先尝试 unstructured
    try:
        from unstructured.partition.xlsx import partition_xlsx

        elements = partition_xlsx(filename=file_path)
        for el in elements:
            el_type = str(type(el).__name__).lower()
            if "table" in el_type:
                block_type = "table"
            elif "title" in el_type or "header" in el_type:
                block_type = "heading"
            else:
                block_type = "paragraph"
            blocks.append(
                ParsedBlock(block_type=block_type, content=str(el))
            )
        logger.info("XLSX 解析完成 (unstructured): %s", path.name)
    except ImportError:
        logger.debug("unstructured 不可用，回退到 openpyxl")
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise ImportError(
                "XLSX 解析需要安装 openpyxl 或 unstructured"
            ) from exc

        wb = load_workbook(file_path, read_only=True, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue

            # 第一行作为表头
            header = [str(c or "") for c in rows[0]]
            for row_idx, row in enumerate(rows[1:], start=2):
                line = " | ".join(
                    f"{header[i]}: {val}"
                    for i, val in enumerate(row)
                    if val is not None
                )
                if line.strip():
                    blocks.append(
                        ParsedBlock(
                            block_type="table",
                            content=line,
                            extra={"sheet": sheet_name, "row": row_idx},
                        )
                    )
        wb.close()
        logger.info("XLSX 解析完成 (openpyxl): %s", path.name)
    except Exception:
        logger.exception("XLSX 解析失败: %s", path.name)
        raise

    doc = ParsedDocument(
        doc_id=f"xlsx:{path.stem}",
        source=path.name,
        file_type="xlsx",
        blocks=blocks,
    )
    return doc


def parse_text(file_path: str) -> ParsedDocument:
    """解析纯文本文件。"""
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    blocks = [
        ParsedBlock(block_type="paragraph", content=p) for p in paragraphs
    ]
    return ParsedDocument(
        doc_id=f"txt:{path.stem}",
        source=path.name,
        file_type="txt",
        blocks=blocks,
    )


# ============================================================================
# 分派引擎
# ============================================================================

# 扩展名 → 解析器映射
_PARSER_REGISTRY: Dict[str, callable] = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
    ".doc":  parse_docx,
    ".md":   parse_markdown,
    ".xlsx": parse_xlsx,
    ".xls":  parse_xlsx,
    ".txt":  parse_text,
    ".csv":  parse_text,
    ".html": parse_text,
    ".htm":  parse_text,
}


def parse_file(file_path: str) -> ParsedDocument:
    """根据文件扩展名自动分派解析引擎。

    Args:
        file_path: 本地文件路径。

    Returns:
        ParsedDocument 结构化中间表示。

    Raises:
        ValueError: 不支持的扩展名。
    """
    suffix = Path(file_path).suffix.lower()
    parser = _PARSER_REGISTRY.get(suffix)
    if parser is None:
        raise ValueError(
            f"不支持的文件类型: {suffix}。"
            f"当前支持: {', '.join(_PARSER_REGISTRY.keys())}"
        )
    return parser(file_path)
