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
# 允许上传的文件格式
# ============================================================================

# 纯文本格式（使用传统解析器）
_TEXT_FORMATS  = frozenset({".txt", ".md"})

# 复杂格式（优先使用 MinerU flash 解析，不可用时回退传统解析器）
_COMPLEX_FORMATS = frozenset({
    ".pdf",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff",
    ".docx", ".pptx",
    ".xlsx", ".xls",
})

# 全部允许的扩展名
ALLOWED_EXTENSIONS: frozenset = _TEXT_FORMATS | _COMPLEX_FORMATS


def filter_chunks(chunks: list, min_chars: int = 20, max_whitespace_ratio: float = 0.6) -> list:
    """分块质量校验：过滤过短块和高噪声块。

    Args:
        chunks:       LangChain Document 列表。
        min_chars:    最小字符数（低于此值舍弃）。
        max_whitespace_ratio: 最大空白比例（超出视为噪声块舍弃）。

    Returns:
        过滤后的 Document 列表。
    """
    kept = []
    discarded = 0
    for c in chunks:
        text = c.page_content
        if len(text) < min_chars:
            discarded += 1
            continue
        whitespace = sum(1 for ch in text if ch in " \t\n\r")
        if len(text) > 0 and whitespace / len(text) > max_whitespace_ratio:
            discarded += 1
            continue
        kept.append(c)
    if discarded:
        logger.info("分块质量校验: 保留 %d, 舍弃 %d (过短/高噪声)", len(kept), discarded)
    return kept


def is_supported_format(filename: str) -> bool:
    """判断文件扩展名是否在允许列表中。

    Args:
        filename: 文件名或完整路径。

    Returns:
        True 表示支持该格式。
    """
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


# ============================================================================
# MinerU Flash 解析器 —— 统一处理 PDF / 图片 / DOCX / PPTX / XLSX
# ============================================================================

# 由 MinerU flash 引擎处理的格式集合
_MINERU_FORMATS = frozenset({
    ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".docx", ".pptx", ".xlsx", ".xls",
})

# MinerU 可用性标记：None=未检测 / True=可用 / False=不可用（遇 SSL 错误等重要异常后置为 False）
_mineru_available: Optional[bool] = None


def _check_mineru() -> bool:
    """检测 MinerU 是否可用，首次调用时会尝试导入并快速探测。

    如果 langchain-mineru 未安装，或导入/加载时发生 SSL / 连接错误，
    则将 _mineru_available 置为 False，后续调用直接走回退解析器，
    避免每次上传都触发长时间的 SSL 超时。

    Returns:
        True 表示 MinerU 可用。
    """
    global _mineru_available

    if _mineru_available is not None:
        return _mineru_available

    try:
        # 实际触发导入 —— 这里可能抛出 ImportError 或 SSL 错误
        from langchain_mineru.document_loaders import MinerULoader  # noqa: F401
        _mineru_available = True
        logger.info("MinerU flash 解析器可用")
    except ImportError as exc:
        logger.warning("langchain-mineru 未安装，将使用传统解析器: %s", exc)
        _mineru_available = False
    except Exception as exc:
        # 包括 httpx.ConnectError / SSL 错误等
        logger.warning(
            "MinerU 初始化失败（网络/SSL 错误），将使用传统解析器: %s", exc
        )
        _mineru_available = False

    return _mineru_available


def parse_with_mineru(file_path: str) -> ParsedDocument:
    """使用 langchain-mineru 的 flash 模式解析复杂文档格式。

    MinerU flash 模式利用轻量级流水线实现快速解析，适用于 PDF、
    图片（PNG/JPG/BMP/TIFF）、DOCX、PPTX、XLSX 等多格式。
    若 MinerU 不可用，回退到对应传统解析器。

    Args:
        file_path: 本地文件路径。

    Returns:
        ParsedDocument 结构化中间表示。
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    # ── MinerU 可用性检查 ──
    if not _check_mineru():
        logger.info("MinerU 不可用，直接回退传统解析: %s", path.name)
        return _fallback_parse(file_path, suffix)

    # ── MinerU 解析 ──
    try:
        from langchain_mineru.document_loaders import MinerULoader  # noqa: F811

        logger.info("使用 MinerU flash 模式解析: %s", path.name)
        loader = MinerULoader(source=str(path), mode="flash")
        docs = loader.load()

        blocks: List[ParsedBlock] = []
        for doc in docs:
            text = doc.page_content
            if not text.strip():
                continue
            page = doc.metadata.get("page_number", doc.metadata.get("page", 1))
            page_num = int(page) if page else 1

            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text.strip()]

            for para in paragraphs:
                blocks.append(
                    ParsedBlock(
                        block_type="paragraph",
                        content=para,
                        page=page_num,
                    )
                )

        if not blocks:
            logger.warning("MinerU 未提取到文本内容: %s", path.name)
            return ParsedDocument(
                doc_id=f"mineru:{path.stem}",
                source=path.name,
                file_type=suffix.lstrip("."),
            )

        logger.info(
            "MinerU 解析完成: %s, %d 块", path.name, len(blocks)
        )
        return ParsedDocument(
            doc_id=f"mineru:{path.stem}",
            source=path.name,
            file_type=suffix.lstrip("."),
            blocks=blocks,
        )

    except ImportError:
        logger.warning("MinerU 导入失败，回退传统解析: %s", path.name)
        return _fallback_parse(file_path, suffix)

    except Exception as exc:
        logger.warning("MinerU 解析异常 (%s)，回退传统解析: %s", exc, path.name)
        return _fallback_parse(file_path, suffix)


def _fallback_parse(file_path: str, suffix: str) -> ParsedDocument:
    """MinerU 失败时的回退解析（路由到原有传统解析器）。"""
    path = Path(file_path)

    if suffix in (".pdf",):
        return parse_pdf(file_path)
    if suffix in (".docx", ".doc"):
        return parse_docx(file_path)
    if suffix in (".xlsx", ".xls"):
        return parse_xlsx(file_path)
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pptx"):
        return ParsedDocument(
            doc_id=f"fallback:{path.stem}",
            source=path.name,
            file_type=suffix.lstrip("."),
        )
    # 兜底：当纯文本处理
    return parse_text(file_path)


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
    """使用 pdfplumber 解析 PDF 文档，支持文本和表格提取。"""
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "PDF 解析需要安装 pdfplumber，请执行: pip install pdfplumber"
        ) from exc

    path = Path(file_path)
    blocks: List[ParsedBlock] = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page_idx, page in enumerate(pdf.pages, start=1):
            # 1. 提取表格
            tables = page.extract_tables() or []
            table_regions = []
            for table in tables:
                if not table:
                    continue
                rows = []
                for row in table:
                    cells = [str(c).strip() if c else "" for c in row]
                    rows.append(" | ".join(cells))
                if rows:
                    blocks.append(
                        ParsedBlock(
                            block_type="table",
                            content="\n".join(rows),
                            page=page_idx,
                        )
                    )
                # 记录表格区域，后续排除
                if table:
                    for row in table:
                        for cell in row:
                            if cell:
                                table_regions.append(cell.strip())

            # 2. 提取文本（排除已提取的表格内容）
            text = page.extract_text() or ""
            if not text.strip():
                continue

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for para in paragraphs:
                # 跳过已被表格覆盖的内容
                if any(cell in para for cell in table_regions if len(cell) > 5):
                    continue
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
        total_pages=total_pages,
    )
    logger.info("PDF 解析完成: %s, %d 页, %d 块", path.name, doc.total_pages, len(blocks))
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
    # ── 纯文本（传统解析器） ──
    ".txt":  parse_text,
    ".md":   parse_markdown,
    ".html": parse_text,
    ".htm":  parse_text,
    ".csv":  parse_text,
    # ── 复杂格式（MinerU flash 解析器） ──
    ".pdf":  parse_with_mineru,
    ".png":  parse_with_mineru,
    ".jpg":  parse_with_mineru,
    ".jpeg": parse_with_mineru,
    ".bmp":  parse_with_mineru,
    ".tiff": parse_with_mineru,
    ".docx": parse_with_mineru,
    ".pptx": parse_with_mineru,
    ".xlsx": parse_with_mineru,
    ".xls":  parse_with_mineru,
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

    # ── 格式白名单校验 ──
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持该文件类型 (.{suffix})，请选择规定类型的文件。"
            f"允许的格式: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    parser = _PARSER_REGISTRY.get(suffix)
    return parser(file_path)
