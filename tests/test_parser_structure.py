# -*- coding: utf-8 -*-
from langchain_core.documents import Document

import src.parsers.smart_parser as smart_parser
from src.connectors.base import BaseConnector
from src.parsers.pipeline import load_and_chunk
from src.parsers.semantic_chunker import SemanticChunker, chunk_with_config
from src.parsers.smart_parser import (
    ParsedBlock,
    ParsedDocument,
    _parse_mineru_markdown,
)
from src.utils.config import get_config


def test_mineru_markdown_preserves_structure_and_page():
    markdown = """# 项目背景

这是第一段正文。

## 经营数据

| 指标 | 第一季度 | 第二季度 |
| --- | --- | --- |
| 营收 | 120 | 186 |

- 重点事项一
- 重点事项二

这是第二段正文。
"""

    blocks = _parse_mineru_markdown(markdown, page=2)

    assert [b.block_type for b in blocks] == [
        "heading",
        "paragraph",
        "heading",
        "table",
        "list_item",
        "paragraph",
    ]
    assert blocks[0].heading_level == 1
    assert blocks[2].heading_level == 2
    assert "营收 | 120 | 186" in blocks[3].content
    assert "| --- |" not in blocks[3].content
    assert all(block.page == 2 for block in blocks)


def test_mineru_markdown_extracts_html_table_as_table_block():
    markdown = """# 产品价格体系

为维护渠道利润，制定以下价格：

<table><tr><th>项目</th><th>价格</th><th>说明</th></tr><tr><td>供货价</td><td>360元/瓶</td><td>基准供货价格</td></tr><tr><td>建议零售价</td><td>456元/瓶</td><td>终端门店标价</td></tr></table>

严禁低于红线价销售。
"""

    blocks = _parse_mineru_markdown(markdown, page=4)

    assert [block.block_type for block in blocks] == [
        "heading",
        "paragraph",
        "table",
        "paragraph",
    ]
    table_block = blocks[2]
    assert table_block.content == (
        "项目 | 价格 | 说明\n"
        "供货价 | 360元/瓶 | 基准供货价格\n"
        "建议零售价 | 456元/瓶 | 终端门店标价"
    )
    assert "<table>" not in table_block.content
    assert table_block.page == 4


def test_semantic_chunker_flushes_on_page_boundary_and_writes_page_range():
    parsed = ParsedDocument(
        doc_id="pdf:guide",
        source="guide.pdf",
        file_type="pdf",
        total_pages=2,
        blocks=[
            ParsedBlock("heading", "第一章", heading_level=1, page=1),
            ParsedBlock("paragraph", "第一页正文内容。", page=1),
            ParsedBlock("heading", "第二章", heading_level=1, page=2),
            ParsedBlock("paragraph", "第二页正文内容。", page=2),
        ],
    )

    chunks = SemanticChunker(chunk_size=800, chunk_overlap=80).chunk_parsed_doc(
        parsed
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["start_page"] == 1
    assert chunks[0].metadata["end_page"] == 1
    assert chunks[0].metadata["page_range"] == "1"
    assert chunks[0].metadata["section_title"] == "第一章"
    assert chunks[1].metadata["page"] == 2
    assert chunks[1].metadata["page_range"] == "2"
    assert chunks[1].metadata["section_title"] == "第二章"


def test_semantic_chunker_merges_cross_page_table_fragments():
    parsed = ParsedDocument(
        doc_id="pdf:report",
        source="report.pdf",
        file_type="pdf",
        total_pages=3,
        blocks=[
            ParsedBlock("heading", "库存分析", heading_level=1, page=1),
            ParsedBlock(
                "table",
                "类别 | 数量（个） | 总计（箱）\n普通门店 | 7398 | 14000",
                page=1,
            ),
            ParsedBlock("table", "合计 | 9398 | 24000", page=2),
            ParsedBlock("paragraph", "下一页继续说明库存去化安排。", page=2),
        ],
    )

    chunks = SemanticChunker(chunk_size=800, chunk_overlap=80).chunk_parsed_doc(
        parsed
    )

    assert len(chunks) == 2
    table_chunk = chunks[0]
    assert table_chunk.metadata["start_page"] == 1
    assert table_chunk.metadata["end_page"] == 2
    assert table_chunk.metadata["page_range"] == "1-2"
    assert "普通门店 | 7398 | 14000" in table_chunk.page_content
    assert "合计 | 9398 | 24000" in table_chunk.page_content
    assert table_chunk.page_content.count("类别 | 数量（个） | 总计（箱）") == 1
    assert chunks[1].metadata["page"] == 2


def test_semantic_chunker_does_not_merge_different_tables_on_same_page():
    parsed = ParsedDocument(
        doc_id="pdf:report",
        source="report.pdf",
        file_type="pdf",
        total_pages=1,
        blocks=[
            ParsedBlock("table", "项目 | 价格\n供货价 | 360", page=1),
            ParsedBlock("table", "类别 | 数量\n普通门店 | 7398", page=1),
        ],
    )

    merged = SemanticChunker._merge_cross_page_tables(parsed.blocks)

    assert len(merged) == 2
    assert "项目 | 价格" in merged[0].content
    assert "类别 | 数量" in merged[1].content


def test_semantic_chunker_reads_per_type_chunk_params_from_config():
    cfg = get_config().document

    pdf_chunker = SemanticChunker(file_path="report.pdf")
    xlsx_chunker = SemanticChunker(file_path="report.xlsx")

    assert (pdf_chunker._chunk_size, pdf_chunker._chunk_overlap) == (
        cfg.get_chunk_params(".pdf")
    )
    assert (xlsx_chunker._chunk_size, xlsx_chunker._chunk_overlap) == (
        cfg.get_chunk_params(".xlsx")
    )


def test_semantic_chunker_explicit_params_override_config():
    chunker = SemanticChunker(
        chunk_size=321,
        chunk_overlap=17,
        file_path="report.pdf",
    )

    assert chunker._chunk_size == 321
    assert chunker._chunk_overlap == 17


def test_chunk_with_config_falls_back_to_config_defaults():
    doc = Document(
        page_content="# 标题\n\n这是一个超过二十个字符的普通段落内容。",
        metadata={"source": "guide.txt"},
    )

    chunks = chunk_with_config([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata["section_title"] == "标题"


def test_mineru_pdf_uses_split_pages_and_keeps_page_metadata(monkeypatch, tmp_path):
    captured = {}

    class FakeMinerULoader:
        def __init__(self, source, mode, split_pages):
            captured["source"] = source
            captured["mode"] = mode
            captured["split_pages"] = split_pages

        def load(self):
            return [
                Document(
                    page_content=(
                        "# 经营数据\n\n"
                        "| 指标 | 第一季度 |\n"
                        "| --- | --- |\n"
                        "| 营收 | 120 |"
                    ),
                    metadata={"page": 2},
                )
            ]

    monkeypatch.setattr(smart_parser, "_check_mineru", lambda: True)
    monkeypatch.setattr(
        "langchain_mineru.document_loaders.MinerULoader",
        FakeMinerULoader,
    )
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    parsed = smart_parser.parse_with_mineru(str(pdf_path))

    assert captured["mode"] == "flash"
    assert captured["split_pages"] is True
    assert parsed.total_pages == 2
    assert [block.block_type for block in parsed.blocks] == [
        "heading",
        "table",
    ]
    assert all(block.page == 2 for block in parsed.blocks)


def test_pipeline_processes_one_file_once_and_preserves_parser_page(tmp_path):
    markdown_path = tmp_path / "guide.md"
    markdown_path.write_text(
        "# 第一章\n\n第一页正文内容。这里补充足够的文字，使最终分块长度超过二十个字符。",
        encoding="utf-8",
    )

    class FakeConnector(BaseConnector):
        def __init__(self):
            super().__init__(source_label="guide.md")

        def load_documents(self):
            return [
                Document(
                    page_content=(
                        "# 第一章\n\n"
                        "第一页正文内容。这里补充足够的文字，使最终分块长度超过二十个字符。"
                    ),
                    metadata={
                        "file_path": str(markdown_path),
                        "source": "guide.md",
                        "page": 7,
                    },
                )
                for _ in range(3)
            ]

    chunks = load_and_chunk(FakeConnector())

    assert len(chunks) == 1
    assert chunks[0].metadata["page"] == 1
    assert chunks[0].metadata["section_title"] == "第一章"
