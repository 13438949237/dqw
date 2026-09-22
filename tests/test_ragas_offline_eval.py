# -*- coding: utf-8 -*-
from pathlib import Path

from langchain_core.documents import Document

from scripts.run_ragas_offline_eval import (
    load_dataset,
    OfflineRetriever,
    QdrantRetriever,
)


ROOT = Path(__file__).resolve().parents[1]


def test_ragas_test_set_schema():
    dataset = load_dataset(ROOT / "data" / "eval" / "ragas_test_set.json")
    documents = dataset["meta"]["documents"]

    assert len(dataset["items"]) >= 25
    for item in dataset["items"]:
        assert item["question"].strip()
        assert item["ground_truth"].strip()
        assert item["id"].startswith("qa_")
        assert all(source in documents for source in item["expected_sources"])


def test_offline_retriever_ranks_expected_source_first():
    documents = [
        Document(
            page_content="2025年公司累计销售收入4.81亿元，毛利3344.06万元。",
            metadata={"source": "经营汇报.docx"},
        ),
        Document(
            page_content="这是一段与经营数据无关的仓库管理内容。",
            metadata={"source": "仓库说明.docx"},
        ),
    ]

    retriever = OfflineRetriever(documents)
    results = retriever.retrieve("公司累计销售收入是多少？", top_k=1)

    assert results[0]["source"] == "经营汇报.docx"


def test_qdrant_retriever_maps_pipeline_documents(monkeypatch):
    from src.retrievers import pipeline

    doc = Document(
        page_content="2026年计划新增五粮液八代订单3790件。",
        metadata={
            "source": "周报.docx",
            "rerank_score": 0.91,
        },
    )

    def fake_retrieve(**kwargs):
        return [doc]

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)

    results = QdrantRetriever().retrieve("五粮液八代新增订单", top_k=5)

    assert results[0]["source"] == "周报.docx"
    assert results[0]["score"] == 0.91
