# -*- coding: utf-8 -*-
"""离线入库与文件管理页面。"""
from __future__ import annotations
import sys, time, uuid
from pathlib import Path
from typing import Dict
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import streamlit as st
from src.connectors.file_connector import FileConnector
from src.embeddings.vector_store import QdrantVectorStore
from src.parsers.smart_parser import is_supported_format

from src.parsers.pipeline import load_and_chunk
from qdrant_client.models import FieldCondition, Filter, MatchValue

st.set_page_config(page_title="文档管理", page_icon="📄", layout="wide")
st.title("📄 文档管理")
st.caption("上传文档、管理已入库文件、查看索引状态")


@st.cache_resource
def _store() -> QdrantVectorStore:
    return QdrantVectorStore()


# ── 上传区 ──
st.header("📤 上传文档")
files = st.file_uploader("选择文件（支持 md / txt / pdf / png / jpg / bmp / docx / pptx / xlsx）", accept_multiple_files=True, type=["md","txt","pdf","png","jpg","jpeg","bmp","tiff","docx","pptx","xlsx","xls","csv"])
if files and st.button("上传并入库", use_container_width=True):
    d = Path("data/uploads"); d.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.name and not is_supported_format(f.name):
            st.error(f"❌ {f.name}: 不支持该文件类型，请选择规定类型的文件（.txt/.md/.pdf/.png/.jpg/.docx/.pptx/.xlsx）")
            continue
        dest = d / f"{uuid.uuid4().hex[:8]}_{f.name}"
        try:
            dest.write_bytes(f.getvalue())
            chunks = load_and_chunk(FileConnector(str(dest)))
            if not chunks:
                st.warning(f"⚠️ {f.name}: 内容为空")
                continue
            _store().add_documents(chunks, generate_sentences=True)
            st.success(f"✅ {f.name}: {len(chunks)} 个分块已入库")
        except Exception as e:
            st.error(f"❌ {f.name}: {e}")


# ── 已入库文档 ──
st.divider()
st.header("📚 已入库文档")
if st.button("🔄 刷新列表"):
    st.rerun()
try:
    s = _store()
    recs, _ = s._client.scroll(collection_name=s._chunk_collection, limit=1000, with_payload=True)
except Exception as e:
    st.error(f"查询失败: {e}"); recs = []

sm: Dict[str, dict] = {}
for rec in recs:
    if not rec.payload:
        continue
    src = rec.payload.get("source", "?")
    if src not in sm:
        sm[src] = {"source": src, "file_type": rec.payload.get("file_type",""), "chunks": 0}
    sm[src]["chunks"] += 1

docs = list(sm.values())
st.metric("文档数量", len(docs))
if docs:
    for d in docs:
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.text(d["source"]); c2.text(d.get("file_type","-")); c3.text(f"{d['chunks']} chunks")
        if c4.button("🗑️ 删除", key=f"del_{d['source']}"):
            try:
                s._client.delete(collection_name=s._chunk_collection, points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=d["source"]))]))
                st.success(f"已删除: {d['source']}"); time.sleep(0.5); st.rerun()
            except Exception as e:
                st.error(str(e))
else:
    st.info("暂无已入库文档，请上传文件后刷新")

# ── 一键清空 ──
st.divider()


@st.dialog("⚠️ 危险操作")
def _clear_all_dialog():
    st.warning("请慎重清空文本库！是否继续执行？")
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("取消", use_container_width=True):
            st.rerun()
    with col_confirm:
        if st.button("确定清空", use_container_width=True, type="primary"):
            try:
                _store().clear()
                st.success("✅ 文本库已清空")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"清空失败: {e}")


if st.button("🗑️ 一键清空文本库", use_container_width=True, type="primary"):
    _clear_all_dialog()
