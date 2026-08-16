# -*- coding: utf-8 -*-
"""离线入库与文件管理页面。"""
from __future__ import annotations
import sys, time, uuid
from pathlib import Path
from typing import Dict, List, Set
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import streamlit as st
from src.embeddings.vector_store import QdrantVectorStore
from src.parsers.smart_parser import is_supported_format
from src.parsers.pipeline import parallel_load_and_chunk
from qdrant_client.models import FieldCondition, Filter, MatchValue

st.set_page_config(page_title="文档管理", page_icon="📄", layout="wide")
st.title("📄 文档管理")
st.caption("上传文档、管理已入库文件、查看索引状态")


@st.cache_resource
def _store() -> QdrantVectorStore:
    return QdrantVectorStore()


def _get_existing_hashes(store: QdrantVectorStore) -> Set[str]:
    """从 Qdrant 中提取已入库文件的哈希集合（用于文件级去重）。"""
    hashes: Set[str] = set()
    try:
        recs, _ = store._client.scroll(
            collection_name=store._chunk_collection,
            limit=200, with_payload=True, with_vectors=False,
        )
        for rec in recs:
            h = (rec.payload or {}).get("file_hash", "")
            if h:
                hashes.add(h)
    except Exception:
        pass
    return hashes

def _delete_by_sources(store: QdrantVectorStore, source_names: set) -> None:
    """按文件名删除 Qdrant 中的旧数据（实现覆盖更新）。"""
    for source in source_names:
        try:
            store._client.delete(
                collection_name=store._chunk_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
            )
            store._client.delete(
                collection_name=store._sent_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                ),
            )
        except Exception:
            pass

# ── 上传区 ──
st.header("📤 上传文档")
files = st.file_uploader(
    "选择文件（支持 md / txt / pdf / png / jpg / bmp / docx / pptx / xlsx）",
    accept_multiple_files=True,
    type=["md","txt","pdf","png","jpg","jpeg","bmp","tiff","docx","pptx","xlsx","xls","csv"],
)
if files and st.button("上传并入库", use_container_width=True):
    d = Path("data/uploads"); d.mkdir(parents=True, exist_ok=True)

    file_paths: List[str] = []
    source_names: List[str] = []
    results: List[dict] = []

    for f in files:
        if f.name and not is_supported_format(f.name):
            results.append({"name": f.name, "status": "skip", "msg": "不支持的文件类型"})
            continue
        dest = d / f"{uuid.uuid4().hex[:8]}_{f.name}"
        try:
            dest.write_bytes(f.getvalue())
            file_paths.append(str(dest))
            source_names.append(f.name)
        except Exception as e:
            results.append({"name": f.name, "status": "fail", "msg": f"保存失败: {e}"})

    if file_paths:
        store = _store()
        skip_hashes = _get_existing_hashes(store)

        existing_sources = {sn for sn in source_names}
        _delete_by_sources(store, existing_sources)

        progress_bar = st.progress(0, text="准备解析...")
        status_text = st.empty()

        def on_progress(current, total, filename):
            progress_bar.progress(current / total, text=f"解析进度: {current}/{total}")
            status_text.text(f"正在处理: {filename}")

        try:
            chunks, file_reports = parallel_load_and_chunk(
                file_paths=file_paths,
                source_names=source_names,
                max_workers=4,
                progress_callback=on_progress,
                skip_hashes=skip_hashes,
            )

            progress_bar.progress(0.95, text="正在向量化入库...")
            status_text.empty()

            if chunks:
                store.add_documents(chunks, generate_sentences=True, dedup=True)

            progress_bar.progress(1.0, text="✅ 全部完成")

            ok = sum(1 for r in file_reports if r["status"] == "success")
            sk = sum(1 for r in file_reports if r["status"] == "skipped")
            fl = sum(1 for r in file_reports if r["status"] == "failed")
            st.success(
                f"✅ 处理完成：{ok} 成功, {sk} 跳过（重复）, {fl} 失败，共 {len(chunks)} 个分块入库"
            )

            if file_reports:
                with st.expander("📋 查看详细处理报告"):
                    for r in file_reports:
                        icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(r["status"], "❓")
                        detail = r.get("reason", f'{r.get("chunks", 0)} chunks')
                        st.text(f'{icon} {r["source"]} — {detail}')

        except Exception as e:
            st.error(f"入库失败: {e}")

    for r in results:
        if r["status"] == "skip":
            st.warning(f"⚠️ {r['name']}: {r['msg']}")
        elif r["status"] == "fail":
            st.error(f"❌ {r['name']}: {r['msg']}")


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
