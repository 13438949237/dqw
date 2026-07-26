from __future__ import annotations
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from src.cache.cache_service import CacheService
from src.connectors.file_connector import FileConnector
from src.embeddings.vector_store import QdrantVectorStore
from src.evaluation.metrics import MetricsTracker
from src.llms.models import ModelFactory
from src.parsers.pipeline import load_and_chunk
from src.rag_chain import RAGChain
from src.utils.config import get_config, load_config
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Enterprise RAG System", page_icon="", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def _init_config():
    load_config()
    return get_config()

@st.cache_resource
def _get_vector_store():
    return QdrantVectorStore()

def _init_session_state():
    for k, v in {"messages": [], "uploaded_files": {}}.items():
        if k not in st.session_state:
            st.session_state[k] = v

cfg = _init_config()
_init_session_state()
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("### Model Configuration")
    llm_providers = ["deepseek", "openai", "dashscope", "ollama"]
    cp = cfg.models.llm.provider
    idx = llm_providers.index(cp) if cp in llm_providers else 0
    provider = st.selectbox("LLM Provider", llm_providers, index=idx)
    model_name = st.text_input("LLM Model", value=cfg.models.llm.model_name)
    if st.button("Apply Model", width='stretch'):
        ModelFactory._llm_cache.clear()
        object.__setattr__(cfg.models.llm, "provider", provider)
        object.__setattr__(cfg.models.llm, "model_name", model_name)
        st.success("Switched to "+provider+"/"+model_name)
    st.divider()
    st.markdown("### Retrieval")
    top_k = st.slider("Top-K Results", 1, 50, cfg.retrieval.top_k)
    use_cache = st.checkbox("Enable Cache", value=cfg.cache.enabled)
    if st.button("Clear Cache", width='stretch'):
        n = CacheService().clear()
        st.info("Cleared "+str(n)+" cache entries")
    st.divider()
    st.markdown("### Status")
    try:
        store = _get_vector_store()
        st.metric("Indexed Chunks", store.count())
    except Exception:
        st.metric("Indexed Chunks", "N/A")
    try:
        s = CacheService().stats()
        st.caption("Cache: "+str(s.get("backend", "unknown")))
    except Exception:
        pass

tab_chat, tab_docs, tab_ops = st.tabs(["Chat", "Documents", "System Ops"])
with tab_chat:
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown("## RAG Q&A")
        uf = st.file_uploader("Upload documents to index", type=["pdf","docx","txt","md","html","csv"], accept_multiple_files=True)
        if uf:
            for fu in uf:
                if fu.name in st.session_state.uploaded_files:
                    continue
                with st.spinner("Processing "+fu.name+"..."):
                    td = tempfile.mkdtemp(prefix="rag_st_")
                    tp = os.path.join(td, fu.name)
                    try:
                        with open(tp, "wb") as fh:
                            fh.write(fu.getbuffer())
                        chunks = load_and_chunk(FileConnector(path=tp))
                        if chunks:
                            _get_vector_store().add_documents(chunks)
                            st.session_state.uploaded_files[fu.name] = len(chunks)
                            st.success(fu.name + " - "+str(len(chunks))+" chunks")
                        else:
                            st.warning(fu.name + " - no content")
                    except Exception as ex:
                        st.error(fu.name + " - "+str(ex))
                    finally:
                        shutil.rmtree(td, ignore_errors=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources and Confidence"):
                        for src in msg["sources"]:
                            score = src.get("score", 0)
                            txt = "**{s}** ({ft}, p{pg}) -- score: {sc:.4f}"
                            st.caption(txt.format(s=src.get("source","?"), ft=src.get("file_type",""), pg=str(src.get("page", "-")), sc=score))
                            if src.get("snippet"):
                                st.text(src["snippet"][:300])
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    t0 = time.perf_counter()
                    tr = MetricsTracker()
                    tr.start_request(prompt)
                    try:
                        result = RAGChain(top_k=top_k, use_cache=use_cache).answer(prompt)
                    except Exception as ex:
                        result = {"question":prompt,"answer":"Error: "+str(ex),"source_metadata":[],"cached":False,"model":"","latency_seconds":0}
                    elapsed = time.perf_counter() - t0
                    num_docs = len(result.get("retrieved_docs",[]))
                    tr.end_request(num_retrieved=num_docs)
                    at = result.get("answer", "")
                    ct = " *(cached)*" if result.get("cached") else ""
                    st.markdown(at + ct)
                    lat_str = "{:.2f}s".format(result.get("latency_seconds", elapsed))
                    st.caption("Model: "+str(result.get("model","N/A"))+" | Latency: "+lat_str)
                    sources = result.get("source_metadata", [])
                    if sources:
                        with st.expander("Sources ("+str(len(sources))+")"):
                            for s in sources:
                                sc = s.get("score", 0)
                                st.markdown("**{src}** -- score: {v:.4f}".format(src=s.get("source","?"), v=sc))
                                st.caption("Section: "+str(s.get("section_title","N/A"))+" | Page: "+str(s.get("page","-")))
                    st.session_state.messages.append({"role":"assistant","content":at,"sources":sources})
    with col_right:
        st.markdown("### Latest Query Stats")
        hist = MetricsTracker().get_history(limit=5)
        if hist:
            last = hist[-1]
            st.metric("Last Latency", "{:.2f}s".format(last.get("latency_seconds",0)))
            st.metric("Docs Retrieved", last.get("num_retrieved",0))
        else:
            st.caption("No queries yet")
with tab_docs:
    st.markdown("## Document Management")
    docs_data = []
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
        sc = {}
        stypes = {}
        offset = None
        while True:
            points, offset = client.scroll(collection_name=cfg.qdrant.collection_name, limit=200, offset=offset, with_payload=["source","file_type"], with_vectors=False)
            for pt in points:
                src = pt.payload.get("source","unknown") if pt.payload else "unknown"
                ft = pt.payload.get("file_type","") if pt.payload else ""
                sc[src] = sc.get(src, 0) + 1
                if src not in stypes and ft:
                    stypes[src] = ft
            if offset is None:
                break
        for src, count in sc.items():
            docs_data.append({"Source":src, "Type":stypes.get(src,""), "Chunks":count})
    except Exception as ex:
        st.warning("Could not connect to Qdrant: "+str(ex))
        for name, count in st.session_state.uploaded_files.items():
            docs_data.append({"Source":name, "Type":Path(name).suffix.lstrip("."), "Chunks":count})
    if docs_data:
        st.dataframe(pd.DataFrame(docs_data), width='stretch', hide_index=True)
        st.markdown("---")
        st.markdown("### Delete Document")
        dd = st.selectbox("Select a document source to delete", [""] + [d["Source"] for d in docs_data])
        if dd and st.button("Delete Selected Document", type="primary"):
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue
                client = QdrantClient(host=cfg.qdrant.host, port=cfg.qdrant.port)
                flt = Filter(must=[FieldCondition(key="source", match=MatchValue(value=dd))])
                client.delete(collection_name=cfg.qdrant.collection_name, points_selector=flt)
                CacheService().clear()
                st.session_state.uploaded_files.pop(dd, None)
                st.success("Deleted: "+dd)
                st.rerun()
            except Exception as ex:
                st.error("Failed: "+str(ex))
        st.markdown("---")
        st.markdown("### ⚠️ Clear All Documents")
        st.caption("This will delete all indexed documents and cannot be undone.")
        if st.button("Clear All Documents", type="secondary"):
            try:
                with st.spinner("Clearing all documents..."):
                    _get_vector_store().clear()
                    CacheService().clear()
                    st.session_state.uploaded_files.clear()
                st.success("All documents cleared successfully.")
                st.rerun()
            except Exception as ex:
                st.error("Failed to clear: " + str(ex))
    else:
        st.info("No documents indexed yet.")
with tab_ops:
    st.markdown("## System Monitoring")
    tracker = MetricsTracker()
    history = tracker.get_history(limit=200)
    summary = tracker.summary()
    cols = st.columns(5)
    with cols[0]:
        st.metric("Total Requests", summary.get("total_requests", 0))
    with cols[1]:
        al = summary.get("avg_latency")
        st.metric("Avg Latency", "{:.3f}s".format(al) if al else "N/A")
    with cols[2]:
        p50 = summary.get("p50_latency")
        st.metric("P50 Latency", "{:.3f}s".format(p50) if p50 else "N/A")
    with cols[3]:
        p99 = summary.get("p99_latency")
        st.metric("P99 Latency", "{:.3f}s".format(p99) if p99 else "N/A")
    with cols[4]:
        ar = summary.get("avg_recall")
        st.metric("Avg Recall", "{:.3f}".format(ar) if ar else "N/A")
    st.markdown("---")
    st.markdown("### Latency Over Time")
    if history:
        ld = pd.DataFrame([{"idx":i, "latency":h.get("latency_seconds",0)} for i,h in enumerate(history)])
        st.line_chart(ld.set_index("idx")["latency"], height=250)
    else:
        st.info("No query history yet")
    st.markdown("### Token Usage")
    if history:
        td = pd.DataFrame([{"idx":i, "input":h.get("tokens_input",0), "output":h.get("tokens_output",0)} for i,h in enumerate(history)])
        if td["input"].sum() > 0 or td["output"].sum() > 0:
            st.line_chart(td.set_index("idx"), height=250)
        else:
            st.info("Token data not available")
    else:
        st.info("No query history yet")
    st.markdown("### Recall and Precision")
    if history:
        rd = pd.DataFrame([{"idx":i, "recall":h.get("recall") or 0, "precision":h.get("precision") or 0} for i,h in enumerate(history)])
        if rd["recall"].sum() > 0 or rd["precision"].sum() > 0:
            st.line_chart(rd.set_index("idx"), height=250)
        else:
            st.info("Recall/precision data available only with ground-truth labels")
    else:
        st.info("No query history yet")
    st.markdown("---")
    st.markdown("### Recent Queries")
    if history:
        rq = pd.DataFrame([{"Time":h.get("timestamp","")[:19], "Query":h.get("query",""), "Latency":"{:.3f}s".format(h.get("latency_seconds",0)), "Docs":h.get("num_retrieved",0)} for h in reversed(history[-30:])])
        st.dataframe(rq, width='stretch', hide_index=True)
    else:
        st.info("No query history yet")
    st.markdown("---")
    st.markdown("### Feedback Summary")
    st.info("Feedback data is collected via the API endpoint (POST /feedback).")
