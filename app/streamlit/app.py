# -*- coding: utf-8 -*-
"""Enterprise RAG — 在线检索与系统指标。"""
from __future__ import annotations
import sys, time
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.evaluation.metrics import MetricsTracker
from src.llms.models import ModelFactory
from src.rag_chain import RAGChain
from src.sessions.session_manager import SessionManager
from src.cache import CacheService

st.set_page_config(page_title="Enterprise RAG", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

# ── session_state 初始化 ──
_DEFAULTS = {
    "current_session_id": "",
    "session_messages": [],
    "generating": False,
    "pending_question": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


@st.cache_resource
def _chain() -> RAGChain:
    return RAGChain()


@st.cache_resource
def _tracker() -> MetricsTracker:
    return MetricsTracker()


@st.cache_resource
def _session_mgr() -> SessionManager:
    return SessionManager()


@st.cache_resource
def _cache_service() -> CacheService:
    return CacheService()


def _switch_session(sid: str):
    st.session_state.current_session_id = sid
    st.session_state.session_messages = _session_mgr().get_messages(sid)
    st.session_state.generating = False
    st.session_state.pending_question = None


# ============================================================================
# 侧边栏：会话管理 + 模型配置 + 检索参数
# ============================================================================

with st.sidebar:
    st.title("🔍 Enterprise RAG")

    # ── 新建会话按钮 ──
    mgr = _session_mgr()
    if st.button("➕ 新建会话", use_container_width=True):
        sid = mgr.create_session()
        st.session_state.current_session_id = sid
        st.session_state.session_messages = []
        st.rerun()

    # ── 会话列表（仿 DeepSeek 侧边栏列表样式）──
    sessions = mgr.list_sessions()
    if sessions:
        st.markdown("---")
        st.caption(f"共 {len(sessions)} 个会话")
        for s in sessions:
            is_active = s["id"] == st.session_state.current_session_id
            btn_label = f"{'▸ ' if is_active else '  '}{s['name']}"
            btn_type = "primary" if is_active else "secondary"
            cols = st.columns([6, 1, 1])
            with cols[0]:
                if st.button(btn_label, key=f"sess_{s['id']}", use_container_width=True, type=btn_type):
                    if not is_active:
                        _switch_session(s["id"])
                        st.rerun()
            with cols[1]:
                if st.button("✏️", key=f"ren_trigger_{s['id']}"):
                    st.session_state[f"_renaming_{s['id']}"] = True
            with cols[2]:
                if st.button("🗑️", key=f"del_{s['id']}"):
                    mgr.delete_session(s["id"])
                    if st.session_state.current_session_id == s["id"]:
                        st.session_state.current_session_id = ""
                        st.session_state.session_messages = []
                    st.rerun()

            if st.session_state.get(f"_renaming_{s['id']}"):
                new_name = st.text_input(
                    "新名称", placeholder=s["name"], key=f"ren_input_{s['id']}",
                    label_visibility="collapsed",
                )
                if st.button("确认", key=f"ren_ok_{s['id']}"):
                    if new_name:
                        mgr.rename_session(s["id"], new_name)
                    st.session_state.pop(f"_renaming_{s['id']}", None)
                    st.rerun()
    else:
        st.info("暂无会话，点击上方按钮新建")

    # ── 模型配置 ──
    st.divider()
    st.header("⚙️ 模型配置")
    provider = st.selectbox("LLM 提供商", ["deepseek", "openai", "dashscope", "ollama"], index=0)
    model = st.text_input("模型名称", value="deepseek-v4-pro")
    if st.button("🔄 切换模型", use_container_width=True):
        try:
            ModelFactory.clear_cache()
            st.cache_resource.clear()
            st.success(f"已切换至 {provider}/{model}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.header("🔍 检索参数")
    top_k = st.slider("Top-K 结果数", 1, 20, 5)
    use_cache_flag = st.checkbox("启用缓存", value=True)

    cache_svc = _cache_service()
    cache_stats = cache_svc.stats()
    cache_backend = cache_stats.get("backend", "unknown")
    mem_entries = cache_stats.get("memory_entries", 0)
    sem_entries = cache_stats.get("semantic_entries", 0)
    sem_enabled = cache_stats.get("semantic_enabled", False)

    if use_cache_flag:
        st.success(f"✅ 缓存已启用 | 后端: **{cache_backend}**")
        st.caption(f"内存缓存: {mem_entries} 条 | 语义缓存: {'✅' if sem_enabled else '❌'} ({sem_entries} 条)")
    else:
        st.warning("⚠️ 缓存已禁用，每次提问将重新检索")

    if st.button("🗑️ 清除缓存", use_container_width=True):
        try:
            count = cache_svc.clear()
            st.success(f"已清除 {count} 条缓存")
            st.rerun()
        except Exception as e:
            st.error(f"清除缓存失败: {e}")

# ============================================================================
# 主区域：在线检索 | 系统指标
# ============================================================================

tab_chat, tab_metrics = st.tabs(["💬 在线检索", "📊 系统指标"])

with tab_chat:
    if not st.session_state.current_session_id:
        st.info("👈 请在侧边栏新建或选择一个会话开始提问")
    else:
        # ── 渲染历史消息 ──
        chat_container = st.container()
        with chat_container:
            for m in st.session_state.session_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    if m.get("sources"):
                        with st.expander(f"📎 引用来源 ({len(m['sources'])} 条)", expanded=False):
                            for i, s in enumerate(m["sources"], 1):
                                sc = s.get("score", 0)
                                em = "🟢" if sc > 0.7 else ("🟡" if sc > 0.4 else "🔴")
                                st.markdown(f"{em} **{i}. {s.get('source','?')}** | score={sc:.3f}")

        # ── 处理挂起的问题（在 rerun 后继续流式输出）──
        if st.session_state.generating and st.session_state.pending_question:
            q = st.session_state.pending_question
            sid = st.session_state.current_session_id

            with chat_container:
                with st.chat_message("assistant"):
                    status_placeholder = st.empty()
                    status_placeholder.markdown("*🔍 正在检索相关知识库...*")
                    output_placeholder = st.empty()

                    ch = _chain()
                    ch._top_k = top_k
                    ch._use_cache = use_cache_flag

                    tracker = _tracker()
                    tracker.start_request(q)
                    t0 = time.perf_counter()

                    full_answer = ""
                    sources = []
                    error_msg = None

                    try:
                        for event in ch.answer_stream(q):
                            etype = event["type"]

                            if etype == "status":
                                status_placeholder.markdown(f"*{event['content']}*")

                            elif etype == "token":
                                if full_answer == "":
                                    status_placeholder.empty()
                                full_answer += event["content"]
                                output_placeholder.markdown(full_answer + "▌")

                            elif etype == "sources":
                                sources = event["content"]

                            elif etype == "done":
                                pass

                            elif etype == "error":
                                error_msg = event["content"]

                    except Exception as e:
                        error_msg = f"❌ 问答失败: {e}"

                    if error_msg:
                        status_placeholder.empty()
                        output_placeholder.markdown(error_msg)
                        full_answer = error_msg

                    else:
                        status_placeholder.empty()
                        output_placeholder.markdown(full_answer)

                    # 显示引用来源
                    if sources:
                        with st.expander(f"📎 引用来源 ({len(sources)} 条)", expanded=False):
                            for i, s in enumerate(sources, 1):
                                sc = s.get("score", 0)
                                em = "🟢" if sc > 0.7 else ("🟡" if sc > 0.4 else "🔴")
                                st.markdown(f"{em} **{i}. {s.get('source','?')}** | score={sc:.3f}")

                    # 记录指标
                    try:
                        from src.evaluation.metrics import calculate_tokens
                        tracker.end_request(
                            num_retrieved=len(sources),
                            tokens_input=calculate_tokens(q),
                            tokens_output=calculate_tokens(full_answer),
                        )
                    except Exception:
                        pass

                    # 持久化消息
                    mgr.add_message(sid, "assistant", full_answer, sources)
                    st.session_state.session_messages.append(
                        {"role": "assistant", "content": full_answer, "sources": sources}
                    )

                    # 重置生成状态
                    st.session_state.generating = False
                    st.session_state.pending_question = None
                    st.rerun()

        # ── 聊天输入框（生成中时禁用）──
        is_generating = st.session_state.generating
        if is_generating:
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown("*⏳ 正在思考中，请稍候...*")

        q = st.chat_input("输入问题..." if not is_generating else "正在回答中，请稍候...", disabled=is_generating)
        if q and not is_generating:
            sid = st.session_state.current_session_id

            # 记录用户消息
            st.session_state.session_messages.append({"role": "user", "content": q, "sources": []})
            mgr.add_message(sid, "user", q, [])

            # 设置生成状态，禁止新输入
            st.session_state.generating = True
            st.session_state.pending_question = q
            st.rerun()


with tab_metrics:
    st.subheader("📊 系统指标监控")
    if st.button("🔄 刷新指标"):
        st.rerun()
    t = _tracker()
    s = t.summary(); h = t.get_history(100)
    ca, cb, cc, cd = st.columns(4)
    ca.metric("总请求数", s.get("total_requests", 0))
    cb.metric("平均延迟 (s)", s.get("avg_latency") or 0)
    cc.metric("P50 延迟 (s)", s.get("p50_latency") or 0)
    cd.metric("P99 延迟 (s)", s.get("p99_latency") or 0)
    if h:
        st.divider()
        st.subheader("📈 延迟趋势")
        st.line_chart([x.get("latency_seconds", 0) for x in h], height=200)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📈 Token 用量")
            st.line_chart(pd.DataFrame({"输入Token": [x.get("tokens_input", 0) for x in h], "输出Token": [x.get("tokens_output", 0) for x in h]}), height=200)
        with c2:
            st.subheader("📈 召回率 / 精确率")
            st.line_chart(pd.DataFrame({"召回率": [x.get("recall") or 0 for x in h], "精确率": [x.get("precision") or 0 for x in h]}), height=200)
    else:
        st.info("暂无请求历史，开始提问后将在此展示指标")
