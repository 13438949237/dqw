# -*- coding: utf-8 -*-
"""RAG 系统监测面板 —— Token 消耗 / 成本 / 检索质量 / 答案质量。"""
from __future__ import annotations
import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.evaluation.metrics import MetricsTracker, calculate_tokens
from src.evaluation.monitor import (
    LangfuseTracker, CostCalculator, RagasEvaluator, MonitorPipeline,
)

st.set_page_config(page_title="系统监测", page_icon="📊", layout="wide")
st.title("📊 系统监测面板")
st.caption("Token 消耗、成本分析、检索质量、答案质量")

tracker_m = MetricsTracker()
tracker_lf = LangfuseTracker()
cost_calc = CostCalculator()
ragas_eval = RagasEvaluator()
pipeline = MonitorPipeline()

# ── 指标卡片行 ──
ms = tracker_m.summary()
ls = tracker_lf.summary()
col1, col2, col3, col4 = st.columns(4)
col1.metric("📝 总请求数", ms.get("total_requests", 0))
col2.metric("⏱️ 平均延迟(s)", ms.get("avg_latency") or 0)
col3.metric("🔤 总 Token 入", f"{ls.get('total_input_tokens',0):,}")
col4.metric("💰 累计成本($)", f"{ls.get('total_cost_usd',0):.6f}")

# ── Token / 成本趋势 ──
st.divider()
st.subheader("📈 Token 消耗与成本趋势")
traces = tracker_lf.get_traces(100)
if traces:
    df_traces = pd.DataFrame(traces)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("总输入 Token", f"{ls.get('total_input_tokens',0):,}")
    with c2:
        st.metric("总输出 Token", f"{ls.get('total_output_tokens',0):,}")
    with c3:
        st.metric("累计成本($)", f"{ls.get('total_cost_usd',0):.6f}")
    st.line_chart(df_traces[["input_tokens","output_tokens"]].tail(50), height=250)
    st.subheader("💵 每次调用成本（美元）")
    st.line_chart(df_traces["cost_usd"].tail(50), height=150)
else:
    st.info("暂无追踪数据，开始问答后将自动记录")

# ── 检索质量 ──
st.divider()
st.subheader("🔍 检索质量")
mh = tracker_m.get_history(100)
if mh:
    df_m = pd.DataFrame(mh)
    c1, c2, c3 = st.columns(3)
    c1.metric("平均召回率", ms.get("avg_recall") or "-")
    c2.metric("平均精确率", ms.get("avg_precision") or "-")
    c3.metric("平均检索数", f"{sum(h.get('num_retrieved',0) for h in mh) // max(len(mh),1)}")
    st.line_chart(pd.DataFrame({
        "召回率": [h.get("recall") or 0 for h in mh],
        "精确率": [h.get("precision") or 0 for h in mh],
    }).tail(50), height=200)
else:
    st.info("暂无检索数据")

# ── 答案质量 (Ragas) ──
st.divider()
st.subheader("✅ 答案质量（Ragas 评估）")
scores_from_traces = [t.get("ragas_scores", {}) for t in traces if t.get("ragas_scores")]
if scores_from_traces:
    faith_vals = [s.get("faithfulness", 0) for s in scores_from_traces if s.get("faithfulness")]
    relev_vals = [s.get("answer_relevancy", 0) for s in scores_from_traces if s.get("answer_relevancy")]
    recall_vals = [s.get("context_recall", 0) for s in scores_from_traces if s.get("context_recall")]
    c1, c2, c3 = st.columns(3)
    c1.metric("忠实度", f"{sum(faith_vals)/len(faith_vals):.3f}" if faith_vals else "-")
    c2.metric("答案相关性", f"{sum(relev_vals)/len(relev_vals):.3f}" if relev_vals else "-")
    c3.metric("上下文召回", f"{sum(recall_vals)/len(recall_vals):.3f}" if recall_vals else "-")
    chart_data: dict = {}
    if faith_vals:
        chart_data["忠实度"] = faith_vals
    if relev_vals:
        chart_data["答案相关性"] = relev_vals
    if recall_vals:
        chart_data["上下文召回"] = recall_vals
    if chart_data:
        st.line_chart(pd.DataFrame(chart_data).tail(50), height=200)
else:
    st.info("暂无 Ragas 评估数据，需安装 ragas 并配置 API key 后自动启用")

# ── 延迟分布 ──
st.divider()
st.subheader("⏱️ 延迟趋势")
if mh:
    st.line_chart([h.get("latency_seconds", 0) for h in mh[-50:]], height=200)

# ── 最近追踪表 ──
st.divider()
st.subheader("📋 最近追踪记录")
if traces:
    table = [{
        "时间": (t.get("timestamp","") or "")[:19],
        "查询": (t.get("query","") or "")[:25],
        "模型": t.get("model",""),
        "入Token": t.get("input_tokens",0),
        "出Token": t.get("output_tokens",0),
        "延迟(s)": t.get("latency_seconds",0),
        "成本($)": f"{t.get('cost_usd',0):.6f}",
    } for t in traces[-20:]]
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

if st.button("🗑️ 清空追踪数据"):
    tracker_lf.clear()
    st.rerun()
