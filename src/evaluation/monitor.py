# -*- coding: utf-8 -*-
"""
监测模块 —— Ragas 评估 + Langfuse 追踪 + 成本计算。

提供：
  RagasEvaluator   — 使用 Ragas 计算 faithfulness / answer_relevancy / context_recall
  LangfuseTracker  — LLM 调用追踪（token 消耗、延迟、成本）
  CostCalculator    — 按模型计费
  MonitorPipeline   — 一键评估管线
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from langfuse import Langfuse

from src.utils.config import ApiKeysConfig, get_config

logger = logging.getLogger(__name__)


# ============================================================================
# 成本计算
# ============================================================================

# 价格：美元 / 百万 token（输入, 输出）
_PRICING: Dict[str, tuple] = {
    "deepseek/deepseek-v4-pro":      (0.14,  0.28),
    "deepseek/deepseek-chat":        (0.14,  0.28),
    "openai/gpt-4o":                 (2.50, 10.00),
    "openai/gpt-4o-mini":            (0.15,  0.60),
    "openai/gpt-3.5-turbo":          (0.50,  1.50),
    "dashscope/qwen-max":            (0.40,  1.20),
    "dashscope/qwen-plus":           (0.20,  0.60),
    "ollama/llama3.2":               (0.00,  0.00),
}
_DEFAULT_PRICE = (0.50, 1.50)  # 未知模型默认价格


class CostCalculator:
    """Token 成本计算器。

    Usage::
        calc = CostCalculator()
        cost = calc.calculate("deepseek/deepseek-v4-pro", input_tokens=1200, output_tokens=300)
        # cost = (1200/1e6)*0.14 + (300/1e6)*0.28 = 0.000168 + 0.000084 = 0.000252 USD
    """

    @staticmethod
    def calculate(model: str, input_tokens: int, output_tokens: int) -> float:
        """计算单次调用的成本（美元）。

        Args:
            model:         模型标识（如 "deepseek/deepseek-v4-pro"）。
            input_tokens:  输入 token 数。
            output_tokens: 输出 token 数。

        Returns:
            成本（美元）。
        """
        iprice, oprice = _PRICING.get(model, _DEFAULT_PRICE)
        cost = (input_tokens / 1_000_000) * iprice + (output_tokens / 1_000_000) * oprice
        return round(cost, 8)

    @staticmethod
    def pricing_table() -> Dict[str, tuple]:
        """返回当前价格表（只读）。"""
        return dict(_PRICING)


# ============================================================================
# Ragas 评估器
# ============================================================================


class RagasEvaluator:
    """基于 Ragas 的 RAG 质量评估器。

    评估维度：
      - faithfulness（忠实度）：答案是否仅基于上下文
      - answer_relevancy（答案相关性）：答案是否切题
      - context_recall（上下文召回）：检索到的上下文覆盖了多少 ground truth

    Usage::
        ragas = RagasEvaluator()
        scores = ragas.evaluate(
            questions=["Q1"],
            answers=["A1"],
            contexts=[["ctx1", "ctx2"]],
        )
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    def _check(self) -> bool:
        """懒检测 Ragas 可用性。"""
        if self._available is not None:
            return self._available
        try:
            import ragas  # noqa: F401
            self._available = True
            logger.info("Ragas 评估器可用")
        except ImportError:
            logger.warning("ragas 未安装，评估指标将使用内置 LLM 评判")
            self._available = False
        except Exception as exc:
            logger.warning("Ragas 初始化失败: %s", exc)
            self._available = False
        return self._available

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """使用 Ragas 批量评估。

        Args:
            questions:     用户问题列表。
            answers:       LLM 生成答案列表。
            contexts:      检索到的上下文列表（每个问题对应一个 str 列表）。
            ground_truths: 可选的标准答案列表。

        Returns:
            {"faithfulness": 0.85, "answer_relevancy": 0.72, "context_recall": 0.63, ...}
        """
        if not self._check():
            return {"error": "ragas 不可用，请通过内置 MetricsTracker 获取指标"}

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_recall,
            )

            dataset_dict: Dict[str, Any] = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            if ground_truths:
                dataset_dict["ground_truth"] = ground_truths

            result = evaluate(
                dataset=dataset_dict,
                metrics=[faithfulness, answer_relevancy, context_recall],
            )

            df = result.to_pandas()
            scores: Dict[str, Any] = {
                "faithfulness": round(float(df["faithfulness"].mean()), 4),
                "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
            }
            if "context_recall" in df.columns:
                scores["context_recall"] = round(float(df["context_recall"].mean()), 4)

            logger.info("Ragas 评估完成: %s", scores)
            return scores

        except Exception as exc:
            logger.exception("Ragas 评估异常: %s", exc)
            return {"error": str(exc)}

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """单条评估。"""
        kwargs: Dict[str, Any] = {
            "questions": [question],
            "answers": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            kwargs["ground_truths"] = [ground_truth]
        return self.evaluate(**kwargs)


# ============================================================================
# Langfuse 追踪器
# ============================================================================


@dataclass
class TraceRecord:
    """单次 LLM 调用的追踪记录。"""
    trace_id: str
    timestamp: str = ""
    query: str = ""
    answer: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    ragas_scores: Dict[str, Any] = field(default_factory=dict)


class LangfuseTracker:
    """Langfuse LLM 调用追踪器。

    记录每次 RAG 调用的完整链路信息（token 消耗、延迟、成本），
    支持本地内存存储（Langfuse 不可用时降级）。

    Usage::
        tracker = LangfuseTracker()
        tracker.record(TraceRecord(...))
        traces = tracker.get_traces(limit=50)
    """

    _instance: Optional["LangfuseTracker"] = None

    def __new__(cls) -> "LangfuseTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._traces: List[TraceRecord] = []
        self._max_traces = 10000
        self._langfuse_client: Any = None
        self._lf_available = False
        self._initialized = True
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """尝试初始化 Langfuse 客户端。"""
        try:
            api_keys: ApiKeysConfig = get_config().api_keys
            pk = api_keys.langfuse_public_key
            sk = api_keys.langfuse_secret_key
            base_url = api_keys.langfuse_base_url
            if pk and sk:
                from langfuse import Langfuse
                self._langfuse_client = Langfuse(public_key=pk, secret_key=sk)
                self._lf_available = True
                logger.info("Langfuse 追踪器可用")
            else:
                logger.info("未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY，使用本地内存追踪")
            if pk and sk:
                self._langfuse_client = Langfuse(
                    public_key=pk, secret_key=sk, base_url=base_url,
                )
                self._lf_available = True
                logger.info("Langfuse 追踪器可用")
            else:
                logger.info("未配置 Langfuse 密钥，使用本地内存追踪")
        except ImportError:
            logger.info("langfuse 未安装，使用本地内存追踪")
        except Exception as exc:
            logger.warning("Langfuse 初始化失败: %s", exc)

    def record(self, trace: TraceRecord) -> None:
        """记录一条追踪。

        Args:
            trace: TraceRecord 实例。
        """
        trace.timestamp = datetime.now(timezone.utc).isoformat()
        self._traces.append(trace)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        # 同步写入 Langfuse
        if self._lf_available and self._langfuse_client:
            try:
                lf_trace = self._langfuse_client.trace(
                    name="rag-query",
                    input=trace.query,
                    output=trace.answer,
                    metadata={
                        "model": trace.model,
                        "input_tokens": trace.input_tokens,
                        "output_tokens": trace.output_tokens,
                        "latency_seconds": trace.latency_seconds,
                        "cost_usd": trace.cost_usd,
                        "ragas_scores": trace.ragas_scores,
                    },
                )
                # 记录 generation
                lf_trace.generation(
                    name="llm-generation",
                    model=trace.model,
                    input=trace.query,
                    output=trace.answer,
                    usage={
                        "input": trace.input_tokens,
                        "output": trace.output_tokens,
                    },
                    metadata={"cost_usd": trace.cost_usd},
                )
            except Exception as exc:
                logger.debug("Langfuse 写入失败: %s", exc)

    def get_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近 N 条追踪记录。

        Args:
            limit: 返回条数。

        Returns:
            字典列表。
        """
        recent = self._traces[-limit:]
        return [
            {
                "trace_id": t.trace_id,
                "timestamp": t.timestamp,
                "query": t.query[:80],
                "answer": t.answer[:200],
                "model": t.model,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "latency_seconds": t.latency_seconds,
                "cost_usd": t.cost_usd,
                "ragas_scores": t.ragas_scores,
            }
            for t in recent
        ]

    def summary(self) -> Dict[str, Any]:
        """返回追踪汇总统计。"""
        if not self._traces:
            return {"total_traces": 0}
        traces = self._traces
        total_in = sum(t.input_tokens for t in traces)
        total_out = sum(t.output_tokens for t in traces)
        total_cost = sum(t.cost_usd for t in traces)
        avg_latency = sum(t.latency_seconds for t in traces) / len(traces)
        return {
            "total_traces": len(traces),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_seconds": round(avg_latency, 4),
        }

    def clear(self) -> None:
        """清空本地追踪记录。"""
        self._traces.clear()
        logger.info("追踪记录已清空")


# ============================================================================
# 监测管线
# ============================================================================


class MonitorPipeline:
    """一键评估管线：成本 + Langfuse 追踪 + Ragas 评估。

    Usage::
        pipeline = MonitorPipeline()
        pipeline.evaluate(
            query="Qdrant 分片策略？",
            answer="建议 5 主分片 + 2 副本...",
            contexts=["ctx1", "ctx2"],
            model="deepseek/deepseek-v4-pro",
            input_tokens=500,
            output_tokens=200,
            latency=1.5,
        )
    """

    def __init__(self) -> None:
        self._cost = CostCalculator()
        self._tracker = LangfuseTracker()
        self._ragas = RagasEvaluator()

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency: float = 0.0,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行完整评估管线。

        Returns:
            {
                "trace_id": "...",
                "cost_usd": 0.000252,
                "ragas_scores": {"faithfulness": 0.85, ...},
                "input_tokens": 500,
                "output_tokens": 200,
                "latency_seconds": 1.5,
            }
        """
        trace_id = str(int(time.time() * 1000))
        cost_usd = self._cost.calculate(model, input_tokens, output_tokens)

        # Ragas 评估
        ragas_scores: Dict[str, Any] = {}
        try:
            ragas_scores = self._ragas.evaluate_single(
                question=query,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        except Exception as exc:
            logger.warning("Ragas 评估失败: %s", exc)
            ragas_scores = {"error": str(exc)}

        # Langfuse 追踪
        trace = TraceRecord(
            trace_id=trace_id,
            query=query,
            answer=answer,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            cost_usd=cost_usd,
            ragas_scores=ragas_scores,
        )
        self._tracker.record(trace)

        return {
            "trace_id": trace_id,
            "cost_usd": cost_usd,
            "ragas_scores": ragas_scores,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency,
        }
