# -*- coding: utf-8 -*-
"""
统一模型工厂。

提供 get_embedding()、get_llm()、get_reranker() 三个入口，根据 config.yaml
中的 provider / model_name 自动路由到对应后端，内置回退链与异常处理。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from src.utils.config import ApiKeysConfig, get_config

logger = logging.getLogger(__name__)

# ============================================================================
# 辅助类型
# ============================================================================


class BaseReranker(ABC):
    """重排序模型抽象基类。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        对候选文档列表重新排序。

        Args:
            query:     用户查询字符串。
            documents: 候选文档列表（按原始顺序）。
            top_k:     返回的最高排名文档数。

        Returns:
            [(原始索引, 相关性分数), ...]，按分数降序排列。
        """
        ...


# ============================================================================
# 内部实现：重排序器
# ============================================================================


class _LocalCrossEncoderReranker(BaseReranker):
    """基于 sentence-transformers 的本地 CrossEncoder 重排序器。"""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "本地重排序需要安装 sentence-transformers，请执行: "
                "pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        logger.info("加载本地重排序模型: %s", model_name)
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores: List[float] = self._model.predict(pairs).tolist()  # type: ignore[union-attr]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class _CohereReranker(BaseReranker):
    """基于 Cohere 云端 API 的重排序器。"""

    def __init__(self, model_name: str, api_key: str):
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "Cohere 重排序需要安装 cohere 包，请执行: pip install cohere"
            ) from exc

        self._model_name = model_name
        self._client = cohere.Client(api_key)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []

        result = self._client.rerank(
            model=self._model_name,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
        )
        return [(r.index, r.relevance_score) for r in result.results]


class _JinaReranker(BaseReranker):
    """基于 Jina AI 云端 API 的重排序器。"""

    def __init__(self, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = "https://api.jina.ai/v1/rerank"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        if not documents:
            return []

        import json
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "Jina 重排序需要安装 httpx，请执行: pip install httpx"
            ) from exc

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_k, len(documents)),
                },
            )
            resp.raise_for_status()
            results = resp.json()["results"]

        return [(r["index"], r["relevance_score"]) for r in results]


# ============================================================================
# 模型工厂
# ============================================================================


class ModelFactory:
    """
    统一的模型工厂，根据配置提供 Embedding / LLM / Reranker 实例。

    使用方式::

        emb = ModelFactory.get_embedding()
        llm = ModelFactory.get_llm()
        reranker = ModelFactory.get_reranker()
    """

    _embedding_cache: dict = {}
    _llm_cache: dict = {}
    _reranker_cache: dict = {}

    # ── Embedding ───────────────────────────────────────────────────────

    @classmethod
    def get_embedding(
        cls,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        获取嵌入模型实例。

        Args:
            provider:   提供商标识，未指定时使用 config.yaml 中的默认值。
            model_name: 模型名称，未指定时使用 config.yaml 中的默认值。
            **kwargs:   传递给底层模型构造函数的额外参数。

        Returns:
            LangChain Embeddings 实例。

        Raises:
            RuntimeError: 所有回退方案均失败时抛出。
        """
        cfg = get_config()
        provider = provider or cfg.models.embedding.provider
        model_name = model_name or cfg.models.embedding.model_name
        cache_key = f"{provider}:{model_name}"

        if cache_key in cls._embedding_cache:
            return cls._embedding_cache[cache_key]

        # 回退链：首选 → 备选1 → 备选2
        fallback_chain = [
            (provider, model_name),
            ("openai", "text-embedding-3-small"),
            ("local", "all-MiniLM-L6-v2"),
        ]

        last_error: Optional[Exception] = None

        for fb_provider, fb_model in fallback_chain:
            try:
                print(f"尝试初始化嵌入模型: {fb_provider}:{fb_model}")
                instance = cls._build_embedding(fb_provider, fb_model, **kwargs)
                cls._embedding_cache[cache_key] = instance
                logger.info(
                    "嵌入模型就绪: provider=%s model=%s",
                    fb_provider,
                    fb_model,
                )
                return instance
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "嵌入模型 (%s:%s) 初始化失败: %s，尝试回退",
                    fb_provider,
                    fb_model,
                    exc,
                )

        raise RuntimeError(
            f"无法初始化任何嵌入模型，最后错误: {last_error}"
        ) from last_error

    @staticmethod
    def _build_embedding(provider: str, model_name: str, **kwargs):
        api_keys: ApiKeysConfig = get_config().api_keys

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_keys.openai_api_key,
                base_url=api_keys.openai_base_url,
                **kwargs,
            )

        if provider == "dashscope":
            from langchain_community.embeddings import DashScopeEmbeddings
            return DashScopeEmbeddings(
                model=model_name,
                dashscope_api_key=api_keys.dashscope_api_key,
                **kwargs,
            )

        if provider in ("local", "huggingface", "sentence-transformers"):
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model_name,
                **kwargs,
            )

        raise ValueError(f"不支持的嵌入模型提供商: {provider}")

    # ── LLM ─────────────────────────────────────────────────────────────

    @classmethod
    def get_llm(
        cls,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        获取大语言模型实例。

        Args:
            provider:    提供商标识。
            model_name:  模型名称。
            temperature: 生成温度，未指定时使用 config.yaml 默认值。
            **kwargs:    传递给底层模型构造函数的额外参数。

        Returns:
            LangChain BaseChatModel 实例。

        Raises:
            RuntimeError: 所有回退方案均失败时抛出。
        """
        cfg = get_config()
        provider = provider or cfg.models.llm.provider
        model_name = model_name or cfg.models.llm.model_name
        temperature = temperature if temperature is not None else cfg.models.llm.temperature
        cache_key = f"{provider}:{model_name}"

        if cache_key in cls._llm_cache:
            return cls._llm_cache[cache_key]

        # 回退链
        fallback_chain = [
            (provider, model_name),
            ("openai", "gpt-4o"),
            ("ollama", "llama3.2"),
        ]

        last_error: Optional[Exception] = None

        for fb_provider, fb_model in fallback_chain:
            try:
                instance = cls._build_llm(
                    fb_provider, fb_model, temperature, **kwargs
                )
                cls._llm_cache[cache_key] = instance
                logger.info(
                    "LLM 就绪: provider=%s model=%s",
                    fb_provider,
                    fb_model,
                )
                return instance
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM (%s:%s) 初始化失败: %s，尝试回退",
                    fb_provider,
                    fb_model,
                    exc,
                )

        raise RuntimeError(
            f"无法初始化任何 LLM，最后错误: {last_error}"
        ) from last_error

    @staticmethod
    def _build_llm(
        provider: str,
        model_name: str,
        temperature: float,
        **kwargs,
    ):
        api_keys: ApiKeysConfig = get_config().api_keys

        # OpenAI
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=kwargs.pop("api_key", api_keys.openai_api_key),
                **kwargs,
            )

        # DeepSeek
        if provider == "deepseek":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=kwargs.pop("api_key", api_keys.deepseek_api_key),
                base_url=kwargs.pop("base_url", api_keys.deepseek_base_url),
                **kwargs,
            )

        # 阿里 DashScope（通义系列）
        if provider == "dashscope":
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=model_name,
                temperature=temperature,
                dashscope_api_key=kwargs.pop(
                    "api_key", api_keys.dashscope_api_key
                ),
                **kwargs,
            )

        # 本地 Ollama
        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError:
                from langchain_community.chat_models import ChatOllama

            return ChatOllama(
                model=model_name,
                temperature=temperature,
                **kwargs,
            )

        raise ValueError(f"不支持的 LLM 提供商: {provider}")

    # ── Reranker ────────────────────────────────────────────────────────

    @classmethod
    def get_reranker(
        cls,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseReranker:
        """
        获取重排序模型实例。

        Args:
            provider:   提供商（huggingface / cohere / jina）。
            model_name: 模型名称。
            top_k:      返回的最高排名数。
            **kwargs:   传递给底层构造函数的额外参数。

        Returns:
            BaseReranker 子类实例。

        Raises:
            RuntimeError: 所有回退方案均失败时抛出。
        """
        cfg = get_config()
        provider = provider or cfg.models.reranker.provider
        model_name = model_name or cfg.models.reranker.model_name
        top_k = top_k or cfg.models.reranker.top_k
        cache_key = f"{provider}:{model_name}"

        if cache_key in cls._reranker_cache:
            return cls._reranker_cache[cache_key]

        # 回退链：HuggingFace 本地 → Cohere API → Jina API
        fallback_chain = [
            (provider, model_name),
            ("huggingface", "BAAI/bge-reranker-v2-m3"),
        ]

        last_error: Optional[Exception] = None

        for fb_provider, fb_model in fallback_chain:
            try:
                instance = cls._build_reranker(
                    fb_provider, fb_model, **kwargs
                )
                cls._reranker_cache[cache_key] = instance
                logger.info(
                    "重排序模型就绪: provider=%s model=%s",
                    fb_provider,
                    fb_model,
                )
                return instance
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "重排序模型 (%s:%s) 初始化失败: %s，尝试回退",
                    fb_provider,
                    fb_model,
                    exc,
                )

        raise RuntimeError(
            f"无法初始化任何重排序模型，最后错误: {last_error}"
        ) from last_error

    @staticmethod
    def _build_reranker(
        provider: str,
        model_name: str,
        **kwargs,
    ) -> BaseReranker:
        api_keys: ApiKeysConfig = get_config().api_keys

        if provider in ("huggingface", "local"):
            return _LocalCrossEncoderReranker(model_name)

        if provider == "cohere":
            cohere_key = kwargs.pop("api_key", api_keys.cohere_api_key)
            if not cohere_key:
                raise ValueError(
                    "Cohere API 密钥未设置，请在 .env 中配置 COHERE_API_KEY"
                )
            return _CohereReranker(model_name, api_key=cohere_key)

        if provider == "jina":
            jina_key = kwargs.pop("api_key", api_keys.jina_api_key)
            if not jina_key:
                raise ValueError(
                    "Jina API 密钥未设置，请在 .env 中配置 JINA_API_KEY"
                )
            return _JinaReranker(model_name, api_key=jina_key)

        raise ValueError(f"不支持的重排序提供商: {provider}")

    # ── 缓存管理 ────────────────────────────────────────────────────────

    @classmethod
    def clear_cache(cls) -> None:
        """清除所有模型缓存，下次调用时将重新初始化。"""
        cls._embedding_cache.clear()
        cls._llm_cache.clear()
        cls._reranker_cache.clear()
        logger.info("模型工厂缓存已清除")
