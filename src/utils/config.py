# -*- coding: utf-8 -*-
"""
全局配置加载器。

使用 Pydantic 校验 config.yaml 结构，同时通过 python-dotenv 读取 .env 文件中的敏感信息
（如 API 密钥）。对外暴露 load_config() 和 get_config() 两个单例入口。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ── 项目启动时自动加载 .env ────────────────────────────────────────────────
_env_loaded = False


def _ensure_dotenv():
    """在首次调用时从项目根目录加载 .env 文件，避免重复加载。"""
    global _env_loaded
    if _env_loaded:
        return
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.info(".env 文件已加载: %s", env_path)
    else:
        logger.debug("未找到 .env 文件: %s", env_path)
    _env_loaded = True


# ── 子配置模型 ─────────────────────────────────────────────────────────────

class EmbeddingConfig(BaseModel):
    """嵌入模型配置。"""
    provider: str = "dashscope"
    model_name: str = "text-embedding-v4"
    dimension: int = 1536
    batch_size: int = 32


class LLMConfig(BaseModel):
    """大语言模型配置。"""
    provider: str = "deepseek"
    model_name: str = "deepseek-v4-pro"
    temperature: float = 0.1
    max_tokens: int = 2048


class RerankerConfig(BaseModel):
    """重排序模型配置。"""
    provider: str = "huggingface"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 10
    score_threshold: float = 0.0


class ModelsConfig(BaseModel):
    """模型选择聚合。"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class QdrantConfig(BaseModel):
    """Qdrant 向量数据库连接配置（Docker 部署默认 localhost:6333）。"""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    api_key: Optional[str] = None
    collection_name: str = "rag_documents"
    vector_size: int = 1536
    distance: str = "Cosine"


class DocumentConfig(BaseModel):
    """文档处理参数。"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    supported_formats: list[str] = Field(
        default_factory=lambda: ["md", "txt", "pdf", "png", "jpg", "jpeg", "bmp", "tiff", "docx", "pptx", "xlsx", "xls", "html", "csv"]
    )


class RetrievalConfig(BaseModel):
    """检索策略参数。"""
    top_k: int = 20
    retrieval_type: str = "hybrid"
    fusion_method: str = "rrf"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3


class CacheConfig(BaseModel):
    """缓存配置。"""
    enabled: bool = True
    backend: str = "redis"
    redis_host: str = "localhost"
    redis_port: int = 6379
    ttl_seconds: int = 3600
    semantic_enabled: bool = True
    similarity_threshold: float = 0.92
    vec_dim: int = 1024


class LoggingConfig(BaseModel):
    """日志配置。"""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file: str = "logs/rag.log"
    console: bool = True


class ServerConfig(BaseModel):
    """FastAPI 服务配置。"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True


class StreamlitConfig(BaseModel):
    """Streamlit 前端配置。"""
    port: int = 8501
    theme: str = "light"


class PostgresConfig(BaseModel):
    """PostgreSQL 会话持久化配置（Docker 部署）。"""
    host: str = "localhost"
    port: int = 5432
    user: str = "rag_user"
    password: str = "rag_password"
    database: str = "rag_sessions"


class ApiKeysConfig(BaseModel):
    """API 密钥配置 —— 仅从环境变量读取，不写入 config.yaml。"""
    # OpenAI 系
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default=None)
    # 阿里 DashScope
    dashscope_api_key: Optional[str] = Field(default=None)
    # DeepSeek（兼容 OpenAI 协议）
    deepseek_api_key: Optional[str] = Field(default=None)
    deepseek_base_url: Optional[str] = Field(default="https://api.deepseek.com")
    # Cohere / Jina（重排序 API）
    cohere_api_key: Optional[str] = Field(default=None)
    jina_api_key: Optional[str] = Field(default=None)
    # Qdrant
    qdrant_api_key: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def _fill_from_env(self) -> "ApiKeysConfig":
        """用环境变量填充所有未显式传入的字段。"""
        for field_name in self.model_fields:
            current = getattr(self, field_name)
            if current is None or current == "":
                env_val = os.environ.get(field_name.upper())
                if env_val:
                    object.__setattr__(self, field_name, env_val)
        return self


# ── 顶层配置 ────────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    """应用全局配置。"""
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    postgresql: PostgresConfig = Field(default_factory=PostgresConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)


# ── 单例 ────────────────────────────────────────────────────────────────────

_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    加载并缓存全局配置。

    1. 自动从项目根目录加载 .env 文件；
    2. 读取 config.yaml，通过 Pydantic 模型校验；
    3. 注入环境变量中的 API 密钥。

    Args:
        config_path: YAML 配置文件路径，默认使用环境变量 RAG_CONFIG_PATH
                     或项目根目录下的 config.yaml。

    Returns:
        经过完整校验的 AppConfig 实例（单例）。
    """
    global _config

    if _config is not None:
        return _config

    _ensure_dotenv()

    if config_path is None:
        config_path = os.environ.get(
            "RAG_CONFIG_PATH",
            str(Path(__file__).parent.parent.parent / "config.yaml"),
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # api_keys 不出现在 YAML 中，通过环境变量单独构建
    raw.setdefault("api_keys", {})
    _config = AppConfig(**raw)

    logger.info(
        "配置加载完成 (embedding=%s, llm=%s, qdrant=%s:%s)",
        _config.models.embedding.provider,
        _config.models.llm.provider,
        _config.qdrant.host,
        _config.qdrant.port,
    )
    return _config


def get_config() -> AppConfig:
    """获取已缓存的全局配置，未加载时自动触发 load_config()。"""
    if _config is None:
        return load_config()
    return _config
