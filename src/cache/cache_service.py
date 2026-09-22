# -*- coding: utf-8 -*-
"""
缓存服务 —— 精确匹配 + 语义缓存双层策略（统一基于 Redis）。

第一层（精确匹配）：问题文本 MD5 → Redis hash，速度极快。
第二层（语义缓存）：问题 embedding → Redis RediSearch 向量相似度，捕获语义等价问法。

Redis 不可用时降级到内存 LRU（仅精确匹配，语义层跳过）。
Redis Stack 不可用时语义层静默跳过，精确层正常工作。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from src.utils.config import get_config

logger = logging.getLogger(__name__)

_SEMANTIC_INDEX = "rag:semantic_idx_v2"
_SEMANTIC_PREFIX = "rag:sem:"


class CacheService:
    """问答缓存服务（精确 + 语义双层）。

    Usage::

        cache = CacheService()
        cache.set("问题", {"answer": "回答"})
        result = cache.get("问题")           # 精确命中
        result = cache.get("类似问法")        # 语义命中
    """
    def __init__(
        self,
        max_memory_entries: int = 1000,
        default_ttl: int = 86400,
    ) -> None:
        """
        Args:
            max_memory_entries: 内存 LRU 最大条目数。
            default_ttl:        默认过期时间（秒，仅 Redis 模式生效）。
        """
        self._max_memory = max_memory_entries
        self._default_ttl = default_ttl
        self._redis = None
        self._redis_available: Optional[bool] = None  # None = 未检测
        self._semantic_index_ready: Optional[bool] = None

        # 内存 LRU 缓存
        self._memory: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    # ── Redis 连接管理 ────────────────────────────────────────────────

    def _init_redis(self) -> bool:
        """尝试连接 Redis，返回是否可用。"""
        if self._redis_available is not None:
            return self._redis_available

        cache_cfg = get_config().cache
        if not cache_cfg.enabled:
            self._redis_available = False
            logger.info("缓存已禁用（config.yaml 中 cache.enabled=false）")
            return False

        try:
            import redis
            self._redis = redis.Redis(
                host=cache_cfg.redis_host,
                port=cache_cfg.redis_port,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=False,
            )
            self._redis.ping()
            self._redis_available = True
            logger.info("Redis 缓存就绪: %s:%s", cache_cfg.redis_host, cache_cfg.redis_port)
            return True
        except Exception as exc:
            self._redis_available = False
            self._redis = None
            logger.warning("Redis 不可用 (%s)，使用内存 LRU 回退", exc)
            return False

    # ── 语义缓存层管理 ────────────────────────────────────────────────

    def _init_semantic_index(self) -> bool:
        """确保 Redis 中存在向量索引，返回是否可用。"""
        if self._semantic_index_ready is not None:
            return self._semantic_index_ready

        if not self._init_redis():
            self._semantic_index_ready = False
            return False

        cache_cfg = get_config().cache
        if not cache_cfg.semantic_enabled:
            self._semantic_index_ready = False
            return False

        try:
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from redis.commands.search.field import (
                VectorField,
                TextField,
                NumericField,
                TagField,
            )

            try:
                self._redis.ft(_SEMANTIC_INDEX).info()
                self._semantic_index_ready = True
                return True
            except Exception:
                pass

            schema = (
                VectorField(
                    "vec",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": cache_cfg.vec_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TextField("question"),
                TagField("scope"),
                NumericField("created_at"),
            )
            definition = IndexDefinition(
                prefix=[_SEMANTIC_PREFIX],
                index_type=IndexType.HASH,
            )
            self._redis.ft(_SEMANTIC_INDEX).create_index(
                fields=schema, definition=definition
            )
            self._semantic_index_ready = True
            logger.info("Redis 语义向量索引已创建: %s (dim=%d)", _SEMANTIC_INDEX, cache_cfg.vec_dim)
            return True
        except Exception as exc:
            self._semantic_index_ready = False
            logger.warning("Redis 语义索引创建失败（需要 Redis Stack）: %s", exc)
            return False

    def _get_embedding(self, question: str) -> Optional[List[float]]:
        try:
            import numpy as np
            from src.llms.models import ModelFactory
            vec = ModelFactory.get_embedding().embed_query(question)
            if isinstance(vec, np.ndarray):
                return vec.tolist()
            return list(vec)
        except Exception as exc:
            logger.warning("语义缓存 embedding 失败: %s", exc)
            return None

    @staticmethod
    def _float_list_to_bytes(vec: List[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    # ── 公开接口 ──────────────────────────────────────────────────────

    def get(
        self,
        question: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """查询缓存（精确匹配优先，未命中再语义匹配）。"""
        # ── 第一层：精确匹配 ──
        exact_result = self._get_exact(question, user_id)
        if exact_result is not None:
            logger.info("精确缓存命中: %s...", question[:40])
            return exact_result

        # ── 第二层：语义匹配 ──
        logger.info("语义缓存初始化: %s", self._init_semantic_index())
        if self._init_semantic_index():
            semantic_result = self._get_semantic(
                question,
                self._user_scope(user_id),
            )
            if semantic_result is not None:
                return semantic_result

        return None

    def set(
            self,
            question: str,
            value: Dict[str, Any],
            ttl: Optional[int] = None,
            user_id: Optional[str] = None,
    ) -> None:
        """写入缓存（精确层 + 语义层同时写入）。"""
        ttl = ttl or get_config().cache.ttl_seconds
        self._set_exact(question, value, ttl, user_id)

        if self._init_semantic_index():
            self._set_semantic(
                question,
                value,
                ttl,
                self._user_scope(user_id),
            )

    def clear(self, pattern: Optional[str] = None) -> int:
        """清除缓存（精确层 + 语义层同时清空）。"""
        count = 0

        if self._init_redis():
            try:
                if pattern:
                    cursor = 0
                    while True:
                        cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                        if keys:
                            count += self._redis.delete(*keys)
                        if cursor == 0:
                            break
                else:
                    all_keys = self._redis.keys("rag:*")
                    if all_keys:
                        count = self._redis.delete(*all_keys)
                    self._semantic_index_ready = None
                logger.info("Redis 缓存清除: %d 条", count)
            except Exception as exc:
                logger.warning("Redis 清除异常: %s", exc)

        with self._lock:
            mem_count = len(self._memory)
            self._memory.clear()
            count += mem_count
        if mem_count:
            logger.info("内存缓存清除: %d 条", mem_count)

        return count

    def stats(self) -> Dict[str, Any]:
        redis_ok = self._init_redis()
        with self._lock:
            mem_entries = len(self._memory)

        info: Dict[str, Any] = {
            "backend": "redis" if redis_ok else "memory",
            "memory_entries": mem_entries,
            "max_memory_entries": self._max_memory,
            "semantic_enabled": bool(self._semantic_index_ready),
        }

        if redis_ok and self._redis:
            try:
                info["redis_dbsize"] = self._redis.dbsize()
                sem_keys = self._redis.keys(f"{_SEMANTIC_PREFIX}*")
                info["semantic_entries"] = len(sem_keys)
            except Exception:
                pass

        return info

    # ── 内部：精确匹配层 ──────────────────────────────────────────────

    def _get_exact(
        self,
        question: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        key = self._make_key(question, user_id)

        if self._init_redis():
            try:
                raw = self._redis.get(key)
                if raw:
                    data = raw if isinstance(raw, str) else raw.decode("utf-8")
                    return json.loads(data)
            except Exception as exc:
                logger.warning("Redis 精确读取异常: %s", exc)

        with self._lock:
            entry = self._memory.get(key)
            if entry:
                self._memory.move_to_end(key)
                return entry["value"]

        return None

    def _set_exact(
            self,
            question: str,
            value: Dict[str, Any],
            ttl: Optional[int] = None,
            user_id: Optional[str] = None,
    ) -> None:
        key = self._make_key(question, user_id)
        ttl = ttl or self._default_ttl

        if self._init_redis():
            try:
                self._redis.setex(key, ttl, json.dumps(value, ensure_ascii=False).encode("utf-8"))
                logger.info("Redis 精确写入成功: %s...", value)
                return
            except Exception as exc:
                logger.warning("Redis 精确写入异常: %s", exc)

        with self._lock:
            while len(self._memory) >= self._max_memory:
                self._memory.popitem(last=False)
            self._memory[key] = {
                "value": value,
                "expire_at": time.time() + ttl,
            }
            self._memory.move_to_end(key)

    # ── 内部：语义缓存层（Redis RediSearch 向量检索） ─────────────────

    def _get_semantic(
        self,
        question: str,
        scope: str,
    ) -> Optional[Dict[str, Any]]:
        vec = self._get_embedding(question)
        if vec is None:
            return None

        try:
            from redis.commands.search.query import Query

            cfg = get_config().cache
            threshold = cfg.similarity_threshold
            vec_bytes = self._float_list_to_bytes(vec)

            q = (
                Query(f"@scope:{{{scope}}}=>[KNN 1 @vec $query_vec AS score]")
                .return_fields("score", "question", "scope", "value")
                .sort_by("score", asc=False)
                .paging(0, 1)
                .dialect(2)
            )
            results = self._redis.ft(_SEMANTIC_INDEX).search(
                q, query_params={"query_vec": vec_bytes}
            )

            if not results.docs:
                return None

            doc = results.docs[0]
            score = float(doc.score) if hasattr(doc, "score") else 0.0
            cosine_sim = 1.0 - score

            if cosine_sim >= threshold:
                cached_value = json.loads(doc.value)
                logger.info(
                    "语义缓存命中 (sim=%.4f): '%s' ≈ '%s'",
                    cosine_sim,
                    question[:30],
                    doc.question[:30] if hasattr(doc, "question") else "",
                )
                return cached_value

        except Exception as exc:
            logger.warning("语义缓存查询异常: %s", exc)

        return None

    def _set_semantic(
            self,
            question: str,
            value: Dict[str, Any],
            ttl: Optional[int] = None,
            scope: str = "public",
    ) -> None:
        vec = self._get_embedding(question)
        if vec is None:
            return

        try:
            cfg = get_config().cache
            ttl = ttl or self._default_ttl
            raw_key = f"{scope}:{question.strip().lower()}"
            key = f"{_SEMANTIC_PREFIX}{hashlib.md5(raw_key.encode('utf-8')).hexdigest()}"

            mapping = {
                "vec": self._float_list_to_bytes(vec),
                "question": question,
                "scope": scope,
                "value": json.dumps(value, ensure_ascii=False),
                "created_at": str(int(time.time())),
            }
            self._redis.hset(key, mapping=mapping)
            self._redis.expire(key, ttl)
            logger.info("语义缓存写入: %s...", question[:40])
        except Exception as exc:
            logger.warning("语义缓存写入异常: %s", exc)

    # ── 工具方法 ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_question(question: str) -> str:
        return " ".join(question.strip().lower().split())

    @staticmethod
    def _user_scope(user_id: Optional[str]) -> str:
        if not user_id:
            return "public"
        safe_user_id = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
        return f"user_{safe_user_id}"

    @classmethod
    def _make_key(
        cls,
        question: str,
        user_id: Optional[str] = None,
    ) -> str:
        normalized = cls._normalize_question(question)
        raw = f"{cls._user_scope(user_id)}:{normalized}"
        return f"rag:qa:{hashlib.md5(raw.encode('utf-8')).hexdigest()}"
