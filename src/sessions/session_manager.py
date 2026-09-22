# -*- coding: utf-8 -*-
"""会话管理器 —— PostgreSQL 持久化 + Redis 消息缓存。"""
from __future__ import annotations
import json, logging, uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from src.utils.config import get_config

logger = logging.getLogger(__name__)
_SESS_MSG_KEY = "rag:session:msgs:"
_CACHE_TTL = 600


class SessionManager:
    """会话管理器。

    Usage::
        mgr = SessionManager()
        sid = mgr.create_session("新会话")
        mgr.add_message(sid, "user", "问题？", [])
        msgs = mgr.get_messages(sid)
    """

    def __init__(self) -> None:
        self._engine: Optional[Engine] = None
        self._redis = None

    def _get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine
        cfg = get_config().postgresql
        url = f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"
        self._engine = create_engine(url, pool_size=5, max_overflow=10, echo=False)
        self._ensure_tables()
        logger.info("PostgreSQL 会话引擎就绪: %s:%s/%s", cfg.host, cfg.port, cfg.database)
        return self._engine

    def _ensure_tables(self) -> None:
        """创建会话表和消息表（幂等），连接失败时只记日志不抛异常。"""
        try:
            with self._engine.connect() as conn:  # type: ignore[union-attr]
                conn.execute(text("""
                                CREATE TABLE IF NOT EXISTS sessions (
                                    id VARCHAR(36) PRIMARY KEY,
                                    name VARCHAR(200) NOT NULL DEFAULT '',
                                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                                    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                                )
                            """))
                conn.execute(text("""
                                CREATE TABLE IF NOT EXISTS messages (
                                    id VARCHAR(36) PRIMARY KEY,
                                    session_id VARCHAR(36) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                    role VARCHAR(20) NOT NULL,
                                    content TEXT NOT NULL DEFAULT '',
                                    sources JSONB DEFAULT '[]'::jsonb,
                                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                                )
                            """))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, created_at)"))
                conn.commit()
        except Exception as exc:
            logger.warning("PostgreSQL 建表失败（服务未启动？）: %s", exc)

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            from src.cache.cache_service import CacheService
            cs = CacheService()
            if cs._init_redis():
                self._redis = cs._redis
        except Exception as exc:
            logger.debug("Redis 不可用: %s", exc)
        return self._redis

    def _cache_msgs(self, sid: str, msgs: list) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            r.setex(_SESS_MSG_KEY + sid, _CACHE_TTL, json.dumps(msgs, ensure_ascii=False, default=str))
        except Exception:
            pass

    def _get_cached_msgs(self, sid: str):
        r = self._get_redis()
        if r is None:
            return None
        try:
            raw = r.get(_SESS_MSG_KEY + sid)
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _invalidate(self, sid: str) -> None:
        r = self._get_redis()
        if r is None:
            return
        try:
            r.delete(_SESS_MSG_KEY + sid)
        except Exception:
            pass

    # ── 会话 CRUD ────────────────────────────────────────────────────

    def create_session(self, name: str = "") -> str:
        sid = str(uuid.uuid4())
        display = name or f"会话 {datetime.now().strftime('%m-%d %H:%M')}"
        now = datetime.now(timezone.utc)
        with self._get_engine().connect() as conn:
            conn.execute(text("INSERT INTO sessions (id,name,created_at,updated_at) VALUES (:id,:name,:c,:u)"), {"id": sid, "name": display, "c": now, "u": now})
            conn.commit()
        logger.info("会话创建: %s (%s)", display, sid[:8])
        return sid

    def delete_session(self, sid: str) -> bool:
        with self._get_engine().connect() as conn:
            r = conn.execute(text("DELETE FROM sessions WHERE id=:id"), {"id": sid})
            conn.commit()
            ok = r.rowcount > 0
        if ok:
            self._invalidate(sid)
            logger.info("会话删除: %s", sid[:8])
        return ok

    def list_sessions(self) -> List[Dict]:
        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT s.id, s.name, s.created_at, s.updated_at, COUNT(m.id) AS mc FROM sessions s LEFT JOIN messages m ON s.id=m.session_id GROUP BY s.id ORDER BY s.updated_at DESC")).fetchall()
        return [{"id": r.id, "name": r.name, "created_at": r.created_at.isoformat() if r.created_at else "", "updated_at": r.updated_at.isoformat() if r.updated_at else "", "message_count": r.mc} for r in rows]

    # ── 消息操作 ────────────────────────────────────────────────────

    def add_message(self, sid: str, role: str, content: str, sources: list = None) -> str:
        """
        向指定会话添加一条消息并持久化到 PostgreSQL
        sid：会话 ID（UUID 格式
        role：消息角色，"user" 或 "assistant"
        content：消息文本内容
        sources：引用来源列表（可选，默认为 None）
`       return：新消息的 ID（UUID）
        """
        # 1.生成消息 ID 和时间戳
        mid = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        # 2.序列化来源数据
        src_json = json.dumps(sources or [], ensure_ascii=False)
        # 3.数据库写入
        with self._get_engine().connect() as conn:
            conn.execute(text("INSERT INTO messages (id,session_id,role,content,sources,created_at) VALUES (:id,:sid,:role,:content,CAST(:src AS jsonb),:now)"), {"id": mid, "sid": sid, "role": role, "content": content, "src": src_json, "now": now})
            conn.execute(text("UPDATE sessions SET updated_at=:now WHERE id=:sid"), {"now": now, "sid": sid})
            conn.commit()
        # 4.更新 Redis 缓存
        cached = self._get_cached_msgs(sid) or []
        cached.append({"role": role, "content": content, "sources": sources or [], "timestamp": now.isoformat()})
        self._cache_msgs(sid, cached)
        return mid

    def get_messages(self, sid: str, limit: int = 100) -> List[Dict]:
        cached = self._get_cached_msgs(sid)
        if cached is not None:
            return cached[-limit:]
        with self._get_engine().connect() as conn:
            rows = conn.execute(text("SELECT role,content,sources,created_at FROM messages WHERE session_id=:sid ORDER BY created_at ASC LIMIT :lim"), {"sid": sid, "lim": limit}).fetchall()
        msgs = []
        for r in rows:
            srcs = r.sources
            if isinstance(srcs, str):
                try:
                    srcs = json.loads(srcs)
                except Exception:
                    srcs = []
            msgs.append({"role": r.role, "content": r.content, "sources": srcs or [], "timestamp": r.created_at.isoformat() if r.created_at else ""})
        self._cache_msgs(sid, msgs)
        return msgs

    def rename_session(self, sid: str, new_name: str) -> bool:
        with self._get_engine().connect() as conn:
            r = conn.execute(text("UPDATE sessions SET name=:n, updated_at=NOW() WHERE id=:id"), {"n": new_name, "id": sid})
            conn.commit()
            return r.rowcount > 0
