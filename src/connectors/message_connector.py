# -*- coding: utf-8 -*-
"""
消息流连接器 —— 模拟轮询消息队列/流式数据源。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Generator, List, Optional

from langchain_core.documents import Document

from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class MessageConnector(BaseConnector):
    """轮询式消息源连接器，模拟从 Kafka / WebSocket / Slack 等流式管道拉取。

    支持两种使用方式：

    - load_documents():      批量拉取（返回当前积压的所有消息）。
    - poll(interval, count): 生成器模式，逐条拉取。

    Usage::

        connector = MessageConnector("slack-general", simulate=True)
        for doc in connector.poll(interval=0.5, count=5):
            print(doc.page_content)
    """

    # 模拟消息模板
    _DEMO_MESSAGES = [
        "系统通知：数据库备份任务已于 03:00 顺利完成。",
        "用户反馈：搜索结果页面加载时间超过 5 秒，需要优化。",
        "告警：API 网关在 10:15 出现短暂 503 错误，已自动恢复。",
        "需求变更：首页布局需要调整为两栏式设计。",
        "运维日志：Redis 集群节点 node-3 于 02:00 完成主从切换。",
        "Code Review: PR #342 存在 SQL 注入风险，已标记待修复。",
        "会议纪要：决定下周起使用 Qdrant 替换 Milvus 作为向量存储。",
        "监控：P99 延迟较昨日下降 12%，优化效果显著。",
        "发布通知：v2.3.1 已通过灰度验证，预计今晚全量发布。",
        "安全：发现 CVE-2024-1234 影响当前依赖版本，CVE-2024-1234 需紧急升级。",
    ]

    def __init__(
        self,
        source_name: str,
        simulate: bool = True,
        message_count: int = 10,
    ) -> None:
        """
        Args:
            source_name:    消息源名称（如频道名、Topic 名）。
            simulate:       为 True 时使用内置假数据，无需外部依赖。
            message_count:  模拟消息总量。
        """
        self._source_name = source_name
        self._simulate = simulate
        self._message_count = message_count
        self._cursor = 0
        super().__init__(source_label=source_name)

    def load_documents(self) -> List[Document]:
        """一次性拉取当前可用的所有消息。"""
        messages: List[dict] = []
        if self._simulate:
            messages = self._generate_messages()
        else:
            logger.warning("MessageConnector: 非模拟模式尚未实现真实消息源接入")
        documents = self._to_documents(messages)
        logger.info("MessageConnector: 批量加载 %d 条消息", len(documents))
        return documents

    def poll(
        self,
        interval: float = 1.0,
        count: Optional[int] = None,
    ) -> Generator[Document, None, None]:
        """轮询模式：每隔 interval 秒产生一条消息。

        Args:
            interval: 轮询间隔（秒）。
            count:    最大消息数，None 表示不限。

        Yields:
            LangChain Document（每条消息一个）。
        """
        if not self._simulate:
            logger.warning("非模拟模式的 poll 尚未实现")
            return

        messages = self._generate_messages()
        for i, msg in enumerate(messages):
            if count is not None and i >= count:
                break
            docs = self._to_documents([msg])
            if docs:
                yield docs[0]
            time.sleep(interval)

    # ── 内部 ────────────────────────────────────────────────────────────

    def _generate_messages(self) -> List[dict]:
        templates = self._DEMO_MESSAGES * (
            (self._message_count // len(self._DEMO_MESSAGES)) + 1
        )
        return [
            {
                "message_id": f"{self._source_name}-{i:04d}",
                "content": templates[i],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "channel": self._source_name,
            }
            for i in range(self._message_count)
        ]

    def _to_documents(self, messages: List[dict]) -> List[Document]:
        docs: List[Document] = []
        for msg in messages:
            meta = self._base_metadata()
            meta["source"] = f"{self._source_name}#{msg.get('message_id', '')}"
            meta["message_id"] = msg.get("message_id", "")
            meta["channel"] = msg.get("channel", self._source_name)
            meta["timestamp"] = msg.get("timestamp", "")
            meta["connector_type"] = "message"

            docs.append(
                Document(page_content=msg["content"], metadata=meta)
            )
        return docs
