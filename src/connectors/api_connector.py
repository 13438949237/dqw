# -*- coding: utf-8 -*-
"""
API 连接器 —— 通过 HTTP 请求从外部接口拉取文档数据。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class APIConnector(BaseConnector):
    """从 RESTful / JSON API 拉取文档。

    Usage::

        connector = APIConnector(
            url="https://api.example.com/articles",
            headers={"Authorization": "Bearer xxx"},
            text_field="content",
            id_field="id",
        )
        docs = connector.load_documents()
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        text_field: str = "content",
        title_field: Optional[str] = None,
        id_field: str = "id",
        method: str = "GET",
        json_payload: Optional[Dict[str, Any]] = None,
        results_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            url:           请求地址。
            headers:       HTTP 请求头。
            params:        URL 查询参数。
            text_field:    JSON 响应体中作为文档内容的字段。
            title_field:   可选标题字段。
            id_field:      数据唯一标识字段。
            method:        HTTP 方法（GET / POST）。
            json_payload:  POST 请求体（仅 method=POST）。
            results_path:  可选 JSONPath 风格路径（如 "data.items"），
                           用于从嵌套结构中提取数组。
        """
        self._url = url
        self._headers = headers or {}
        self._params = params or {}
        self._text_field = text_field
        self._title_field = title_field
        self._id_field = id_field
        self._method = method.upper()
        self._json_payload = json_payload
        self._results_path = results_path
        super().__init__(source_label=url)

    def load_documents(self) -> List[Document]:
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "APIConnector 需要安装 requests，请执行: pip install requests"
            ) from exc

        logger.info("请求 %s %s", self._method, self._url)

        try:
            if self._method == "POST":
                resp = requests.post(
                    self._url,
                    headers=self._headers,
                    params=self._params,
                    json=self._json_payload,
                    timeout=30,
                )
            else:
                resp = requests.get(
                    self._url,
                    headers=self._headers,
                    params=self._params,
                    timeout=30,
                )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("API 请求失败: %s", exc)
            raise RuntimeError(f"API 请求失败: {exc}") from exc

        data = resp.json()
        items = self._extract_items(data)

        documents: List[Document] = []
        for item in items:
            page_content = item.get(self._text_field, "")
            if not page_content:
                continue

            meta = self._base_metadata()
            meta["source"] = f"{self._url}#{item.get(self._id_field, '')}"
            meta["api_url"] = self._url
            meta["item_id"] = str(item.get(self._id_field, ""))
            meta["ingestion_timestamp"] = datetime.now(timezone.utc).isoformat()

            if self._title_field:
                title = item.get(self._title_field, "")
                if title:
                    page_content = f"{title}\n\n{page_content}"

            documents.append(Document(page_content=page_content, metadata=meta))

        logger.info("APIConnector: 从 %s 加载了 %d 个文档", self._url, len(documents))
        return documents

    def _extract_items(self, data: Any) -> List[dict]:
        """从响应体中提取文档数组。"""
        if self._results_path is None:
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "results" in data:
                return data["results"]
            return [data] if isinstance(data, dict) else []
        # 简单点分号路径解析
        parts = self._results_path.split(".")
        result = data
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part, [])
            elif isinstance(result, list):
                try:
                    result = result[int(part)]
                except (ValueError, IndexError):
                    return []
            else:
                return []
        return result if isinstance(result, list) else [result]
