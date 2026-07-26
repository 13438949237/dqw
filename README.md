# Enterprise RAG Q&A System

基于 LangChain 的企业级检索增强生成（RAG）问答系统，支持多数据源接入、多路召回、重排序、缓存加速与 API / Web 双前端。

## 项目结构

```
src/
  connectors/    # 各类数据源适配器（PDF、DB、API 等）
  parsers/       # 文档解析器
  embeddings/    # 嵌入模型封装
  retrievers/    # 检索策略（多路召回）
  rerankers/     # 重排序模型
  llms/          # 大语言模型封装
  cache/         # 缓存实现
  evaluation/    # 评估指标
  utils/         # 通用工具
app/
  streamlit/     # Streamlit 前端页面
  api/           # FastAPI 路由
tests/           # 测试用例
docs/            # 文档
```

## 快速启动

### 1. 环境准备

创建虚拟环境并安装依赖：

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动依赖服务

```bash
docker-compose up -d
```

这将启动 Qdrant（向量数据库）和 Redis（缓存）服务。

### 3. 配置

编辑 `config.yaml` 设置模型提供商、API Key 等参数。敏感信息建议通过环境变量注入：

```bash
export OPENAI_API_KEY=sk-...
export QDRANT_API_KEY=...
```

### 4. 启动 API 服务

```bash
python -m app.api.main
```

API 文档自动生成于 `http://localhost:8000/docs`。

### 5. 启动 Streamlit 前端

```bash
streamlit run app/streamlit/app.py
```

## 技术栈

| 组件       | 技术选型              |
| ---------- | --------------------- |
| 框架       | LangChain             |
| 向量数据库 | Qdrant                |
| API 服务   | FastAPI + Uvicorn     |
| 前端       | Streamlit             |
| 缓存       | Redis                 |
| 配置管理   | PyYAML + Pydantic     |

## License

MIT
