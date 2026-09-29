# 🎓 本地化校園知識問答助手 (MCP-RAG)

> 完全本地運行、零雲端依賴的 AI 問答系統。採用 **MCP (Model Context Protocol)** 架構 + **RAG 檢索增強生成**，所有推理與資料處理皆在本機完成。

---

## 🖼️ 實際對話展示

| 第一輪提問 | 第二輪追問（上下文繼承） |
|:---:|:---:|
| ![第一輪](./screenshots/q1.png) | ![第二輪](./screenshots/q2.png) |
| `圖書館隔天有開嗎？` | `承上，隔天有開嗎？` |

> AI 正確理解「承上」指的是前一輪的圖書館語境，並結合知識庫給出精準回答。

---

## 🏗️ 系統架構

```
┌──────────────────────────────────────────────────────────────┐
│                  Streamlit (MCP Host)                        │
│   對話歷史管理 │ 意圖調度 │ 本地 LLM (qwen2.5:7b)            │
└────────────────────────┬─────────────────────────────────────┘
                         │  stdio (MCP Protocol)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  MCP Server (FastMCP)                        │
│                                                              │
│  📦 Resource          🔧 Tool              📝 Prompt         │
│  知識庫狀態查詢        向量相似度檢索        問答提示詞模板     │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │   ChromaDB (本地)   │
              │   nomic-embed-text │
              └────────────────────┘
```

**核心設計：Host 與 Server 完全解耦。** 更換模型、新增工具、替換前端，彼此互不影響。

---

## ⚡ 快速啟動

```bash
# 拉取模型
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 安裝依賴 → 建構知識庫 → 啟動
pip install -r requirements.txt
python build_knowledge_base.py
streamlit run app.py
```

---

## 🔑 技術亮點

| # | 亮點 | 說明 |
|---|------|------|
| 1 | **完全本地化** | 模型推理、向量檢索、資料儲存全在本機，適合處理校園敏感資料 |
| 2 | **標準 MCP 協議** | 嚴格分離 Tools / Resources / Prompts 三大原語，符合 Anthropic MCP 規範 |
| 3 | **RAG 管道** | 文本切塊 → 向量化 → 相似度檢索 → 上下文注入，有效抑制 LLM 幻覺 |
| 4 | **多輪對話記憶** | 歷史訊息隨每次請求一併送入 LLM，支援「承上」「剛剛那個」等指代消解 |
| 5 | **高擴展性** | 換模型改一行、加工具加一個裝飾器、換前端零改動 |

---

## 🛠️ 技術棧

| 層級 | 技術 |
|------|------|
| 生成模型 | Ollama · `qwen2.5:7b` |
| 嵌入模型 | Ollama · `nomic-embed-text` |
| 向量資料庫 | ChromaDB |
| RAG / LLM 串接 | LangChain |
| MCP 實作 | FastMCP (Python) |
| 前端 | Streamlit |
| 傳輸 | stdio（本地進程通訊） |

---

## 📁 專案結構

```
├── app.py                    # MCP Host：對話管理 + LLM 調度
├── mcp_server.py             # MCP Server：Tools / Resources / Prompts
├── build_knowledge_base.py   # 向量化腳本
├── knowledge_base/           # 原始校園文件
│   └── campus_faq.md
├── data/chroma_db/           # 向量資料庫（不入版控）
├── screenshots/              # 對話截圖
└── requirements.txt
```

---

## 🔮 後續擴展

- Hybrid Search（BM25 + 向量）提升檢索精準度
- Metadata Filtering 依文件類型 / 日期過濾
- Docker 容器化部署
- FastAPI 化，提供 RESTful API 供校務系統串接

---

## License
Copyright © 2026 hanwu910514.

詳情請參閱[Apache License 2.0](LICENSE)檔案
