# 🎓 星光大學校園 AI 助手 (Agentic MCP)

> **⚠️ 專案狀態聲明 (To 面試官)**
> 本專案為**應面試官要求**之限時考核專案。
> - **當前狀態**：Agentic MCP 核心架構 (Router, RAG, Query Rewriting) 已完成並可穩定展示。
> - **預計完工日**：本周日 (2026/10/04) 完成最終 Debug 與完整交付。
> - **關於 `booking_request` (預約 Agent Tool)**：此功能剛完成初步開發，目前正進行邊界測試與 Debug。為確保當前 Demo 的流暢度與系統穩定性，暫先將其屏蔽並列為 **Future Work**，將於本周日完工日修復並推送。

---

## 💡 專案簡介
基於 **Model Context Protocol (MCP)** 與 **Agentic Workflow** 構建的本地化校園 AI 助手。系統具備意圖路由、查詢重寫與 Token 預算控管機制，展現 Agent 系統架構設計與工具呼叫 (Tool Calling) 的整合能力。

## 🚀 核心技術亮點
- **Agentic Router**：LLM 自動分類意圖並分發對應的 Agent 處理邏輯。
- **Query Rewriting**：解決多輪對話中的指代消解問題，提升向量檢索精準度。
- **Standardized MCP**：嚴格分離 Resources, Tools, Prompts，遵循標準化協議設計 Agent 後端服務。
- **Token Budgeting**：實作動態歷史對話裁剪，確保不超出 LLM Context Window 限制。

## 🛠️ 快速啟動 (Quick Start)

```bash
# 1. 準備 Ollama 本地模型
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 2. 安裝依賴與建構向量知識庫
pip install -r requirements.txt
python build_knowledge_base.py

# 3. 啟動 Streamlit UI (MCP Server 會於背景自動啟動，使用 headless 模式避免彈出瀏覽器)
streamlit run app.py --server.headless true
```

## 📂 專案結構
```text
├── app.py                  # Streamlit UI 與 Agent 核心邏輯
├── mcp_server.py           # MCP Server (Resources, Tools, Prompts)
├── build_knowledge_base.py # ChromaDB 向量化腳本
├── requirements.txt        # Python 依賴套件清單
├── .gitignore              # Git 忽略檔案設定
├── data/                   # (執行後產生) ChromaDB 向量資料庫
└── knowledge_base/         # 校園 FAQ 原始數據
```

---

## License
Copyright © 2026 hanwu910514.

詳情請參閱[Apache License 2.0](LICENSE)檔案