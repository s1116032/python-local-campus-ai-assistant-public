# app.py
import streamlit as st
import json
import asyncio
import traceback
from fastmcp import Client
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


# ==========================================
# 非同步輔助函數 (封裝 MCP Client 完整生命週期)
# ==========================================
async def _async_read_resource(uri: str):
    """連線 → 讀取 Resource → 斷線"""
    async with Client("mcp_server.py") as client:
        return await client.read_resource(uri)


async def _async_search_and_prompt(question: str):
    """連線 → 呼叫 Tool → 呼叫 Prompt → 斷線"""
    async with Client("mcp_server.py") as client:
        tool_result = await client.call_tool(
            "search_knowledge", {"query": question, "k": 3}
        )
        context = tool_result.content[0].text

        prompt_result = await client.get_prompt(
            "campus_qa", {"question": question, "context": context}
        )
        prompt_text = prompt_result.messages[0].content.text

        return context, prompt_text


# ==========================================
# 1. 頁面設置
# ==========================================
st.set_page_config(
    page_title="星光大學校園助手 (MCP 架構)", page_icon="🎓", layout="wide"
)


@st.cache_resource
def get_llm():
    """初始化本地 LLM (Qwen 2.5)"""
    return ChatOllama(model="qwen2.5:7b", temperature=0.1)


# ==========================================
# 2. 側邊欄：系統狀態與知識庫資訊
# ==========================================
st.sidebar.title("📊 系統狀態")

try:
    llm = get_llm()
    st.sidebar.success("本地 LLM (qwen2.5:7b): 就緒")

    kb_info_response = asyncio.run(_async_read_resource("campus://kb-info"))
    raw_text = kb_info_response[0].text
    kb_info = json.loads(raw_text)

    st.sidebar.success("MCP Server: 已連線")
    st.sidebar.info(f"知識庫狀態: {kb_info['status']}")
    st.sidebar.info(f"向量化文本塊數: {kb_info['document_count']}")

except Exception as e:
    st.sidebar.error(f"系統連線失敗: {str(e)}")
    st.error("無法連線到 MCP Server。")
    st.error(f"詳細錯誤:\n```\n{traceback.format_exc()}\n```")
    st.stop()

# ==========================================
# 3. 主介面：對話與問答邏輯
# ==========================================
st.title("🎓 星光大學校園知識問答助手")
st.markdown("""
這是一個**完全本地化**的 AI 助手，採用 **MCP (Model Context Protocol)** 架構。
*   **Server**: 提供知識庫搜尋 (Tool) 與提示詞 (Prompt)。
*   **Host (本頁面)**: 負責調度、管理對話歷史，並呼叫本地 LLM 產生回答。
""")

# 初始化對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史對話
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收使用者輸入
if prompt := st.chat_input("請輸入您的校園問題，例如：圖書館週末有開嗎？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("正在透過 MCP 檢索知識庫並讓 LLM 思考中..."):
            try:
                # ✅ 步驟 1: 透過 MCP 檢索知識庫 + 組裝 Prompt
                context, final_prompt = asyncio.run(_async_search_and_prompt(prompt))

                # ✅ 步驟 2: 組裝對話歷史 (只保留最近 10 輪，避免超出上下文窗口)
                recent_history = st.session_state.messages[
                    -11:-1
                ]  # 排除剛加入的當前問題

                history_messages = []
                for msg in recent_history:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        history_messages.append(AIMessage(content=msg["content"]))

                # ✅ 步驟 3: 組合「歷史對話 + 當前問題(含RAG上下文)」
                current_message = HumanMessage(content=final_prompt)
                all_messages = history_messages + [current_message]

                # ✅ 步驟 4: 送給本地 LLM 產生最終回答
                response = llm.invoke(all_messages)
                final_answer = response.content

                st.markdown(final_answer)

                # 顯示檢索到的原始資料
                with st.expander("🔍 查看檢索到的原始上下文 (MCP Tool 結果)"):
                    st.text_area("Context", context, height=200, disabled=True)

                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

            except Exception as e:
                error_msg = f"處理問題時發生錯誤: {str(e)}"
                st.error(error_msg)
                st.error(f"詳細錯誤:\n```\n{traceback.format_exc()}\n```")
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
