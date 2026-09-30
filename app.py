# app.py
import streamlit as st
import json
import asyncio
import traceback
from fastmcp import Client
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)  # ✅ 新增 SystemMessage


# ==========================================
# 非同步輔助函數 (MCP Client)
# ==========================================
async def _async_read_resource(uri: str):
    async with Client("mcp_server.py") as client:
        return await client.read_resource(uri)


async def _async_search_and_prompt(question: str):
    async with Client("mcp_server.py") as client:
        tool_result = await client.call_tool(
            "search_knowledge", {"query": question, "k": 3}
        )
        context = tool_result.content[0].text

        # ✅ 修正：只傳入 context，不傳 question
        prompt_result = await client.get_prompt("campus_qa", {"context": context})
        system_prompt_text = prompt_result.messages[0].content.text

        return context, system_prompt_text


# ==========================================
# 1. 頁面設置與初始化
# ==========================================
st.set_page_config(
    page_title="星光大學校園助手 (Agentic MCP)", page_icon="🎓", layout="wide"
)


@st.cache_resource
def get_llm():
    return ChatOllama(model="qwen2.5:7b", temperature=0.1)


# ==========================================
# 2. 核心機制：Token 預算控管
# ==========================================
def get_trimmed_history(history, max_chars=3000):
    trimmed = []
    current_chars = 0
    for msg in reversed(history):
        msg_len = len(msg["content"])
        if current_chars + msg_len > max_chars:
            break
        trimmed.append(msg)
        current_chars += msg_len
    return list(reversed(trimmed))


# ==========================================
# 3. 側邊欄：系統狀態
# ==========================================
st.sidebar.title("📊 系統狀態")
try:
    llm = get_llm()
    st.sidebar.success("本地 LLM (qwen2.5:7b): 就緒")
    kb_info_response = asyncio.run(_async_read_resource("campus://kb-info"))
    kb_info = json.loads(kb_info_response[0].text)
    st.sidebar.success("MCP Server: 已連線")
    st.sidebar.info(f"知識庫狀態: {kb_info['status']}")
    st.sidebar.info(f"向量化文本塊數: {kb_info['document_count']}")
except Exception as e:
    st.sidebar.error(f"系統連線失敗: {str(e)}")
    st.stop()


# ==========================================
# 4. 核心邏輯：Semantic Router
# ==========================================
def get_intent(question: str, history_text: str) -> str:
    router_prompt = f"""你是一個校園 AI 助手的意圖路由器。
請根據「對話歷史」與「當前問題」，判斷使用者的意圖屬於哪一類。
**只**回覆以下四個單字之一：
- `campus_info`：需要查詢校園專屬規定、設施、圖書館、宿舍等資訊。
- `chitchat`：日常問候、打招呼、結尾語。
- `general`：一般世界常識或與校園無關的問題。
- `refuse`：危險、違規或不適合回答的問題。

對話歷史：
{history_text if history_text else "無"}

當前問題：{question}
你的判斷："""

    response = llm.invoke([HumanMessage(content=router_prompt)])
    intent = response.content.strip().lower()
    valid_intents = ["campus_info", "chitchat", "general", "refuse"]
    return intent if intent in valid_intents else "general"


# ==========================================
# 5. 主介面
# ==========================================
st.title("🎓 星光大學校園知識問答助手")
st.markdown("""
採用 **Agentic Workflow** 與 **標準 Chat Message 結構**，確保多輪對話的連貫性與 Token 穩定性。
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入您的校園問題，例如：圖書館週末有開嗎？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            raw_recent_history = st.session_state.messages[-11:-1]
            safe_history = get_trimmed_history(raw_recent_history, max_chars=3000)

            history_messages = []
            history_text = ""
            for msg in safe_history:
                role = "User" if msg["role"] == "user" else "AI"
                history_text += f"{role}: {msg['content']}\n"
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))

            with st.spinner("🧠 正在分析問題意圖..."):
                intent = get_intent(prompt, history_text)

            history_rounds = len(safe_history) // 2  # 1輪 = 1問 + 1答 = 2則
            total_rounds = len(st.session_state.messages) // 2

            st.sidebar.caption(f"🎯 當前意圖: `{intent}`")
            st.sidebar.caption(f"📜 本次參考歷史: {history_rounds} 輪")
            st.sidebar.caption(f"💬 累積對話: {total_rounds} 輪")

            final_answer = ""
            context = ""

            # ✅ 終極修正：標準化 Message 結構 [System] + [History...] + [Current User]
            current_user_msg = HumanMessage(content=prompt)  # 純粹的使用者提問

            if intent == "campus_info":
                with st.spinner("🔍 正在檢索知識庫..."):
                    context, system_prompt_text = asyncio.run(
                        _async_search_and_prompt(prompt)
                    )
                    system_msg = SystemMessage(content=system_prompt_text)

                    # 組合：[系統規範+Context] + [歷史對話] + [當前純粹提問]
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )
                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content

            elif intent == "chitchat":
                with st.spinner("💬 正在思考..."):
                    system_msg = SystemMessage(
                        content="你是星光大學的友善校園助手。請簡短、友善地回覆使用者的日常問候，不要使用信件格式。"
                    )
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )
                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content

            elif intent == "general":
                with st.spinner("🌍 正在搜尋常識..."):
                    system_msg = SystemMessage(
                        content="請憑你的常識回答使用者的問題，若不知道請直說，保持語氣自然。"
                    )
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )
                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content

            elif intent == "refuse":
                final_answer = "抱歉，我是校園知識助手，無法回答這個問題。"

            st.markdown(final_answer)

            if intent == "campus_info" and context:
                with st.expander("🔍 查看檢索到的原始上下文"):
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
