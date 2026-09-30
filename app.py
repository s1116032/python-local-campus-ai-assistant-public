# app.py
import streamlit as st
import json
import asyncio
import traceback
from fastmcp import Client
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ==========================================
# 非同步輔助函數 (MCP Client)
# ==========================================
async def _async_read_resource(uri: str):
    async with Client("mcp_server.py") as client:
        return await client.read_resource(uri)


async def _async_search_and_prompt(search_query: str):
    async with Client("mcp_server.py") as client:
        # 使用重寫後的 search_query 進行檢索
        tool_result = await client.call_tool(
            "search_knowledge", {"query": search_query, "k": 3}
        )
        context = tool_result.content[0].text

        prompt_result = await client.get_prompt("campus_qa", {"context": context})
        system_prompt_text = prompt_result.messages[0].content.text

        return context, system_prompt_text, search_query


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
# 4. 核心邏輯：Semantic Router & Query Rewriting
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


def rewrite_query(question: str, history_text: str) -> str:
    """
    查詢重寫：將依賴上下文的簡短問題，重寫為獨立的完整檢索 Query
    """
    rewrite_prompt = f"""你是一個專業的查詢重寫助手。
請根據「對話歷史」與「當前問題」，將當前問題重寫為一個獨立、完整、包含關鍵實體的查詢語句，以便在向量資料庫中進行精準檢索。
如果當前問題已經很完整且不依賴歷史，請保持原樣。
**只輸出重寫後的查詢語句，絕對不要包含任何解釋、前言或標點符號。**

對話歷史：
{history_text if history_text else "無"}

當前問題：{question}

重寫後的查詢語句："""

    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    return response.content.strip()


# ==========================================
# 5. 主介面
# ==========================================
st.title("🎓 星光大學校園知識問答助手")
st.markdown("""
採用 **Agentic Workflow**，內建 **Query Rewriting (查詢重寫)** 機制，完美解決多輪對話中的指代消解問題，確保 RAG 檢索精準度。
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

            history_rounds = len(safe_history) // 2
            total_rounds = len(st.session_state.messages) // 2
            st.sidebar.caption(f"🎯 當前意圖: `{intent}`")
            st.sidebar.caption(f"📜 本次參考歷史: {history_rounds} 輪")
            st.sidebar.caption(f"💬 累積對話: {total_rounds} 輪")

            final_answer = ""
            context = ""
            rewritten_query = ""

            current_user_msg = HumanMessage(content=prompt)

            if intent == "campus_info":
                # ✅ 新增：Query Rewriting
                with st.spinner("✍️ 正在重寫查詢語句 (解決指代消解)..."):
                    rewritten_query = rewrite_query(prompt, history_text)

                with st.spinner("🔍 正在使用重寫後的語句檢索知識庫..."):
                    # 傳入 rewritten_query 進行檢索
                    context, system_prompt_text, searched_query = asyncio.run(
                        _async_search_and_prompt(rewritten_query)
                    )
                    system_msg = SystemMessage(content=system_prompt_text)

                    # 注意：送給 LLM 生成最終回答的，依然是「原始的使用者提問」，以維持對話自然度
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )
                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content

            elif intent == "chitchat":
                with st.spinner("💬 正在思考..."):
                    system_msg = SystemMessage(
                        content="你是星光大學的友善校園助手。請全程使用繁體中文，以禮貌、自然的語氣簡短回覆使用者的日常問候，不要使用制式信件格式。"
                    )
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )
                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content

            elif intent == "general":
                with st.spinner("🌍 正在搜尋常識..."):
                    system_msg = SystemMessage(
                        content="請全程使用繁體中文，以禮貌、自然的語氣憑常識回答使用者的問題。若不知道請直說。"
                    )
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )
                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content

            elif intent == "refuse":
                final_answer = "抱歉，我是校園知識助手，無法回答這個問題。"

            st.markdown(final_answer)

            # ✅ 優化：在 Expander 中展示 Query Rewriting 的成果
            if intent == "campus_info" and context:
                with st.expander("🔍 檢索詳情與上下文 (RAG Details)"):
                    st.markdown(f"**原始提問:** `{prompt}`")
                    if rewritten_query and rewritten_query != prompt:
                        st.markdown(f"**重寫後的檢索 Query:** `{rewritten_query}` ✨")
                    else:
                        st.markdown(f"**檢索 Query (無需重寫):** `{prompt}`")
                    st.markdown("---")
                    st.text_area("檢索到的 Context", context, height=200, disabled=True)

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
