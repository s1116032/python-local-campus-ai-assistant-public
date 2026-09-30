import streamlit as st
import json
import asyncio
import traceback
from fastmcp import Client
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


async def read_resource(uri: str):
    """非同步讀取 MCP 資源"""
    async with Client("mcp_server.py") as client:
        return await client.read_resource(uri)


async def search_and_prompt(search_query: str):
    """非同步呼叫搜尋工具並獲取對應的 System Prompt"""
    async with Client("mcp_server.py") as client:
        tool_result = await client.call_tool(
            "search_knowledge", {"query": search_query, "k": 3}
        )
        context = tool_result.content[0].text
        prompt_result = await client.get_prompt("campus_qa", {"context": context})
        system_prompt_text = prompt_result.messages[0].content.text
        return context, system_prompt_text, search_query


st.set_page_config(
    page_title="星光大學校園助手 (Agentic MCP)", page_icon="🎓", layout="wide"
)


@st.cache_resource
def get_llm():
    return ChatOllama(model="qwen2.5:7b", temperature=0.1)


if "messages" not in st.session_state:
    st.session_state.messages = []


def get_trimmed_history(history, max_chars=3000):
    """依據字元長度裁剪歷史紀錄，避免超出 Token 限制"""
    trimmed, current_chars = [], 0
    for msg in reversed(history):
        msg_len = len(msg["content"])
        if current_chars + msg_len > max_chars:
            break
        trimmed.append(msg)
        current_chars += msg_len
    return list(reversed(trimmed))


st.sidebar.title("📊 系統狀態")
try:
    llm = get_llm()
    st.sidebar.success("本地 LLM (qwen2.5:7b): 就緒")

    @st.cache_resource
    def get_kb_info_cached():
        return asyncio.run(read_resource("campus://kb-info"))

    kb_info_response = get_kb_info_cached()
    raw_text = kb_info_response[0].text
    kb_info = json.loads(raw_text)

    st.sidebar.success("MCP Server: 已連線")
    st.sidebar.info(f"知識庫狀態: {kb_info.get('status', 'unknown')}")

    if "document_count" in kb_info:
        st.sidebar.info(f"向量化文本塊數: {kb_info['document_count']}")
    else:
        st.sidebar.error(f"知識庫讀取失敗: {kb_info.get('message', '未知錯誤')}")

except Exception as e:
    st.sidebar.error(f"系統連線失敗: {str(e)}")
    st.stop()


def get_intent(question: str, history_text: str) -> str:
    """判斷使用者問題的意圖分類"""
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
    """重寫查詢語句以解決多輪對話中的指代消解問題"""
    rewrite_prompt = f"""你是一個專業的查詢重寫助手。請根據「對話歷史」與「當前問題」，將當前問題重寫為一個獨立、完整、包含關鍵實體的查詢語句。如果當前問題已經很完整，請保持原樣。**只輸出重寫後的查詢語句。**
對話歷史：\n{history_text if history_text else "無"}\n\n當前問題：{question}\n\n重寫後的查詢語句："""
    return llm.invoke([HumanMessage(content=rewrite_prompt)]).content.strip()


st.title("🎓 星光大學校園知識問答助手")
st.markdown("""
採用 **Agentic Workflow**，具備 **RAG 檢索** 與 **Query Rewriting** 機制。
""")

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

            history_messages, history_text = [], ""
            for msg in safe_history:
                role = "User" if msg["role"] == "user" else "AI"
                history_text += f"{role}: {msg['content']}\n"
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=msg["content"]))

            with st.spinner("🧠 正在分析問題意圖..."):
                intent = get_intent(prompt, history_text)

            st.sidebar.caption(f"🎯 當前意圖: `{intent}`")
            current_user_msg = HumanMessage(content=prompt)

            if intent == "campus_info":
                rewritten_query = prompt
                with st.spinner("🔍 正在檢索知識庫..."):
                    if "承上" in prompt or "那個" in prompt or "那" in prompt:
                        with st.spinner("✍️ 正在重寫查詢語句..."):
                            rewritten_query = rewrite_query(prompt, history_text)

                    context, system_prompt_text, _ = asyncio.run(
                        search_and_prompt(rewritten_query)
                    )
                    system_msg = SystemMessage(content=system_prompt_text)
                    messages_to_llm = (
                        [system_msg] + history_messages + [current_user_msg]
                    )

                    response = llm.invoke(messages_to_llm)
                    final_answer = response.content
                    st.markdown(final_answer)

                    with st.expander("🔍 檢索詳情"):
                        st.markdown(f"**原始提問:** `{prompt}`")
                        if rewritten_query != prompt:
                            st.markdown(f"**重寫後 Query:** `{rewritten_query}` ✨")
                        st.text_area("Context", context, height=200, disabled=True)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )

            elif intent == "chitchat":
                system_msg = SystemMessage(
                    content="你是星光大學的友善校園助手。請全程使用繁體中文，以禮貌、自然的語氣簡短回覆。"
                )
                response = llm.invoke(
                    [system_msg] + history_messages + [current_user_msg]
                )
                st.markdown(response.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content}
                )

            elif intent == "general":
                system_msg = SystemMessage(
                    content="請全程使用繁體中文，以禮貌、自然的語氣憑常識回答。若不知道請直說。"
                )
                response = llm.invoke(
                    [system_msg] + history_messages + [current_user_msg]
                )
                st.markdown(response.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content}
                )

            elif intent == "refuse":
                final_answer = "抱歉，我是校園知識助手，無法回答這個問題。"
                st.markdown(final_answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": final_answer}
                )

        except Exception as e:
            st.error(f"發生錯誤: {str(e)}\n```\n{traceback.format_exc()}\n```")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"系統錯誤: {str(e)}"}
            )
