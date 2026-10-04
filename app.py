import asyncio
import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from utils import (
    call_tool,
    check_llm_connection,
    extract_booking_details,
    get_intent,
    get_llm,
    get_trimmed_history,
    read_resource,
    rewrite_query,
    search_and_prompt,
)

load_dotenv()

# ==========================================
# 日誌設定
# ==========================================
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. 頁面設置與 Session State 初始化
# ==========================================
st.set_page_config(
    page_title="星光大學校園助手 (Agentic MCP)", page_icon="✨", layout="wide"
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_booking" not in st.session_state:
    st.session_state.pending_booking = None
if "last_booking_id" not in st.session_state:
    st.session_state.last_booking_id = None

# ==========================================
# 2. 側邊欄：系統狀態
# ==========================================
st.sidebar.title("📊 系統狀態")
try:
    model_name = os.getenv("GENERATIVE_MODEL", "qwen2.5:7b")

    if "llm_connection_checked" not in st.session_state:
        with st.spinner("正在檢查 LLM 連線..."):
            st.session_state.llm_connection_ok = check_llm_connection()
            st.session_state.llm_connection_checked = True

    if st.session_state.llm_connection_ok:
        st.sidebar.success(f"本地 LLM ({model_name}): 就緒")
        llm = get_llm()
    else:
        st.sidebar.error(f"本地 LLM ({model_name}): 連線失敗")
        st.sidebar.warning("請確認 Ollama 已啟動")
        st.stop()

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

except Exception:
    logger.exception("系統初始化連線失敗")
    error_id = int(time.time())
    st.sidebar.error(f"系統連線失敗，請稍後再試（錯誤編號: {error_id}）")
    st.stop()

# ==========================================
# 3. 主介面：歷史對話顯示
# ==========================================
st.title("✨ 星光大學校園知識問答助手")
st.markdown(
    "採用 **Agentic Workflow**，具備 **RAG 檢索**、**Query Rewriting** 與 **Human-in-the-Loop (雙重確認預約)** 機制。"
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. 狀態機 1：處理「待確認的預約草案」
# ==========================================
if st.session_state.pending_booking:
    draft = st.session_state.pending_booking
    st.warning("### 📋 預約確認 (Human-in-the-Loop)")
    st.markdown(f"""
    | 項目 | 內容 |
    |:---|:---|
    | **地點** | `{draft.get("room_id", "未知")}` |
    | **日期** | `{draft.get("date", "未知")}` |
    | **時間** | `{draft.get("start_time", "未知")} ~ {draft.get("end_time", "未知")}` |
    | **用途分類** | `{draft.get("purpose_category", "未知")}` |
    | **用途說明** | `{draft.get("purpose_detail", "未知")}` |
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 確認預約", use_container_width=True, type="primary"):
            with st.spinner("正在向 MCP Server 提交預約..."):
                tool_args = {
                    k: v for k, v in draft.items() if k not in ["rejected", "reason"]
                }
                result = asyncio.run(call_tool("book_study_room", tool_args))
                st.session_state.messages.append(
                    {"role": "assistant", "content": result}
                )
                if "預約碼: BK" in result:
                    st.session_state.last_booking_id = result.split("預約碼: ")[
                        1
                    ].split("\n")[0]
                st.session_state.pending_booking = None
                st.rerun()
    with col2:
        if st.button("❌ 取消，不預約了", use_container_width=True):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "好的，已為您取消預約。有其他需要隨時告訴我！",
                }
            )
            st.session_state.pending_booking = None
            st.rerun()
    st.stop()

# ==========================================
# 5. 狀態機 2：提供「取消剛完成的預約」選項
# ==========================================
if st.session_state.last_booking_id:
    st.info(f"📝 您剛剛預約了 `{st.session_state.last_booking_id}`。")
    if st.button(f"↩️ 發現訂錯了，幫我取消 {st.session_state.last_booking_id}"):
        with st.spinner("正在向 MCP Server 提交取消請求..."):
            cancel_result = asyncio.run(
                call_tool(
                    "cancel_booking", {"booking_id": st.session_state.last_booking_id}
                )
            )
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": f"幫我取消預約 {st.session_state.last_booking_id}",
                }
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": cancel_result}
            )
            st.session_state.last_booking_id = None
            st.rerun()

# ==========================================
# 6. 狀態機 3：接收新的使用者輸入與意圖分發
# ==========================================
if prompt := st.chat_input(
    "請輸入您的校園問題，例如：圖書館週末有開嗎？或幫我預約A101"
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            raw_recent_history = st.session_state.messages[-11:-1]
            safe_history = get_trimmed_history(raw_recent_history, max_chars=1500)

            history_messages, history_text = [], ""
            for msg in safe_history:
                role = "User" if msg["role"] == "user" else "AI"
                content = msg.get("content") or ""
                history_text += f"{role}: {content}\n"
                if msg["role"] == "user":
                    history_messages.append(HumanMessage(content=content))
                elif msg["role"] == "assistant":
                    history_messages.append(AIMessage(content=content))

            with st.spinner("🧠 正在分析問題意圖..."):
                intent = get_intent(prompt, history_text)
            st.sidebar.caption(f"🎯 當前意圖: `{intent}`")

            current_user_msg = HumanMessage(content=prompt)

            if intent == "booking_request":
                with st.spinner("📝 正在解析預約參數並審核用途..."):
                    has_rejection = any(
                        msg["role"] == "assistant"
                        and isinstance(msg.get("content"), str)
                        and ("預約被拒絕" in msg["content"] or "🚫" in msg["content"])
                        for msg in safe_history
                    )

                    if has_rejection:
                        clean_history_text = "無"
                        logger.info(
                            "偵測到歷史包含拒絕資訊，已清空歷史以預防上下文污染。"
                        )
                    else:
                        clean_history_text = history_text

                    draft = extract_booking_details(prompt, clean_history_text)

                    if draft and draft.get("missing_info"):
                        ask_msg = draft.get(
                            "prompt", "請提供您想預約的具體日期與時間。"
                        )
                        st.info(f"💡 {ask_msg}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"💡 {ask_msg}"}
                        )

                    elif draft and not draft.get("rejected", False):
                        st.session_state.pending_booking = draft
                        st.rerun()

                    elif draft and draft.get("rejected", False):
                        reject_reason = draft.get("reason", "不符合討論室使用規範。")
                        st.error(f"🚫 預約被拒絕：{reject_reason}")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": f"🚫 預約被拒絕：{reject_reason}",
                            }
                        )
                    else:
                        err_msg = "無法解析預約資訊，請提供更多細節（如：預約 A01，明天下午兩點）。"
                        st.error(err_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err_msg}
                        )

            elif intent == "campus_info":
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

            elif intent in ["chitchat", "general"]:
                sys_content = (
                    "你是星光大學的友善校園助手。請全程使用繁體中文，以禮貌、自然的語氣簡短回覆。"
                    if intent == "chitchat"
                    else "請全程使用繁體中文，以禮貌、自然的語氣憑常識回答。若不知道請直說。"
                )
                system_msg = SystemMessage(content=sys_content)
                response = llm.invoke(
                    [system_msg] + history_messages + [current_user_msg]
                )
                st.markdown(response.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content}
                )

            elif intent == "refuse":
                final_answer = "抱歉，我是校園知識助手，無法回答這個問題。"
                st.warning(final_answer)

                if (
                    st.session_state.messages
                    and st.session_state.messages[-1]["role"] == "user"
                ):
                    st.session_state.messages.pop()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "⚠️ 系統已攔截並拒絕該違規或惡意問題。請提問與校園相關的內容。",
                    }
                )

        except Exception:
            logger.exception("對話處理發生未預期錯誤")
            error_id = int(time.time())
            st.error(f"系統繁忙，請稍後再試（錯誤編號: {error_id}）")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"系統繁忙，請稍後再試（錯誤編號: {error_id}）",
                }
            )
