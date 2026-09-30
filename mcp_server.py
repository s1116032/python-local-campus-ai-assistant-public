# mcp_server.py
from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import json
import os

# 初始化 MCP Server，名稱為 "Local Campus AI"
mcp = FastMCP("Local Campus AI")

DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "campus_knowledge"


def get_vectorstore():
    """
    輔助函數：連線到本地 ChromaDB
    """
    return Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )


# ==========================================
# 1. MCP RESOURCE (資源)
# ==========================================
@mcp.resource("campus://kb-info")
def get_kb_info() -> str:
    """
    回傳本地知識庫的狀態與文件數量 (供 Host 端讀取)
    """
    try:
        vs = get_vectorstore()
        # 使用底線 _ 呼叫內部方法來取得數量，避免不必要的變數警告
        count = vs._collection.count()
        return json.dumps(
            {"status": "ready", "document_count": count, "location": DB_PATH},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


# ==========================================
# 2. MCP TOOL (工具)
# ==========================================
@mcp.tool()
def search_knowledge(query: str, k: int = 3) -> str:
    """
    在本地校園知識庫中搜索相關資訊 (供 Host 端呼叫)
    """
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=k)

        if not docs:
            return "未找到相關資訊。"

        # 將檢索到的文件片段格式化為字串
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知來源")
            results.append(
                f"[片段 {i + 1}] (來源: {os.path.basename(source)})\n{doc.page_content}"
            )

        return "\n\n".join(results)
    except Exception as e:
        return f"搜尋時發生錯誤: {str(e)}"


# ==========================================
# 3. MCP PROMPT (提示詞)
# ==========================================
@mcp.prompt()
def campus_qa(context: str) -> str:
    """
    校園知識問答的 System Prompt 模板
    """
    return f"""你是星光大學的專業校園助手。
請嚴格根據提供的「上下文資訊」來回答使用者的問題。

【回答規範】
1. 請全程使用**繁體中文**進行回答。
2. 語氣必須**自然、禮貌、友善**，像人類對話一樣，**絕對不要**使用「感謝來信」、「敬上」等制式信件格式。
3. 答案要簡潔明瞭，直接切中要害。
4. 如果上下文中**沒有**相關資訊，請禮貌地回答「抱歉，目前的校園知識庫中沒有相關紀錄。」，絕對不要編造答案。

以下是檢索到的上下文資訊，請僅基於此資訊回答：
{context}
"""


if __name__ == "__main__":
    # 啟動 MCP Server (預設使用 stdio 標準輸入輸出傳輸)
    mcp.run()
