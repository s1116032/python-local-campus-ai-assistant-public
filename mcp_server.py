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
def campus_qa(question: str, context: str) -> str:
    """
    校園知識問答的提示詞模板 (供 Host 端組合後傳送給 LLM)
    """
    return f"""你是一位友善且專業的星光大學校園助手。
請嚴格根據提供的「上下文資訊」來回答使用者的問題。
如果上下文中沒有相關資訊，請直接回答「抱歉，根據目前的校園知識庫，我無法回答這個問題。」，絕對不要編造答案。

上下文資訊：
{context}

使用者問題：
{question}

請以繁體中文簡潔、有禮貌地回答："""


if __name__ == "__main__":
    # 啟動 MCP Server (預設使用 stdio 標準輸入輸出傳輸)
    mcp.run()
