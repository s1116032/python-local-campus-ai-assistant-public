from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import json
import os
import logging
import chromadb

logging.basicConfig(
    filename="mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

mcp = FastMCP("Local Campus AI")

CHROMA_DB_PATH = os.path.abspath("./data/chroma_db")
COLLECTION_NAME = "campus_knowledge"

_vectorstore = None


def get_vectorstore():
    """獲取 ChromaDB 實例。使用單例模式確保生命週期內僅初始化一次。"""
    global _vectorstore
    if _vectorstore is None:
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            _vectorstore = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
            )
        except Exception as e:
            logging.error(f"ChromaDB 初始化失敗: {str(e)}", exc_info=True)
            raise e
    return _vectorstore


@mcp.resource("campus://kb-info")
def get_kb_info() -> str:
    """回傳本地知識庫的狀態與文件數量 (Resource)。"""
    try:
        vs = get_vectorstore()
        all_docs = vs.get()
        count = len(all_docs["ids"]) if all_docs and "ids" in all_docs else 0

        return json.dumps(
            {"status": "ready", "document_count": count, "location": CHROMA_DB_PATH},
            ensure_ascii=False,
        )
    except Exception as e:
        logging.error(f"get_kb_info 錯誤: {str(e)}", exc_info=True)
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)


@mcp.tool()
def search_knowledge(query: str, k: int = 3) -> str:
    """在本地校園知識庫中搜索相關資訊 (Tool)。"""
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=k)

        if not docs:
            return "未找到相關資訊。"

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "未知來源")
            results.append(
                f"[片段 {i + 1}] (來源: {os.path.basename(source)})\n{doc.page_content}"
            )

        return "\n\n".join(results)
    except Exception as e:
        logging.error(f"search_knowledge 錯誤: {str(e)}", exc_info=True)
        return f"搜尋時發生錯誤: {str(e)}"


@mcp.prompt()
def campus_qa(context: str) -> str:
    """校園知識問答的 System Prompt 模板 (Prompt)。"""
    return f"""你是星光大學的專業校園助手。
請嚴格根據提供的「上下文資訊」來回答使用者的問題。

【回答規範】
1. 請全程使用繁體中文進行回答。
2. 語氣必須自然、禮貌、友善，像人類對話一樣，絕對不要使用「感謝來信」、「敬上」等制式信件格式。
3. 答案要簡潔明瞭，直接切中要害。
4. 如果上下文中沒有相關資訊，請禮貌地回答「抱歉，目前的校園知識庫中沒有相關紀錄。」，絕對不要編造答案。

以下是檢索到的上下文資訊，請僅基於此資訊回答：
{context}
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
