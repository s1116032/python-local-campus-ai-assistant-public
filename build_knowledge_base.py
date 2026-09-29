import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def build_db():
    db_path = "./data/chroma_db"

    # 如果資料庫已存在，先清除以確保重新生成
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print("已清除舊的 ChromaDB 資料...")

    # 1. 載入 Markdown 文檔
    loader = TextLoader("./knowledge_base/campus_faq.md", encoding="utf-8")
    docs = loader.load()

    # 2. 文本分割
    # chunk_size 設小一點，適合 FAQ 這種短問答的檢索
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # 3. 初始化本地嵌入模型
    print("正在載入本地嵌入模型 (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. 寫入 ChromaDB
    print("正在建立向量並存入 ChromaDB...")
    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="campus_knowledge",
    )

    print(f"✅ 知識庫建構完成！共切分為 {len(chunks)} 個文本塊。")
    print(f"📂 資料庫已保存至: {db_path}")


if __name__ == "__main__":
    build_db()
