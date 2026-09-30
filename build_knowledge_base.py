import os
import shutil
import chromadb
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def build_db():
    db_path = os.path.abspath("./data/chroma_db")

    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print("已清除舊的 ChromaDB 資料...")

    loader = TextLoader("./knowledge_base/campus_faq.md", encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    print("正在載入本地嵌入模型 (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("正在建立向量並存入 ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="campus_knowledge",
    )

    doc_count = vectorstore._collection.count()

    print(f"✅ 知識庫建構完成！共切分為 {len(chunks)} 個文本塊。")
    print(f"🔍 驗證結果：ChromaDB 中目前共有 {doc_count} 筆向量資料。")
    print(f"📂 資料庫已保存至: {db_path}")


if __name__ == "__main__":
    build_db()
