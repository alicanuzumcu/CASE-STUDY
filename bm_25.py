from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import json

class BM_25_Retriever:
    def __init__(self, json_path="products.json"):
        self.json_path = json_path
        docs = self.load_documents(json_path)
        self.bm25_retriever = BM25Retriever.from_documents(docs, k=len(docs))

    def load_documents(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = []
        for p in data:
            text_parts = []
            for key, value in p.items():
                if value is None:
                    continue
                elif isinstance(value, list):
                    text_parts.append(" ".join(str(item).lower() for item in value))
                else:
                    text_parts.append(str(value).lower())
            page_content = " ".join(text_parts)
            docs.append(Document(page_content=page_content, metadata=p))
        return docs

    def get_filtered_results(self, query, max_results=10):
        query = query.strip().lower()
        initial_results = self.bm25_retriever.invoke(query)
        query_words = query.split()

        # OR mantığı
        filtered_results = [
            r for r in initial_results
            if any(word in r.page_content for word in query_words)
        ]

        return filtered_results[:max_results]
"""      
if __name__ == "__main__":
    print("🔍 BM25 Ürün Arama Sistemi Başlatılıyor...")
    try:
        retriever = BM_25_Retriever("products.json")
        print("✅ Hazır! Sorgu girin (çıkmak için 'q' yazın):\n")
    except FileNotFoundError:
        print("❌ products.json dosyası bulunamadı!")
        exit(1)

    while True:
        query = input("🔍 Sorgu: ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            print("🚪 Çıkılıyor...")
            break

        if not query:
            print("⚠️  Boş sorgu girdiniz. Lütfen bir şey yazın.\n")
            continue

        results = retriever.get_filtered_results(query, max_results=10)

        print(f"\n📦 Eşleşen {len(results)} sonuç:\n")
        if results:
            for i, doc in enumerate(results, 1):
                meta = doc.metadata
                print(f"{i}.")
                print(f"   id: {meta.get('id', 'N/A')}")
                print(f"   name: {meta.get('name', 'N/A')}")
                print(f"   description: {meta.get('description', 'N/A')}")
                print(f"   price: {meta.get('price', 'N/A')}")
                print(f"   brand: {meta.get('brand', 'N/A')}")
                print(f"   categories: {meta.get('categories', [])}")
                print(f"   tags: {meta.get('tags', [])}")
                print()
        else:
            print("❌ Hiç sonuç bulunamadı.\n")

        print("-----------------------------------------------------\n")
"""