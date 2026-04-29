from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pdfplumber
import os


class FAQResponse(BaseModel):
    answer: str = Field(description="Kullanıcının sorusuna verilen cevap")
    source_pages: List[int] = Field(default_factory=list, description="Cevabın bulunduğu sayfa numaraları")
    is_faq_question: bool = Field(default=True, description="Sorunun FAQ ile ilgili olup olmadığı")


class FAQAgent:
    def __init__(self, faq_pdf_path: str = "data/FAQ.pdf"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct"
        )
        self.ensemble_retriever = None
        self.vectorstore_path = "vectorstore/faiss_faq"
        os.makedirs(self.vectorstore_path, exist_ok=True)

        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.1)

        self.system_prompt = """Sen Gevheri Store mağazasının müşteri hizmetleri asistanısın.
Sana verilen FAQ (Sıkça Sorulan Sorular) belgesi içeriğine dayanarak müşterilerin sorularını yanıtla.

Kurallar:
- YALNIZCA verilen bağlam (context) içindeki bilgilere dayanarak cevap ver.
- Bağlamda olmayan bilgileri kesinlikle uydurma.
- Eğer sorunun cevabı bağlamda yoksa bunu açıkça söyle.
- Cevaplarını Türkçe ver, açık ve anlaşılır bir dil kullan.
- Müşteriye karşı her zaman nazik ve yardımsever bir ton koru.
- Cevap verirken mümkün olduğunca kapsamlı ve detaylı ol.

Bağlam:
{context}
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{query}")
        ])

        self.chain = self.prompt | self.llm

        self.load_pdf(faq_pdf_path)

    def load_pdf(self, pdf_path: str):
        """PDF'yi sayfa sayfa okuyup her sayfayı bir chunk olarak yükler."""
        try:
            print(f"📄 FAQ PDF yükleniyor: {pdf_path}")
            docs = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"   📃 Toplam sayfa sayısı: {total_pages}")

                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()

                    if not text or not text.strip():
                        print(f"   ⚠️ Sayfa {page_num} boş veya okunamadı, atlanıyor.")
                        continue

                    doc = Document(
                        page_content=text.strip(),
                        metadata={
                            "page": page_num,
                            "source": pdf_path,
                            "total_pages": total_pages,
                        }
                    )
                    docs.append(doc)

            if not docs:
                print("❌ PDF'den hiç içerik çıkarılamadı!")
                return

            print(f"   ✅ {len(docs)} sayfa başarıyla okundu.")

            # FAISS vectorstore oluştur veya yükle
            if os.path.exists(os.path.join(self.vectorstore_path, "index.faiss")):
                print("📁 Kayıtlı FAISS vectorstore bulundu. Yükleniyor...")
                vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            else:
                print("📁 Yeni FAISS vectorstore oluşturuluyor ve kaydediliyor...")
                vectorstore = FAISS.from_documents(docs, self.embedding_model)
                vectorstore.save_local(self.vectorstore_path)

            # BM25 retriever (keyword/lexical search)
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 5

            # FAISS retriever (semantic/vector search)
            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # Hybrid ensemble: BM25 + FAISS eşit ağırlıkla
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )

            print(f"✅ FAQ Hybrid Retriever hazır. ({len(docs)} sayfa indexlendi)")

        except FileNotFoundError:
            print(f"❌ {pdf_path} dosyası bulunamadı!")
        except Exception as e:
            print(f"❌ PDF yükleme hatası: {e}")

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Hybrid search ile ilgili FAQ sayfalarını döndürür.
        main.py ile uyumlu: List[Dict] döner.
        """
        if not self.ensemble_retriever:
            print("   ❌ FAQ retriever hazır değil!")
            return []

        print(f"\n   📊 FAQ Hybrid Search (BM25 + FAISS): '{query}'")
        candidate_docs = self.ensemble_retriever.invoke(query)
        print(f"   📦 Aday sayısı: {len(candidate_docs)}")

        results = []
        seen_pages = set()

        for doc in candidate_docs:
            page_num = doc.metadata.get("page")
            if page_num in seen_pages:
                continue
            seen_pages.add(page_num)

            results.append({
                "type": "faq",
                "page": page_num,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "FAQ.pdf"),
            })

            if len(results) >= limit:
                break

        print(f"   ✅ FAQ sonuç sayısı: {len(results)}")
        return results

    def generate_answer(
        self,
        query: str,
        user_info: str = "",
        chat_history: str = "",
    ) -> str:
        """
        Soruyu hybrid search ile FAQ'dan ilgili sayfalara yönlendirir
        ve LLM ile doğal dil cevabı üretir.
        final_agent'ın aksine bu agent kendi LLM cevabını üretir
        çünkü FAQ cevapları ürün seçimi değil bilgi sunumudur.
        """
        if not self.ensemble_retriever:
            return "FAQ sistemi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."

        retrieved_docs = self.ensemble_retriever.invoke(query)

        if not retrieved_docs:
            return "Bu konuda FAQ belgemizde bilgi bulunamadı. Müşteri hizmetlerimizle iletişime geçebilirsiniz."

        context = "\n\n---\n\n".join([
            f"[Sayfa {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        try:
            response = self.chain.invoke({
                "context": context,
                "query": query
            })
            # ChatOllama .content attribute döner
            answer = response.content if hasattr(response, "content") else str(response)
            return answer
        except Exception as e:
            print(f"⚠️ FAQ LLM hatası: {e}")
            return "Cevap oluşturulurken bir hata oluştu."


"""
# ========================
# Test Döngüsü
# ========================
if __name__ == "__main__":
    agent = FAQAgent(faq_pdf_path="data/FAQ.pdf")

    print("\n❓ FAQ Arama Sistemi Başlatıldı!")
    print("Çıkmak için 'çık' veya 'exit' yazın.\n")

    while True:
        try:
            user_query = input("🔍 FAQ sorusu girin: ").strip()
            if user_query.lower() in ["çık", "exit", "quit", "q"]:
                print("👋 Görüşmek üzere!")
                break
            if not user_query:
                print("⚠️ Lütfen bir soru girin.\n")
                continue

            print("\n--- Hybrid Search Sonuçları ---")
            results = agent.search(user_query, limit=5)
            for r in results:
                print(f"📄 Sayfa {r['page']}: {r['content'][:200]}...\n")

            print("\n--- LLM Cevabı ---")
            answer = agent.generate_answer(user_query)
            print(f"\n💬 {answer}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Çıkılıyor...")
            break
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}\n")
            
"""