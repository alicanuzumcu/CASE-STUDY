from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import os


class AksesuarQuery(BaseModel):
    brand: Optional[str] = Field(default=None)
    renk: Optional[str] = Field(default=None)
    price_min: Optional[int | float] = Field(default=None)
    price_max: Optional[int | float] = Field(default=None)
    price_exact: Optional[int | float] = Field(default=None)
    materyal: Optional[str] = Field(default=None)
    categories: Optional[list[str]] = Field(default=None)
    yıl: Optional[str] = Field(default=None)
    boyut: Optional[str] = Field(default=None)
    kondisyon: Optional[str] = Field(default=None)
    orijinallik: Optional[str] = Field(default=None)


class AksesuarQueryParser:
    def __init__(self, brands_list: List[str]):  # groq_api_key kaldırıldı
        brands_str = ", ".join(brands_list)
        self.system_prompt = f"""Sen bir aksesuar mağazası uzmanısın. Kullanıcı sorgusunu analiz et ve filtreleme parametrelerini çıkar.

        Kurallar:
        -> Kullanıcının sorgusuna bakarak şunları eğer bilgi varsa doldur yoksa da null bırak:
            - brand: Eğer kullanıcı marka belirtmişse doldur
            - renk: Eğer müşteri renk belirtmişse doldur
            - price_min: Eğer kullanıcı minimum fiyat veya bütçe belirtmişse float değer ile doldur
            - price_max: Eğer kullanıcı maximum fiyat veya bütçe belirtmişse float değer ile doldur
            - price_exact: Eğer kullanıcı net bir fiyat belirtmişse float değer ile doldur
            - materyal: Eğer kullanıcı materyal belirtmişse doldur (deri, altın, gümüş, kadife vs.)
            - categories: Kullanıcı eğer aksesuar türleri belirtmişse liste olarak doldur. (Kategoriler: Aksesuar, Takı, Gözlük, Bileklik, Şal & Eşarp & Fular, Kemer, Diğer, Küpe, Kolye, Şapka & Bere, Broş, Yüzük, Saç Aksesuarları, Mücevher) - liste olarak döndür
            - yıl: Eğer kullanıcı üretim yılı belirtmişse doldur (örn: "2018", "2020")
            - boyut: Eğer kullanıcı boyut belirtmişse doldur (örn: "55 - 100 cm", "Tek Ebat")
            - kondisyon: Ürün durumu (örn: "Çok İyi", "İyi", "Yeni Gibi")
            - orijinallik: Orijinallik bilgisi (örn: "Orijinal İkinci El")

        -> Marka isimlerini doğru yazdığından emin ol. Markalar: {brands_str}
        -> MARKA DÜZELTMELERİ (ÇOK ÖNEMLİ):
           Kullanıcı yanlış yazsa bile doğru marka ismini bul:
           - "luiz viton", "viton"→ LOUIS VUITTON
           - "gucci", "guci", "gucı" → GUCCI
           - "prada", "pırada" → PRADA
           - "dior", "kristian dior" → CHRISTIAN DIOR
           - "channel", "chanel", "şanel" → CHANEL
           Benzer şekilde tüm markalar için en yakın eşleşmeyi kullan.
        -> renk boyut materyal gibi filtreleme parametrelerini özel isim değilse MUTLAKA TÜRKÇE olarak oluştur.
        """

        # Ollama ile llama3.1:latest modeli
        self.llm = ChatOllama(
            model="gpt-oss:120b-cloud",
            temperature=0.1,
        ).with_structured_output(AksesuarQuery)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{query}")
        ])

        self.chain = self.prompt | self.llm

    def parse_query(self, query: str) -> AksesuarQuery:
        try:
            parsed_query = self.chain.invoke({"query": query})
            return parsed_query
        except Exception as e:
            print(f"⚠️ Parser hatası: {e}")
            return AksesuarQuery()


class AksesuarAgent:
    def __init__(self, aksesuar_json_path: str = "data/aksesuar.json"):  # groq_api_key kaldırıldı
        self.brands_list = self.get_brands_from_json(aksesuar_json_path)
        self.query_parser = AksesuarQueryParser(self.brands_list)  # API key yok
        self.embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
        self.ensemble_retriever = None
        self.vectorstore_path = "vectorstore/faiss_aksesuar"
        os.makedirs(self.vectorstore_path, exist_ok=True)
        self.load_json(aksesuar_json_path)

    def get_brands_from_json(self, file_path: str) -> List[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        brand_counts = {}
        for item in data:
            brand = item.get("brand")
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        return [brand for brand, count in sorted_brands]

    def load_json(self, json_path: str):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                products = json.load(f)

            docs = []
            for p in products:
                text_parts = []
                for key, value in p.items():
                    if value is None:
                        continue
                    elif isinstance(value, list):
                        text_parts.append(" ".join(str(item).lower() for item in value))
                    else:
                        text_parts.append(str(value).lower())
                page_content = " ".join(text_parts)

                meta = {
                    'id': p.get('id'),
                    'name': p.get('name', ''),
                    'price': float(p.get('price', 0)),
                    'brand': p.get('brand', '').upper(),
                    'renk': str(p.get('renk', '')).lower(),
                    'materyal': str(p.get('materyal', '')).lower(),
                    'yıl': str(p.get('yıl', '')),
                    'boyut': str(p.get('boyut', '')),
                    'kondisyon': str(p.get('kondisyon', '')),
                    'orijinallik': str(p.get('orijinallik', '')),
                    'categories': [str(cat).lower() for cat in p.get('categories', [])]
                }
                docs.append(Document(page_content=page_content, metadata=meta))

            if os.path.exists(os.path.join(self.vectorstore_path, "index.faiss")):
                print("📁 Kayıtlı FAISS vectorstore bulundu. Yükleniyor...")
                vectorstore = FAISS.load_local(
                    self.vectorstore_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
            else:
                print("📁 Kayıtlı vectorstore yok. Yeni FAISS vectorstore oluşturuluyor ve kaydediliyor...")
                vectorstore = FAISS.from_documents(docs, self.embedding_model)
                vectorstore.save_local(self.vectorstore_path)

            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 100
            faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 100})

            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever],
                weights=[0.5, 0.5]
            )

            print(f"✅ {len(products)} aksesuar ürünü yüklendi ve ensemble retriever hazır.")

        except FileNotFoundError:
            print(f"❌ {json_path} dosyası bulunamadı!")
        except Exception as e:
            print(f"❌ Ürün yükleme hatası: {e}")

    def _apply_filters(self, doc: Document, parsed_query: AksesuarQuery) -> bool:
        meta = doc.metadata

        if parsed_query.brand and parsed_query.brand.upper() not in meta.get("brand", ""):
            return False
        if parsed_query.renk and parsed_query.renk.lower() not in meta.get("renk", ""):
            return False
        if parsed_query.materyal and parsed_query.materyal.lower() not in meta.get("materyal", ""):
            return False

        if parsed_query.categories:
            categories_match = any(
                cat.lower() in meta.get("categories", [])
                for cat in parsed_query.categories
            )
            if not categories_match:
                return False

        if parsed_query.yıl and parsed_query.yıl not in meta.get("yıl", ""):
            return False
        if parsed_query.boyut and parsed_query.boyut not in meta.get("boyut", ""):
            return False
        if parsed_query.kondisyon and parsed_query.kondisyon.lower() not in meta.get("kondisyon", "").lower():
            return False
        if parsed_query.orijinallik and parsed_query.orijinallik.lower() not in meta.get("orijinallik", "").lower():
            return False

        price = meta.get("price", 0)
        if parsed_query.price_exact:
            tolerance = parsed_query.price_exact * 0.05
            if not (parsed_query.price_exact - tolerance <= price <= parsed_query.price_exact + tolerance):
                return False
        else:
            if parsed_query.price_min and price < parsed_query.price_min:
                return False
            if parsed_query.price_max and price > parsed_query.price_max:
                return False

        return True

    def search(self, query_text: str, limit: int = 12) -> List[Dict]:
        if not self.ensemble_retriever:
            return []

        print(f"\n   📊 Ensemble Search (BM25 + FAISS): '{query_text}'")

        candidate_docs = self.ensemble_retriever.invoke(query_text)
        print(f"   📦 Ensemble aday sayısı: {len(candidate_docs)}")

        parsed = self.query_parser.parse_query(query_text)
        print(f"   🧠 Parsed Filtreler: {parsed.model_dump()}")

        results = []
        for doc in candidate_docs:
            if self._apply_filters(doc, parsed):
                results.append(doc)
                if len(results) >= limit:
                    break

        if not results:
            print("   ⚠️ Filtreleme sonrası sonuç yok. İlk adaylar döndürülüyor.")
            results = candidate_docs[:limit]

        output = []
        for doc in results[:limit]:
            meta = doc.metadata
            output.append({
                'type': 'aksesuar',
                'id': meta.get("id"),
                'name': meta.get("name"),
                'brand': meta.get("brand"),
                'price': meta.get("price", 0),
                'renk': meta.get("renk"),
                'materyal': meta.get('materyal'),
                'yıl': meta.get('yıl'),
                'boyut': meta.get('boyut'),
                'kondisyon': meta.get('kondisyon'),
                'orijinallik': meta.get('orijinallik'),
                'categories': meta.get("categories")
            })

        print(f"   ✅ Sonuç sayısı: {len(output)}")
        return output


# ========================
# Test için While Döngüsü
# ========================
if __name__ == "__main__":
    # GROQ API KEY artık gerekmiyor
    agent = AksesuarAgent(aksesuar_json_path="data/aksesuar.json")

    print("💍 Aksesuar Arama Sistemi Başlatıldı! (Yerel: llama3.1:latest)")
    print("Çıkmak için 'çık' veya 'exit' yazın.\n")

    while True:
        try:
            user_query = input("🔍 Aksesuar sorgusu girin: ").strip()
            if user_query.lower() in ["çık", "exit", "quit", "q"]:
                print("👋 Görüşmek üzere!")
                break

            if not user_query:
                print("⚠️ Lütfen bir sorgu girin.\n")
                continue

            results = agent.search(user_query, limit=10)

            print(f"\n✅ Toplam {len(results)} sonuç bulundu:")
            print("-" * 60)
            for i, item in enumerate(results, 1):
                print(f"{i}. {item.get('name', 'İsimsiz Ürün')}")
                print(f"   Marka: {item.get('brand', 'N/A')} | Fiyat: {item.get('price', 'N/A')} TL")
                print(f"   Renk: {item.get('renk', 'N/A')} | Kategori: {', '.join(item.get('categories', []))}")
                print()

            print("\n")

        except KeyboardInterrupt:
            print("\n\n👋 Ctrl+C alındı. Çıkılıyor...")
            break
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}\n")