from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import os


class AyakkabiQuery(BaseModel):
    brand: Optional[str] = Field(default=None)
    renk1: Optional[str] = Field(default=None)
    renk2: Optional[str] = Field(default=None)
    numara: Optional[int | float] = Field(default=None)
    price_min: Optional[int | float] = Field(default=None)
    price_max: Optional[int | float] = Field(default=None)
    price_exact: Optional[int | float] = Field(default=None)
    materyal: Optional[str] = Field(default=None)
    topuk_boyu: Optional[int | float] = Field(default=None)
    platform_boyu: Optional[int | float] = Field(default=None)
    categories: Optional[List[str]] = Field(default=None)


class AyakkabiQueryParser:
    def __init__(self, brands_list: List[str]):  # groq_api_key kaldırıldı
        brands_str = ", ".join(brands_list)
        self.system_prompt = f"""Sen bir ayakkabı mağazası uzmanısın. Kullanıcı sorgusunu analiz et ve filtreleme parametrelerini çıkar.

        Kurallar:
        -> Kullanıcının sorgusuna bakarak şunları eğer bilgi varsa doldur yoksa da null bırak:
            - brand: Eğer kullanıcı marka belirtmişse doldur
            - renk1: Eğer müşteri ana renk belirtmişse MUTLAKA STRING olarak doldur
            - renk2: Eğer müşteri ikinci renk belirtmişse MUTLAKA STRING olarak doldur
            - numara: Eğer müşteri ayakkabı numarası olarak tam sayı belirtmişse int, Float bir numara belirtmişse float değer ile doldur
            - price_min: Eğer kullanıcı minimum fiyat veya bütçe belirtmişse float değer ile doldur
            - price_max: Eğer kullanıcı maximum fiyat veya bütçe belirtmişse float değer ile doldur
            - price_exact: Eğer kullanıcı net bir fiyat veya bütçe belirtmişse float değer ile doldur
            - materyal: Eğer kullanıcı materyal belirtmişse doldur (deri, süet, kumaş vs)
            - topuk_boyu: Eğer kullanıcı topuk boyu belirtmişse float değer ile doldur
            - platform_boyu: Eğer kullanıcı platform boyu belirtmişse float değer ile doldur
            - categories: Eğer kullanıcı ayakkabı türleri belirtmişse liste olarak doldur. (Kategoriler: Ayakkabı, Topuklu Ayakkabı, Bot & Çizme, Sneaker & Spor Ayakkabı, Terlik, Sandalet, Babet, Loafer, Slingback, Espadril, Stiletto.)

        ip ucu:
            - Eğer kullanıcı sivri uçlu olan bir üründen bahsediyorsa bu stiletto kategorisine girer,
            ama eğer yuvarlak veya oval gibi topuklu ayakkabı diyorsa bu topuklu ayakkabı kategorisine girer.

        -> Ürünleri markalara göre filtrelerken marka isimlerini doğru yazdığından emin ol ve filtrede yalnızca şu markaları yaz: Markalar: {brands_str}
        -> MARKA DÜZELTMELERİ (ÇOK ÖNEMLİ):
           Kullanıcı yanlış yazsa bile doğru marka ismini bul:
           - "celline", "seline" → CELINE
           - "gucci", "guci", "gucı" → GUCCI
           - "prada", "pırada" → PRADA
           - "dior", "kristian dior" → CHRISTIAN DIOR
           - "versace", "versaçe" → VERSACE
           - "blü marine", "bulumarin" → BLUMARINE
           Benzer şekilde tüm markalar için en yakın eşleşmeyi kullan.
        -> renk numara materyal gibi filtreleme parametrelerini özel isim değilse MUTLAKA TÜRKÇE olarak oluştur.
        """

        # ✅ Ollama ile llama3.1:latest modeli
        self.llm = ChatOllama(
            model="gpt-oss:120b-cloud",
            temperature=0.1,
        ).with_structured_output(AyakkabiQuery)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{query}")
        ])

        self.chain = self.prompt | self.llm

    def parse_query(self, query: str) -> AyakkabiQuery:
        try:
            parsed_query = self.chain.invoke({"query": query})
            return parsed_query
        except Exception as e:
            print(f"⚠️ Parser hatası: {e}")
            return AyakkabiQuery()


class AyakkabiAgent:
    def __init__(self, ayakkabi_json_path: str = "data/ayakkabi.json"):  # groq_api_key kaldırıldı
        self.brands_list = self.get_brands_from_json(ayakkabi_json_path)
        self.query_parser = AyakkabiQueryParser(self.brands_list)  # API key yok
        self.embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
        self.ensemble_retriever = None
        self.vectorstore_path = "vectorstore/faiss_ayakkabi"
        os.makedirs(self.vectorstore_path, exist_ok=True)
        self.load_json(ayakkabi_json_path)

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
                    'numara': str(p.get('numara', '')),
                    'materyal': str(p.get('materyal', '')).lower(),
                    'topuk_boyu': p.get('topuk_boyu'),
                    'platform_boyu': p.get('platform_boyu'),
                    'kondisyon': p.get('kondisyon', ''),
                    'orijinallik': p.get('orijinallik', ''),
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

            print(f"✅ {len(products)} ayakkabı ürünü yüklendi ve ensemble retriever hazır.")

        except FileNotFoundError:
            print(f"❌ {json_path} dosyası bulunamadı!")
        except Exception as e:
            print(f"❌ Ürün yükleme hatası: {e}")

    def _apply_filters(self, doc: Document, parsed_query: AyakkabiQuery) -> bool:
        meta = doc.metadata

        if parsed_query.brand and parsed_query.brand.upper() not in meta.get("brand", ""):
            return False

        if parsed_query.renk1 or parsed_query.renk2:
            ürün_rengi = meta.get("renk", "").lower()
            if not (
                (parsed_query.renk1 and parsed_query.renk1.lower() in ürün_rengi) or
                (parsed_query.renk2 and parsed_query.renk2.lower() in ürün_rengi)
            ):
                return False

        if parsed_query.numara:
            numara_str = str(parsed_query.numara)
            meta_numara = str(meta.get("numara", ""))
            if numara_str not in meta_numara:
                return False

        if parsed_query.materyal and parsed_query.materyal.lower() not in meta.get("materyal", ""):
            return False

        if parsed_query.categories:
            if not any(cat.lower() in meta.get("categories", []) for cat in parsed_query.categories):
                return False

        if parsed_query.topuk_boyu:
            meta_topuk = meta.get("topuk_boyu")
            if meta_topuk is None or str(parsed_query.topuk_boyu) not in str(meta_topuk):
                return False

        if parsed_query.platform_boyu:
            meta_platform = meta.get("platform_boyu")
            if meta_platform is None or str(parsed_query.platform_boyu) not in str(meta_platform):
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
                'type': 'ayakkabi',
                'id': meta.get("id"),
                'name': meta.get("name"),
                'brand': meta.get("brand"),
                'price': meta.get("price", 0),
                'renk': meta.get("renk"),
                'numara': meta.get('numara'),
                'materyal': meta.get('materyal'),
                'topuk_boyu': meta.get('topuk_boyu'),
                'platform_boyu': meta.get('platform_boyu'),
                'kondisyon': meta.get('kondisyon'),
                'orijinallik': meta.get('orijinallik'),
                'categories': meta.get("categories")
            })

        print(f"   ✅ Sonuç sayısı: {len(output)}")
        return output


# ========================
# Test Döngüsü
# ========================
if __name__ == "__main__":
    # GROQ API KEY artık gerekmiyor
    agent = AyakkabiAgent(ayakkabi_json_path="data/ayakkabi.json")

    print("👟 Ayakkabı Arama Sistemi Başlatıldı! (Yerel: llama3.1:latest)")
    print("Çıkmak için 'çık' veya 'exit' yazın.\n")

    while True:
        try:
            user_query = input("🔍 Ayakkabı sorgusu girin: ").strip()
            if user_query.lower() in ["çık", "exit", "quit", "q"]:
                print("👋 Görüşmek üzere!")
                break
            if not user_query:
                print("⚠️ Lütfen bir sorgu girin.\n")
                continue

            results = agent.search(user_query, limit=12)

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