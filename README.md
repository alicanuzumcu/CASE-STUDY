# 🛍️ Gevheri Store — Multi-Agent Shopping Assistant

LangGraph tabanlı çok ajanlı (multi-agent) e-ticaret asistan sistemi. Kullanıcı sorgularını analiz ederek ilgili ürün ajanlarına yönlendirir; kargo, iade gibi mağaza politikası sorularını ise FAQ PDF'inden yanıtlar.

---

## 📁 Proje Yapısı

```
project/
├── main.py                  # FastAPI sunucusu & LangGraph sistemi
├── supervisor_agent.py      # Sorgu yönlendirme ajanı
├── faq_agent.py             # FAQ PDF ajanı (BM25 + FAISS hybrid)
├── ayakkabi_agent.py
├── giyim_agent.py
├── canta_agent.py
├── aksesuar_agent.py
├── final_agent.py
├── bm_25.py                 # BM25 arama motoru
├── products.json
├── data/
│   ├── FAQ.pdf
│   ├── ayakkabi.json
│   ├── giyim.json
│   ├── canta.json
│   └── aksesuar.json
├── vectorstore/
│   └── faiss_faq/           # Otomatik oluşturulur
└── templates/
    └── index.html
```

---

## ⚙️ Gereksinimler

- Python 3.10+
- Node.js (opsiyonel, docx üretimi için)
- Ollama (yerel LLM sunucusu)
- `gpt-oss:120b-cloud` modeli Ollama üzerinde çalışır durumda olmalı

---

## 🚀 Kurulum

### 1. Repoyu klonla

```bash
git clone <repo-url>
cd <proje-klasörü>
```

### 2. Sanal ortam oluştur ve aktifleştir

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 4. Ollama kurulumu

[https://ollama.com](https://ollama.com) adresinden Ollama'yı indirip kurun, ardından modeli çekin:

```bash
ollama pull gpt-oss:120b-cloud
```

Ollama'nın arka planda çalıştığından emin olun:

```bash
ollama serve
```

### 5. Veri dosyalarını yerleştir

`data/` klasörü altına aşağıdaki dosyaları ekleyin:

| Dosya | Açıklama |
|---|---|
| `data/FAQ.pdf` | Mağaza sıkça sorulan sorular belgesi |
| `data/ayakkabi.json` | Ayakkabı ürün verisi |
| `data/giyim.json` | Giyim ürün verisi |
| `data/canta.json` | Çanta ürün verisi |
| `data/aksesuar.json` | Aksesuar ürün verisi |
| `products.json` | BM25 için birleşik ürün verisi (proje kökünde) |

---

## ▶️ Çalıştırma

```bash
python main.py
```

Sunucu başarıyla başladığında terminalde şunu görmelisiniz:

```
✅ FastAPI server hazır!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Tarayıcıdan `http://127.0.0.1:8000` adresine giderek web arayüzünü kullanabilirsiniz.

---

## 🔌 API Kullanımı

### Web Arayüzü (Form)

`POST /` — HTML form üzerinden sorgu gönderir, yanıtı aynı sayfada gösterir.

### REST API

`POST /chat`

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "kargo ücreti ne kadar?",
    "user_info": [{"isim": "Zeynep", "beden": "36, S", "ayakkabı numarası": 38}],
    "thread_id": "kullanici-1"
  }'
```

**Yanıt:**

```json
{
  "response": "Kargo ücretleri hakkında...",
  "selected_ids": []
}
```

---

## 🤖 Agent Akışı

```
Kullanıcı Sorgusu
      │
      ▼
 SupervisorAgent
 (Sorguyu analiz eder, route belirler)
      │
      ├── elastic_search  →  BM25Retriever
      ├── ayakkabi        →  AyakkabiAgent
      ├── giyim           →  GiyimAgent
      ├── canta           →  CantaAgent
      ├── aksesuar        →  AksesuarAgent
      └── faq             →  FAQAgent (BM25 + FAISS Hybrid)
                                  │
                                  ▼
                             FinalAgent
                          (Yanıt oluşturur)
```

### Route Örnekleri

| Kullanıcı Sorusu | Route |
|---|---|
| `"kargo ücreti ne kadar?"` | `["faq"]` |
| `"iade nasıl yapılır?"` | `["faq"]` |
| `"siyah çanta öner"` | `["elastic_search", "canta"]` |
| `"36 numara mavi elbise"` | `["elastic_search", "giyim"]` |
| `"kombin öner"` | `["elastic_search", "giyim", "ayakkabi", "canta", "aksesuar"]` |

---

## 🗂️ FAISS Vektör Deposu

İlk çalıştırmada `FAQ.pdf` okunur ve `vectorstore/faiss_faq/` klasörüne kaydedilir. Sonraki başlatmalarda mevcut indeks yüklenir (yeniden hesaplama yapılmaz).

Vektör deposunu sıfırlamak için:

```bash
rm -rf vectorstore/faiss_faq/
```

---

## 🛠️ Sık Karşılaşılan Hatalar

| Hata | Çözüm |
|---|---|
| `FAQAgent object has no attribute 'answer'` | `main.py` içinde `faq_agent.answer()` → `faq_agent.generate_answer()` olarak değiştirin |
| `FileNotFoundError: data/FAQ.pdf` | `data/` klasörüne `FAQ.pdf` dosyasını ekleyin |
| `Connection refused` (Ollama) | `ollama serve` komutunu çalıştırın |
| `FAISS index not found` | İlk çalıştırmada otomatik oluşturulur, beklemeniz yeterli |

---

## 📝 Notlar

- Sohbet geçmişi `MemorySaver` ile `thread_id` bazında tutulur; her kullanıcı oturumu izoledir.
- FAQ ajanı hem BM25 (anahtar kelime) hem FAISS (anlamsal) arama yapar; sonuçlar `0.5 / 0.5` ağırlıkla birleştirilir.
- Embedding modeli: `intfloat/multilingual-e5-large-instruct` (Türkçe destekli).