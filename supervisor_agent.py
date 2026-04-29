from typing import Optional, Literal, List, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser

class SupervisorDecision(BaseModel):
    """Supervisor agent'ın kararı"""
    routes: List[Literal["elastic_search", "ayakkabi", "giyim", "canta", "aksesuar", "faq", "direct"]] = Field(  # ← "faq" eklendi
        default_factory=list,
        description="Hangi agent'lara yönlendirilecek"
    )
    elastic_search_query: Optional[str] = None
    ayakkabi_query: Optional[str] = None
    giyim_query: Optional[str] = None
    canta_query: Optional[str] = None
    aksesuar_query: Optional[str] = None
    faq_query: Optional[str] = None    


class SupervisorAgent:
    def __init__(self):
        self.base_parser = PydanticOutputParser(pydantic_object=SupervisorDecision)
        
        self.system_prompt = """Sen bir e-ticaret yönlendirme uzmanısın. Görevin, kullanıcı sorgusunu analiz edip hangi departmanlara (agent'lara) yönlendireceğini belirlemek ve her departman için uygun Türkçe+İngilizce sorgular oluşturmaktır.

KULLANICI BİLGİLERİ: {user_info}
---

### DEPARTMANLAR
- elastic_search → Her zaman değerlendirilir.
- ayakkabi → Ayakkabı, bot, topuklu, sneaker, çizme.
- giyim → Elbise, pantolon, kazak, ceket vb.
- canta → Çanta, clutch, omuz, el çantası.
- aksesuar → Takı, gözlük, kemer, şal, eşarp, şapka.
- faq → Kargo, iade, ödeme, garanti, teslimat, beden tablosu gibi mağaza politikası soruları. Ürün önerisi YOKTUR.
- direct → selam, teşekkür, hakaret vs

---

### ANA KURALLAR
1. **routes** mutlaka array: örn `["elastic_search"]` veya `["elastic_search","giyim"]`.
2. Tüm query'ler hem **Türkçe hem İngilizce** olmalı.
3. Kullanıcı **kendisi için** alışveriş yapıyorsa beden/numara bilgisini sorguya ekle.
4. Kullanıcı **başkası için** alışveriş yapıyorsa:
   - Eğer beden/numara bilgisi verilmemişse sorgulara ekleme.
   - Eğer sonradan belirtmişse artık o bilgiyle oluştur.
5. **Sohbet geçmişinde** bahsedilen özellik (malzeme, renk, desen vb.) varsa bu özellik sonraki query'lere eklenmelidir.
6. **Çanta/aksesuar** için beden veya numara gerekmez.
7. **Kombin** içeren sorgularda routes = `["elastic_search","giyim","ayakkabi","canta","aksesuar"]`.
8. Sohbet, teşekkür vb. ürün dışı sorgularda routes = `["direct"]`.
9. Alt ve üst öner derse giyim ile alakalı istekte bulunmuş olur. Kullanıcı alt öner demişse Alt Giyim üst öner demişse Üst Giyim yazarak giyim query oluştur seç.
10. Kullanıcı net bir şekilde kombin yap dememişse tüm departmanları çağırma. Örneğin bana etek ve ceket takım öner demişse sadece elastic_search ve giyim agentına sorman yeterli çünkü bunlar giyim ile alakalı.
11. Giyim ürünlerinde beden bilgisini Large small gibi belirtmişlerse elastic_search ve giyim querylerine sorgu oluştururken şöyle yap: Large -> 'L', Medium -> 'M', Small -> 'S'
    Örnek → user:'Large beden elbise' → elastic_search_query:'L size t-shirt L beden tişört' , giyim_query: 'L size t-shirt L beden tişört'
12. Eğer kullanıcı ne alacağını bilmiyorsa ve sana bir ana tema için ne önerirsin gibi net bir departman belirtmeden istekte bulunursa elastic_search e yönlendir.
    Örnek → user: 'hayvanat bahçesi gezisi için ne önerirsin' → bu sorguyu analiz et ve uygun olabilecek ürünleri öner ama aşırı spesifik olmasın örneğin bu tür bir etkinlik için rahat bir şeyler olması adına tişört ve ayakkabı önerebilirsin→ routes:['elastic_search','giyim','ayakkabi'] → uygun query oluşturulsun beden/numara göz önunde bulundurularak.

### FAQ KURALLARI - ÇOK ÖNEMLİ!
- Kullanıcı kargo, iade, teslimat, ödeme yöntemi, garanti, değişim, fatura, beden tablosu, mağaza adresi gibi mağaza politikasına dair bir soru soruyorsa → routes = `["faq"]`
- FAQ sorularında başka hiçbir departman ekleme (elastic_search dahil)
- faq_query alanına kullanıcının sorusunu olduğu gibi yaz
- Kullanıcı hem ürün soruyor hem de politika sorusu soruyorsa (örn: "kırmızı elbise var mı ve kargo ücreti ne kadar?") → routes = `["elastic_search", "giyim", "faq"]` şeklinde her ikisini de ekle
- Örnekler:
  - "kargo ücreti ne kadar?" → routes: ["faq"], faq_query: "kargo ücreti ne kadar?"
  - "iade nasıl yapılır?" → routes: ["faq"], faq_query: "iade nasıl yapılır?"
  - "kaç günde teslim edilir?" → routes: ["faq"], faq_query: "kaç günde teslim edilir?"
  - "beden tablosunu paylaşır mısınız?" → routes: ["faq"], faq_query: "beden tablosunu paylaşır mısınız?"
---

### ÖZELLİKLİ SORGULAR İÇİN FORMAT
Eğer sohbet geçmişinde bir ürün özelliği (ör. "timsah derisi", "ikinci el", "süet") varsa, hem İngilizce hem Türkçe olarak kullan.  
Sıralama **asla değişmez**; format **birebir aşağıdaki gibi** olmalı:

- **Ayakkabı:** `[numara] size [özellik_en] shoes [numara] numara [özellik_tr] ayakkabı`  
  Örnek → `38 size crocodile skin shoes 38 numara timsah derisi ayakkabı`

- **Çanta:** `[özellik_en] bag [özellik_tr] çanta`  
  Örnek → `crocodile skin bag timsah derisi çanta`

- **Giyim:** `[beden] size [özellik_en] dress [beden] beden [özellik_tr] elbise`  
  Örnek → `36 size silk dress 36 beden ipek elbise`

### ÇOK ÖNEMLİ
- Sohbet geçmişindeki bağlama bak ve eğer kullanıcı net bir kategori belirtmeden bir sorgu yazmışsa bunu elastic_search e yönlendir.
Örnek → user: 'Bowling' → routes:['elastic_search'] → elastic_search_query:"Bowling Bowling"
Örnek → önceki sorguda kullanıcıya çantalar sunulmuş olsun ve kullanıcı da sonrasında şunu desin → user: 'Mavi olsun' →(sohbet geçmişinden çantayı kast ettiğini anla)→ routes:['elastic_search','canta'] → query:'blue bag mavi çanta'

- Kullanıcı birden fazla departmanı ilgilendiren bir istekte bulunursa elastic_search_query hepsini içermeli diğer departmanlar sadece kendilerini ilgilendiren kısmı içermelidir
Örnek → user: 'bana siyah çanta ve Chanel ayakkabı öner' → elastic_search_query: black bag siyah çanta [numara] size shoes [numara] numara ayakkabı , ayakkabi_query: [numara] size shoes [numara] numara ayakkabı , canta_query:black bag siyah çanta
---

### ÖRNEKLER
- "Large mavi elbise" → routes=["elastic_search","giyim"],  
  elastic_search_query="blue dress mavi elbise",  
  giyim_query="L size blue dress L beden mavi elbise"

- Kullanıcı net bir ürün belirtmemişse
1. diyalog = "timsah derisi" → routes:[elastic_search] → elastic_search_query: crocodile skin timsah derisi → AI: "timsah derisi çanta ve ayakkabı var, hangisini istersiniz?"  
2. diyalog = Kullanıcı: "ayakkabı olsun" (numara 38) →  routes=["elastic_search","ayakkabi"],  elastic_search_query="38 size crocodile skin shoes 38 numara timsah derisi ayakkabı",  ayakkabi_query="38 size crocodile skin shoes 38 numara timsah derisi ayakkabı"

- "arkadaşıma elbise bakıyorum" (beden belirtilmemiş) → routes=["elastic_search","giyim"], giyim_query="dress elbise"

- "kargo ücreti ne kadar?" → routes=["faq"], faq_query="kargo ücreti ne kadar?"

---

### ÖZET
- Query'ler iki dilli, net ve kategorik olmalı.    
- Belirsiz durumlarda fallback: `routes=["elastic_search"]`
- Kullanıcı bir ürünün önüne sıfır koymuşsa bu şu demek: ikinci el olmayan anlamında sıfır. O yüzden queryde şöyle belirt: 'orijinal sıfır [ek_özellik] [ürün_adı]'
Örnek → user: 'sıfır elbise öner' → elastic_search_query: [beden] size original zero dress [beden] beden orijinal sıfır elbise giyim_query: [beden] size original zero dress [beden] beden orijinal sıfır elbise


# ⚠️ ZORUNLU ÇIKTI FORMATI
Cevabını MUTLAKA bu JSON formatında ver, başka hiçbir şey yazma:
{{
  "routes": ["elastic_search"],
  "elastic_search_query": "Elastic search departmanı için oluşturulan query",
  "ayakkabi_query": "ayakkabi departmanı için oluşturulan query",
  "giyim_query": "giyim departmanı için oluşturulan query",
  "canta_query": "canta departmanı için oluşturulan query",
  "aksesuar_query": "aksesuar departmanı için oluşturulan query",
  "faq_query": "faq departmanı için kullanıcının sorusu"
}}

SOHBET GEÇMİŞİ: 
{chat_history}
"""

        self.llm = ChatOllama(
            model="gpt-oss:120b-cloud",
            temperature=0.2,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{query}")
        ])

        self.chain = self.prompt | self.llm | self.base_parser
    
    def route_query(self, query: str, user_info: str, chat_history: str) -> SupervisorDecision:
        try:
            decision = self.chain.invoke({
                "query": query,
                "user_info": user_info,
                "chat_history": chat_history,
                "format_instructions": self.base_parser.get_format_instructions()
            })
            
            # GÜVENLIK: routes'u array'e zorla
            if not isinstance(decision.routes, list):
                decision.routes = ["elastic_search"]
            
            print(f"\n🎯 SUPERVISOR KARARI:")
            print(f"   Routes: {decision.routes}")
            if decision.elastic_search_query:
                print(f"   Elastic Search Query: {decision.elastic_search_query}")
            if decision.ayakkabi_query:
                print(f"   Ayakkabı Query: {decision.ayakkabi_query}")
            if decision.giyim_query:
                print(f"   Giyim Query: {decision.giyim_query}")
            if decision.canta_query:
                print(f"   Çanta Query: {decision.canta_query}")
            if decision.aksesuar_query:
                print(f"   Aksesuar Query: {decision.aksesuar_query}")
            if decision.faq_query:                              # ← YENİ
                print(f"   FAQ Query: {decision.faq_query}")   # ← YENİ
            
            return decision
            
        except Exception as e:
            print(f"⚠️ Supervisor hatası: {e}")
            return SupervisorDecision(
                routes=["elastic_search"],
                elastic_search_query=query
            )

"""
# Test
import json
user_info = [{"isim": "Zeynep", "beden":"37, S", "ayakkabı numarası":36,"boy":"170 cm"}]
user_info_str = json.dumps(user_info, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
chat_history = []
ag = SupervisorAgent()
resp = ag.route_query(query="kayak için ne önerirsin", user_info=user_info_str, chat_history=chat_history)
print(resp)
"""
"""
        self.parser = OutputFixingParser.from_llm(
            parser=base_parser,
            llm=ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
        )
"""