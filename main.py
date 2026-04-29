from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from supervisor_agent import SupervisorAgent
from bm_25 import BM_25_Retriever
from ayakkabi_agent import AyakkabiAgent
from giyim_agent import GiyimAgent
from canta_agent import CantaAgent
from aksesuar_agent import AksesuarAgent
from faq_agent import FAQAgent          # ← YENİ
from final_agent import FinalAgent
import json
from pprint import pprint
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import os

API_KEY = "ds-dXqYxzkjzWBUHnVaHg31KkK00MWp67EfuQk611JPbggY26JhWGAo8jHfa"

class ShoppingState(TypedDict):
    user_query: str
    user_info: str
    chat_history: List[Dict[str, str]]
    routes: List[str]
    elastic_search_query: str
    ayakkabi_query: str
    giyim_query: str
    canta_query: str
    aksesuar_query: str
    faq_query: str                      # ← YENİ
    elastic_search_results: List[Dict]
    ayakkabi_results: List[Dict]
    giyim_results: List[Dict]
    canta_results: List[Dict]
    aksesuar_results: List[Dict]
    faq_results: str                    # ← YENİ (FAQ cevabı düz metin)
    selected_ids: List[int]
    final_response: str

class LangGraphShoppingSystem:
    def __init__(self,
                 ayakkabi_json: str = "data/ayakkabi.json",
                 giyim_json: str = "data/giyim.json",
                 canta_json: str = "data/canta.json",
                 aksesuar_json: str = "data/aksesuar.json",
                 products_json: str = "products.json"):
        
        print("🚀 LangGraph Multi-Agent Shopping System başlatılıyor...\n")
        self.supervisor = SupervisorAgent()
        self.ayakkabi_agent = AyakkabiAgent(ayakkabi_json)
        self.giyim_agent = GiyimAgent(giyim_json)
        self.canta_agent = CantaAgent(canta_json)
        self.aksesuar_agent = AksesuarAgent(aksesuar_json)
        self.faq_agent = FAQAgent()     # ← YENİ
        self.final_agent = FinalAgent()
        self.elastic_search_engine = BM_25_Retriever(products_json)
        self.app = self._build_graph()
        print("\n✅ LangGraph sistemi hazır!\n")
    
    def _build_graph(self):
        workflow = StateGraph(ShoppingState)
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("elastic_search", self.elastic_search_node)
        workflow.add_node("ayakkabi", self.ayakkabi_node)
        workflow.add_node("giyim", self.giyim_node)
        workflow.add_node("canta", self.canta_node)
        workflow.add_node("aksesuar", self.aksesuar_node)
        workflow.add_node("faq", self.faq_node)             # ← YENİ
        workflow.add_node("final", self.final_node)
        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges("supervisor", self.route_agents, {
            "elastic_search": "elastic_search",
            "ayakkabi": "ayakkabi",
            "giyim": "giyim",
            "canta": "canta",
            "aksesuar": "aksesuar",
            "faq": "faq",               # ← YENİ
            "final": "final"
        })
        workflow.add_edge("elastic_search", "final")
        workflow.add_edge("ayakkabi", "final")
        workflow.add_edge("giyim", "final")
        workflow.add_edge("canta", "final")
        workflow.add_edge("aksesuar", "final")
        workflow.add_edge("faq", "final")                   # ← YENİ
        workflow.add_edge("final", END)
        return workflow.compile(checkpointer=MemorySaver())
    
    def supervisor_node(self, state: ShoppingState) -> ShoppingState:
        print(f"\n{'='*80}\n💬 KULLANICI: {state['user_query']}\n{'='*80}\n")
        user_info = state.get("user_info", "")
        chat_history = state.get("chat_history", [])
        chat_history_str = self._format_chat_history(chat_history)
        decision = self.supervisor.route_query(query=state['user_query'], user_info=user_info, chat_history=chat_history_str)
        return {
            "routes": decision.routes,
            "elastic_search_query": decision.elastic_search_query or state["user_query"],
            "ayakkabi_query": decision.ayakkabi_query or state['user_query'],
            "giyim_query": decision.giyim_query or state['user_query'],
            "canta_query": decision.canta_query or state['user_query'],
            "aksesuar_query": decision.aksesuar_query or state['user_query'],
            # Supervisor'dan faq_query gelmiyorsa user_query'yi fallback olarak kullan
            "faq_query": getattr(decision, "faq_query", None) or state['user_query'],  # ← YENİ
        }
    
    def _format_chat_history(self, chat_history: List[Dict]) -> str:
        if not chat_history:
            return "Sohbet geçmişi boş."
        return "\n".join([f"{msg.get('role', '').upper()}: {msg.get('content', '')}" for msg in chat_history])
    
    def route_agents(self, state: ShoppingState) -> List[str]:
        routes = state.get("routes", [])
        if not isinstance(routes, list):
            return ["elastic_search"]

        # Sadece "faq" route'u varsa → sadece faq node'una git
        if routes == ["faq"]:
            return ["faq"]

        if "direct" in routes and "elastic_search" not in routes:
            return ["final"]
        if "direct" in routes and "elastic_search" in routes:
            return ["elastic_search"]
        if not routes:
            return ["elastic_search"]
        return routes
    
    def elastic_search_node(self, state: ShoppingState) -> ShoppingState:
        query = state["elastic_search_query"]
        results = self.elastic_search_engine.get_filtered_results(query=query, max_results=8)
        print(f"\n{'='*80}\n🌀 ELASTIC SEARCH ÇALIŞIYOR...\n   ✅ {len(results)} sonuç bulundu\nSONUÇLAR:")
        pprint(results, width=100, sort_dicts=False)
        return {"elastic_search_results": results}

    def ayakkabi_node(self, state: ShoppingState) -> ShoppingState:
        query = state['ayakkabi_query']
        routes = state["routes"]
        limit = 20 if len(routes) == 2 else 10
        results = self.ayakkabi_agent.search(query, limit=limit)
        print(f"\n{'='*80}\n👠 AYAKKABI AGENT'I ÇALIŞIYOR...\n   ✅ {len(results)} sonuç bulundu\nSONUÇLAR:")
        pprint(results, width=100, sort_dicts=False)
        return {"ayakkabi_results": results}

    def giyim_node(self, state: ShoppingState) -> ShoppingState:
        query = state['giyim_query']
        routes = state["routes"]
        limit = 20 if len(routes) == 2 else 10
        results = self.giyim_agent.search(query, limit=limit)
        print(f"\n{'='*80}\n👕 GİYİM AGENT'I ÇALIŞIYOR...\n   LIMIT: {limit} ✅ {len(results)} sonuç bulundu\nSONUÇLAR:")
        pprint(results, width=100, sort_dicts=False)
        return {"giyim_results": results}

    def canta_node(self, state: ShoppingState) -> ShoppingState:
        query = state['canta_query']
        routes = state["routes"]
        limit = 20 if len(routes) == 2 else 10
        results = self.canta_agent.search(query, limit=limit)
        print(f"\n{'='*80}\n👜 ÇANTA AGENT'I ÇALIŞIYOR...\n   ✅ {len(results)} sonuç bulundu\nSONUÇLAR:")
        pprint(results, width=100, sort_dicts=False)
        return {"canta_results": results}

    def aksesuar_node(self, state: ShoppingState) -> ShoppingState:
        query = state['aksesuar_query']
        routes = state["routes"]
        limit = 16 if len(routes) == 2 else 8
        results = self.aksesuar_agent.search(query, limit=limit)
        print(f"\n{'='*80}\n💍 AKSESUAR AGENT'I ÇALIŞIYOR...\n   ✅ {len(results)} sonuç bulundu\nSONUÇLAR:")
        pprint(results, width=100, sort_dicts=False)
        return {"aksesuar_results": results}

    def faq_node(self, state: ShoppingState) -> ShoppingState:
        query = state["faq_query"]
        print(f"\n{'='*80}\n❓ FAQ AGENT ÇALIŞIYOR...\n   Soru: {query}")
        answer = self.faq_agent.generate_answer(query)  # ← DÜZELTME
        print(f"   ✅ Cevap üretildi\n{'='*80}")
        return {"faq_results": answer}

    def final_node(self, state: ShoppingState) -> ShoppingState:
        print(f"\n{'='*80}\n🎯 FINAL AGENT CEVAP OLUŞTURUYOR...\n{'='*80}")
        user_info = state.get("user_info", "")
        chat_history = state.get("chat_history", [])
        chat_history_str = self._format_chat_history(chat_history)
        
        response_text, selected_ids = self.final_agent.generate_response(
            query=state['user_query'], user_info=user_info, chat_history=chat_history_str,
            elastic_search_results=state.get("elastic_search_results", []),
            ayakkabi_results=state.get("ayakkabi_results", []),
            giyim_results=state.get("giyim_results", []),
            canta_results=state.get("canta_results", []),
            aksesuar_results=state.get("aksesuar_results", []),
            faq_results=state.get("faq_results", ""),       # ← YENİ
        )
        
        current_history = state.get("chat_history", []).copy()
        current_history = current_history[4:] if len(current_history) >= 4 else current_history
        current_history.append({"role": "user", "content": state['user_query']})
        current_history.append({"role": "assistant", "content": response_text})
        
        print(f"{'='*80}\n")
        return {"final_response": response_text, "selected_ids": selected_ids, "chat_history": current_history}
    
    def process_query(self, user_query: str, user_info: List[Dict], thread_id: str = "default") -> tuple[str, List[int]]:
        user_info_str = json.dumps(user_info, ensure_ascii=False)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            current_state = self.app.get_state(config)
            chat_history = current_state.values.get("chat_history", []) if current_state.values else []
        except:
            chat_history = []
        
        initial_state = {
            "user_query": user_query, "user_info": user_info_str, "chat_history": chat_history,
            "routes": [], "elastic_search_query": "", "ayakkabi_query": "", "giyim_query": "",
            "canta_query": "", "aksesuar_query": "",
            "faq_query": "", "faq_results": "",             # ← YENİ
            "elastic_search_results": [],
            "ayakkabi_results": [], "giyim_results": [], "canta_results": [],
            "aksesuar_results": [], "selected_ids": [], "final_response": ""
        }
        
        result = self.app.invoke(initial_state, config)
        return result["final_response"], result.get("selected_ids", [])


shopping_system_api = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global shopping_system_api
    shopping_system_api = LangGraphShoppingSystem(
        ayakkabi_json="data/ayakkabi.json",
        giyim_json="data/giyim.json",
        canta_json="data/canta.json",
        aksesuar_json="data/aksesuar.json",
        products_json="products.json"
    )
    print("✅ FastAPI server hazır!")
    yield

app = FastAPI(title="Deluxe Seconds Shopping Assistant", lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

class ChatRequest(BaseModel):
    query: str
    user_info: List[Dict]
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    selected_ids: List[int]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"response": None}
    )

@app.post("/", response_class=HTMLResponse)
async def chat_web(request: Request, user_input: str = Form(...)):
    if not shopping_system_api:
        raise HTTPException(status_code=503, detail="Sistem henüz hazır değil")
    
    user_info = [{"isim": "Zeynep", "beden": "36, S", "ayakkabı numarası": 38, "boy": "170 cm"}]
    thread_id = "web-user-2"
    
    try:
        llm_response, selected_ids = shopping_system_api.process_query(
            user_query=user_input,
            user_info=user_info,
            thread_id=thread_id
        )

        results = [{"id": pid} for pid in selected_ids]

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"response": {"llm": llm_response, "results": results}}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_api(request: ChatRequest):
    if not shopping_system_api:
        raise HTTPException(status_code=503, detail="Sistem henüz hazır değil")
    try:
        response, selected_ids = shopping_system_api.process_query(
            user_query=request.query,
            user_info=request.user_info,
            thread_id=request.thread_id
        )
        return ChatResponse(response=response, selected_ids=selected_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)