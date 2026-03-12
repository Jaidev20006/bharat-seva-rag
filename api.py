import os, re, httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.utils import embedding_functions

VECTOR_DB_PATH  = "data/vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL    = "gemini-2.0-flash"
MYMEMORY_URL    = "https://api.mymemory.translated.net/get"

app = FastAPI(title="Bharat Seva AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Bharat Seva AI is running!</h1>")

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
try:
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection("govt_services", embedding_function=ef)
    print(f"ChromaDB loaded: {collection.count()} chunks")
except Exception as e:
    print(f"ChromaDB error: {e}")
    collection = None

class ChatRequest(BaseModel):
    message: str
    language: str = "en"
    conversation_history: List[dict] = []

SERVICE_KEYWORDS = {
    "aadhaar":          ["aadhaar","aadhar","uid","uidai","enrollment","biometric"],
    "pan_card":         ["pan","permanent account","income tax","epan","nsdl"],
    "passport":         ["passport","psp","tatkal","psk","ecr","booklet"],
    "ration_card":      ["ration","pds","food","bpl","apl","nfsa"],
    "voter_id":         ["voter","epic","electoral","election","nvsp","eci"],
    "gst":              ["gst","goods and services","gstin","gstr"],
    "driving_license":  ["driving","licence","license","learner","parivahan","rto"],
    "digilocker":       ["digilocker","digi locker","digital locker"],
    "pm_yojana":        ["yojana","kisan","ayushman","mudra","awas","ujjwala","jan dhan","sukanya","scheme"],
}
SEARCH_QUERIES = {
    "aadhaar":         {"default":"aadhaar card services","apply":"aadhaar enrollment documents","update":"aadhaar update correction","download":"download e-aadhaar","lost":"lost aadhaar reprint"},
    "pan_card":        {"default":"PAN card services","apply":"apply PAN card documents","link":"link aadhaar PAN","download":"download e-PAN","lost":"lost PAN reprint"},
    "passport":        {"default":"passport services","apply":"apply passport documents","tatkal":"tatkal passport urgent","renew":"passport renewal","child":"passport minor child","lost":"lost passport FIR"},
    "ration_card":     {"default":"ration card services","apply":"new ration card documents","add":"add member ration card"},
    "voter_id":        {"default":"voter ID services","apply":"voter ID registration","download":"download e-EPIC"},
    "gst":             {"default":"GST services","apply":"GST registration documents","return":"GST return filing"},
    "driving_license": {"default":"driving license services","apply":"driving licence documents","renew":"driving licence renewal"},
    "digilocker":      {"default":"DigiLocker services","apply":"DigiLocker account create","download":"download documents DigiLocker"},
    "pm_yojana":       {"default":"PM government schemes","kisan":"PM Kisan farmer","ayushman":"Ayushman Bharat","mudra":"MUDRA loan"},
}
OFF_TOPIC = ["cricket","movie","film","song","music","actor","ipl","recipe","cook","game","football","weather","stock","crypto"]
GOVT_KW   = ["aadhaar","aadhar","pan","passport","voter","gst","driving","digilocker","ration","yojana","scheme","kisan","ayushman","mudra","how to","what is","apply","download","update","lost","help","documents","fees","status"]

async def translate_text(text, src, tgt):
    if src == tgt: return text
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(MYMEMORY_URL, params={"q": text[:400], "langpair": f"{src}|{tgt}"})
            t = r.json().get("responseData", {}).get("translatedText", "")
            return t if t and "INVALID" not in t.upper() else text
    except: return text

LANG_MAP = {"hi":"hi","ta":"ta","te":"te","bn":"bn","mr":"mr","gu":"gu","kn":"kn","ml":"ml"}

async def to_english(text, lang):
    return await translate_text(text, LANG_MAP.get(lang,"en"), "en") if lang != "en" else text

async def from_english(text, lang):
    return await translate_text(text, "en", LANG_MAP.get(lang,"en")) if lang != "en" else text

def detect_service(text):
    t = text.lower()
    for svc, kws in SERVICE_KEYWORDS.items():
        if any(k in t for k in kws): return svc
    return None

def build_query(question, service):
    t = question.lower()
    if any(w in t for w in ["child","minor","baby","below 18","kid"]): return "passport minor child documents"
    if service and service in SEARCH_QUERIES:
        sq = SEARCH_QUERIES[service]
        for intent, q in sq.items():
            if intent != "default" and intent in t: return q
        return sq["default"]
    return question

def is_relevant(text):
    t = text.lower()
    if any(w in t for w in GOVT_KW): return True
    return not any(w in t for w in OFF_TOPIC)

def get_chunks(question, service=None, n=4):
    if not collection: return []
    try:
        q = build_query(question, service)
        where = {"service": service} if service else None
        res = collection.query(query_texts=[q], n_results=min(n, collection.count()), where=where, include=["documents","metadatas","distances"])
        return [{"text": d, "meta": m, "score": s} for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]) if s < 1.5]
    except Exception as e:
        print(f"Retrieve error: {e}"); return []

async def ask_gemini(question, chunks):
    if not GEMINI_API_KEY: return None
    ctx = "\n\n".join([c["text"] for c in chunks[:3]])
    prompt = f"You are Bharat Seva AI, an expert on Indian government services.\nAnswer using ONLY this context. Be clear and use bullet points.\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt).text
    except Exception as e:
        print(f"Gemini error: {e}"); return None

def raw_answer(chunks):
    if not chunks: return "I couldn't find specific information. Please visit the official government portal or call the helpline."
    seen, parts = set(), []
    for c in chunks[:3]:
        k = c["text"][:80]
        if k not in seen:
            seen.add(k); parts.append(c["text"].strip())
    return "\n\n".join(parts)

@app.get("/stats")
async def stats():
    return {"total_chunks": collection.count() if collection else 0, "api_key_set": bool(GEMINI_API_KEY), "status": "ready" if collection else "error"}

@app.get("/search")
async def search(q: str):
    chunks = get_chunks(q)
    return {"query": q, "results": [{"text": c["text"][:200], "score": c["score"]} for c in chunks]}

@app.post("/chat")
async def chat(req: ChatRequest):
    msg  = req.message.strip()
    lang = req.language or "en"
    if not msg: return {"response": "Please type a question.", "service": None}
    en_msg = await to_english(msg, lang)
    print(f"[{lang}] '{msg}' -> '{en_msg}'")
    if not is_relevant(en_msg):
        reply = "I can only help with Indian government services like Aadhaar, PAN, Passport, Voter ID, GST, Driving License, DigiLocker, Ration Card, and PM Yojanas."
        return {"response": await from_english(reply, lang), "service": None}
    service = detect_service(en_msg)
    chunks  = get_chunks(en_msg, service)
    answer  = await ask_gemini(en_msg, chunks) or raw_answer(chunks)
    if lang != "en": answer = await from_english(answer, lang)
    return {"response": answer, "service": service}
