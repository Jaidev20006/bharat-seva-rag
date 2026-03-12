"""
BHARAT SEVA RAG - Final api.py v9
Run: uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import os
import json
import httpx
from typing import Optional

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

VECTOR_DB_PATH  = "data/vectordb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL    = "gemini-2.0-flash"
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")

app = FastAPI(title="Bharat Seva RAG API", version="9.0.0")

@app.get("/", response_class=HTMLResponse)
async def root():
    if os.path.exists("index.html"):
        with open("index.html", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Bharat Seva AI is running!</h1>")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Starting Bharat Seva RAG API...")
collection    = None
embed_fn      = None
gemini_client = None

def load_database():
    global collection, embed_fn
    if not os.path.exists(VECTOR_DB_PATH):
        print("Vector DB not found! Run build_db.py first.")
        return False
    print("Loading embedding model...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
    collection = client.get_collection(name="bharat_seva", embedding_function=embed_fn)
    print(f"Loaded {collection.count()} chunks")
    return True

load_database()

if GEMINI_API_KEY and GEMINI_AVAILABLE:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"Gemini ready: {GEMINI_MODEL}")
else:
    print("No Gemini - using raw chunks + MyMemory translation")

class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "en"
    conversation_history: Optional[list] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int
    method: str
    language: str

def translate_to_english(text, source_lang):
    if not source_lang or source_lang == "en":
        return text
    try:
        r = httpx.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:300], "langpair": source_lang + "|en"},
            timeout=6.0
        )
        if r.status_code == 200:
            result = r.json().get("responseData", {}).get("translatedText", "")
            if result:
                print(f"  EN query: '{result[:80]}'")
                return result
        return text
    except Exception as e:
        print(f"Query translation error: {e}")
        return text

def translate_to_lang(text, target_lang):
    if not target_lang or target_lang == "en":
        return text
    try:
        r = httpx.get(
            "https://api.mymemory.translated.net/get",
            params={"q": text[:500], "langpair": "en|" + target_lang},
            timeout=8.0
        )
        if r.status_code == 200:
            result = r.json().get("responseData", {}).get("translatedText", "")
            if result:
                return result
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def detect_service(text):
    t = text.lower()
    if any(w in t for w in ["aadhaar", "aadhar", "uidai", "uid"]):
        return "aadhaar"
    if any(w in t for w in ["pan card", "pan number", "permanent account"]):
        return "pan_card"
    if "pan" in t and any(w in t for w in ["link", "lost", "apply", "download", "duplicate", "income tax", "update", "correct"]):
        return "pan_card"
    if any(w in t for w in ["passport", "tatkal", "psk", "passportindia"]):
        return "passport"
    if any(w in t for w in ["voter", "epic", "election", "eci", "voting"]):
        return "voter_id"
    if any(w in t for w in ["gst", "gstin", "goods and service", "tax registration"]):
        return "gst"
    if any(w in t for w in ["driving license", "driving licence", "learner license", "learner licence", "parivahan"]):
        return "driving_license"
    if any(w in t for w in ["digilocker", "digi locker", "digital locker"]):
        return "digilocker"
    if any(w in t for w in ["ration card", "ration", "nfsa", "food security"]):
        return "ration_card"
    if any(w in t for w in ["pm kisan", "kisan", "ayushman", "pmjay", "mudra", "pm awas", "pmay", "yojana", "scheme", "ujjwala", "jan dhan", "sukanya"]):
        return "pm_yojanas"
    return None

def build_search_query(question, service):
    t = question.lower()
    if service == "aadhaar":
        if any(w in t for w in ["reject", "failed", "problem", "issue", "error", "not working"]):
            return "Aadhaar update rejected failed problem fix"
        if any(w in t for w in ["document", "required", "needed", "what to bring"]):
            return "Aadhaar documents required proof of identity address"
        if any(w in t for w in ["download", "e-aadhaar", "pdf"]):
            return "Aadhaar download e-Aadhaar online PDF"
        if any(w in t for w in ["mobile", "phone", "otp not"]):
            return "Aadhaar update mobile number phone change"
        if any(w in t for w in ["address", "update", "change"]):
            return "Aadhaar update address online change"
        if any(w in t for w in ["lost", "damage", "reprint"]):
            return "Aadhaar card lost damaged reprint order"
        if any(w in t for w in ["name", "correction", "date of birth"]):
            return "Aadhaar name update correction date of birth"
        if any(w in t for w in ["link", "bank", "lock", "status", "check"]):
            return "Aadhaar link bank account status check"
        return "Aadhaar new enrollment how to apply get Aadhaar card"
    if service == "pan_card":
        if any(w in t for w in ["link", "aadhaar"]):
            return "PAN card link Aadhaar income tax mandatory"
        if any(w in t for w in ["lost", "duplicate", "reprint"]):
            return "PAN card lost duplicate reprint download"
        if any(w in t for w in ["name", "correction", "update", "change", "detail"]):
            return "PAN card name correction update change details"
        if any(w in t for w in ["not received", "delayed", "track", "status"]):
            return "PAN card not received delayed track status"
        if any(w in t for w in ["two", "surrender", "illegal"]):
            return "Two PAN cards surrender duplicate illegal"
        if any(w in t for w in ["minor", "child", "below 18"]):
            return "PAN card for minor child below 18 years"
        return "PAN card apply online instant e-PAN free"
    if service == "passport":
        if any(w in t for w in ["child", "minor", "baby", "son", "daughter", "below 18", "kid"]):
            return "passport for child minor below 18 years documents"
        if any(w in t for w in ["tatkal", "urgent", "fast", "emergency"]):
            return "tatkal passport urgent fast track emergency"
        if any(w in t for w in ["renew", "renewal", "expired", "reissue"]):
            return "passport renewal reissue expired damaged"
        if any(w in t for w in ["track", "status", "check", "police"]):
            return "passport track status application police verification"
        if any(w in t for w in ["lost", "stolen", "fir"]):
            return "passport lost stolen police complaint FIR"
        if any(w in t for w in ["reject", "refused", "why"]):
            return "passport rejected application reason how to fix"
        if any(w in t for w in ["document", "required", "checklist"]):
            return "passport documents required list checklist"
        if any(w in t for w in ["appointment", "book", "slot", "psk"]):
            return "passport appointment book reschedule PSK"
        return "passport apply fresh new application steps"
    if service == "voter_id":
        if any(w in t for w in ["download", "e-epic", "digital"]):
            return "voter ID download e-EPIC digital voter card"
        if any(w in t for w in ["name", "correction", "address", "update"]):
            return "voter ID name correction update address change"
        if any(w in t for w in ["lost", "damage", "reprint"]):
            return "voter ID not received lost damaged reprint"
        if any(w in t for w in ["deleted", "removed", "not in list"]):
            return "voter ID deleted name removed from voter list"
        return "voter ID new registration apply Form 6"
    if service == "gst":
        if any(w in t for w in ["return", "file", "gstr"]):
            return "GST file returns GSTR-1 GSTR-3B monthly annual"
        if any(w in t for w in ["login", "password", "forgot", "reset"]):
            return "GST login password forgot reset problem"
        if any(w in t for w in ["cancel", "surrender", "close"]):
            return "GST cancellation surrender close business"
        if any(w in t for w in ["invoice", "calculate", "hsn", "rate"]):
            return "GST invoice rules tax calculation HSN SAC"
        if any(w in t for w in ["refund", "itc", "input tax"]):
            return "GST refund claim process input tax credit ITC"
        return "GST new registration apply online GSTIN"
    if service == "driving_license":
        if any(w in t for w in ["renew", "renewal", "expired"]):
            return "driving license renewal expired renew online"
        if any(w in t for w in ["lost", "damage", "duplicate"]):
            return "driving license lost damaged duplicate apply"
        if any(w in t for w in ["permanent", "after learner"]):
            return "permanent driving license apply test parivahan"
        if any(w in t for w in ["address", "name", "correction", "update"]):
            return "driving license address change name correction update"
        if any(w in t for w in ["document", "required", "checklist"]):
            return "driving license documents required checklist"
        if any(w in t for w in ["fail", "failed", "rejected", "retake"]):
            return "driving license test failed rejected what to do"
        return "driving license learner license apply online parivahan"
    if service == "digilocker":
        if any(w in t for w in ["login", "password", "pin", "forgot"]):
            return "DigiLocker login problem password forgot PIN reset"
        if any(w in t for w in ["download", "marksheet", "degree"]):
            return "DigiLocker download documents Aadhaar PAN marksheet degree"
        if any(w in t for w in ["not showing", "not found", "unavailable", "error"]):
            return "DigiLocker document not showing not found unavailable"
        return "DigiLocker create account setup how to use"
    if service == "ration_card":
        if any(w in t for w in ["document", "required", "eligibility"]):
            return "ration card documents required eligibility"
        if any(w in t for w in ["add", "member", "newborn", "baby"]):
            return "ration card add name member family newborn"
        if any(w in t for w in ["not working", "rejected", "cancelled", "problem"]):
            return "ration card not working rejected cancelled suspended"
        if any(w in t for w in ["transfer", "surrender", "migration", "new state"]):
            return "ration card transfer surrender migration new state"
        return "ration card apply new how to get state process"
    if service == "pm_yojanas":
        if any(w in t for w in ["kisan", "farmer", "installment"]):
            return "PM Kisan apply registration status check payment"
        if any(w in t for w in ["ayushman", "health", "hospital"]):
            return "Ayushman Bharat health card apply PMJAY"
        if any(w in t for w in ["mudra", "loan", "business"]):
            return "MUDRA loan apply business Pradhan Mantri"
        if any(w in t for w in ["awas", "house", "home", "housing"]):
            return "PM Awas Yojana apply housing loan subsidy"
        if any(w in t for w in ["ujjwala", "lpg", "gas"]):
            return "Ujjwala Yojana LPG gas connection apply free"
        if any(w in t for w in ["jan dhan", "zero balance", "bank account"]):
            return "Jan Dhan account zero balance bank account open"
        if any(w in t for w in ["sukanya", "girl child"]):
            return "Sukanya Samriddhi girl child savings account"
        if any(w in t for w in ["pension", "apy", "atal"]):
            return "APY Atal Pension Yojana NPS national pension"
        return "PM Kisan apply registration status check payment"
    return question

def retrieve_chunks(question, n=4):
    if not collection:
        return []
    try:
        service = detect_service(question)
        query   = build_search_query(question, service)
        print(f"  Service: {service} | Search: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            sim  = 1 - results["distances"][0][i]
            meta = results["metadatas"][0][i]
            if service and meta.get("service") != service:
                continue
            if sim > 0.1:
                chunks.append({"text": results["documents"][0][i], "metadata": meta, "sim": round(sim, 3)})
        if not chunks:
            for i in range(len(results["documents"][0])):
                sim = 1 - results["distances"][0][i]
                if sim > 0.1:
                    chunks.append({"text": results["documents"][0][i], "metadata": results["metadatas"][0][i], "sim": round(sim, 3)})
                    break
        return chunks
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def ask_gemini(question, chunks, language):
    if not gemini_client:
        return None
    lang_names = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
                  "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "en": "English"}
    lang = lang_names.get(language, "English")
    context = "\n\n---\n\n".join([f"[{c['metadata']['title']} | {c['metadata']['url']}]\n{c['text']}" for c in chunks])
    prompt = f"You are Bharat Seva AI. Answer ONLY from context. Use numbered steps. Include official website and helpline. Reply in {lang}.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}")
        return None

def build_raw_answer(chunks):
    best   = chunks[0]
    answer = best["text"]
    if len(chunks) > 1 and chunks[1]["metadata"].get("service") == best["metadata"].get("service"):
        answer += "\n\n" + chunks[1]["text"]
    return answer

OFF_TOPIC = ["cricket", "movie", "song", "recipe", "weather", "sport", "bollywood", "bitcoin", "celebrity", "girlfriend", "boyfriend", "love"]
def is_govt_topic(text):
    t = text.lower()
    GOVT_KEYWORDS = ["aadhaar", "aadhar", "pan", "passport", "voter", "gst", "driving", "digilocker", "ration", "yojana", "scheme", "kisan", "ayushman", "mudra", "tell me", "how to", "what is", "apply", "download", "update", "lost", "help"]
    if any(w in t for w in GOVT_KEYWORDS):
        return True
    return not any(w in t for w in OFF_TOPIC)

@app.get("/")
def root():
    return {"status": "running", "chunks": collection.count() if collection else 0}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty message")

    if not is_govt_topic(question):
        return ChatResponse(
            answer="I only answer questions about Indian government services - Aadhaar, PAN, Passport, GST, Voter ID, Driving License, PM Schemes, DigiLocker, Ration Card.",
            sources=[], chunks_used=0, method="filtered", language=request.language
        )

    print(f"\nQuery: {question[:80]}")

    # Translate non-English query to English for RAG matching
    en_question = translate_to_english(question, request.language) if request.language and request.language != "en" else question

    chunks = retrieve_chunks(en_question, n=4)
    print(f"Found: {len(chunks)} chunks")

    if not chunks:
        answer = "I could not find specific information about that.\n\nHelplines:\n- Aadhaar (UIDAI): 1947\n- Passport: 1800-258-1800\n- PAN / Income Tax: 1800-180-1961\n- GST: 1800-103-4786\n- Election Commission: 1950\n- Transport (DL): 1800-120-1553"
        if request.language and request.language != "en":
            answer = translate_to_lang(answer, request.language)
        return ChatResponse(answer=answer, sources=[], chunks_used=0, method="no_match", language=request.language)

    answer = ask_gemini(en_question, chunks, request.language)
    method = "rag_gemini"
    if not answer:
        answer = build_raw_answer(chunks)
        method = "rag_direct"
        if request.language and request.language != "en":
            print(f"Translating answer to {request.language}...")
            answer = translate_to_lang(answer, request.language)

    sources = []
    seen = set()
    for c in chunks:
        url = c["metadata"]["url"]
        if url not in seen:
            seen.add(url)
            sources.append({"title": c["metadata"]["title"], "url": url, "source": c["metadata"].get("source", ""), "relevance": c["sim"]})

    return ChatResponse(answer=answer, sources=sources, chunks_used=len(chunks), method=method, language=request.language)

@app.get("/search")
async def search(q: str, n: int = 3):
    chunks = retrieve_chunks(q, n=n)
    return {"query": q, "results": [{"preview": c["text"][:200], "title": c["metadata"]["title"], "url": c["metadata"]["url"], "similarity": c["sim"]} for c in chunks]}

@app.get("/stats")
async def stats():
    if not collection:
        return {"error": "Database not loaded"}
    return {"total_chunks": collection.count(), "embedding_model": EMBEDDING_MODEL, "api_key_set": bool(GEMINI_API_KEY), "gemini_model": GEMINI_MODEL}
