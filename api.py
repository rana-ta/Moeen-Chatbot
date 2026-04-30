import json                                                          
import math                                                          
from collections import defaultdict 

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware    

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder                   

#  ------------------- Configurations -----------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "faq_collection"
BM25_INDEX_PATH = "bm25_index.json"                                  

# cosine distance: looks for the smaller result
MAX_DISTANCE = 0.30
TOP_K = 10                                                           
BM25_WEIGHT = 0.4                                                    
SEMANTIC_WEIGHT = 0.6                                                

#  -------------------  FastAPI Setup -----------------------

app = FastAPI(title="FAQ Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#  ------------------- Request and Load Models -----------------------

class ChatRequest(BaseModel):
    query: str

# The Embedding model used for semantic search 
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Cross-encoder model used for for reranking results (Choosing the best)
cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

#  ------------------- Connecting to Chroma Database -----------------------

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embed_fn
)

#  ------------------- Loading the BM25 -----------------------
                    
with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
    bm25_data = json.load(f)

BM25_IDS: list         = bm25_data["ids"]
BM25_DOCS: list        = bm25_data["tokenized_docs"]
BM25_DF: dict          = bm25_data["df"]
BM25_DOC_COUNT: int    = bm25_data["doc_count"]
BM25_AVGDL: float      = bm25_data["avgdl"]


#  ------------------- BM25 scoring function-----------------------
                                        
def bm25_score(query_tokens: list[str], doc_index: int, k1=1.5, b=0.75) -> float:
    doc = BM25_DOCS[doc_index]
    dl = len(doc)
    tf_map = defaultdict(int)
    for t in doc:
        tf_map[t] += 1

    score = 0.0
    for term in query_tokens:
        if term not in BM25_DF:
            continue
        idf = math.log((BM25_DOC_COUNT - BM25_DF[term] + 0.5) / (BM25_DF[term] + 0.5) + 1)
        tf = tf_map[term]
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / BM25_AVGDL))
    return score

#  ------------------- API Endpoints-----------------------

@app.get("/")
def home():
    return {"message": "FAQ chatbot API is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    query = request.query.strip()

    if not query:
        return {"answer": "Please enter a question", "answered": False, "distance": None}

   

    # 1: Semantic Search
    results = collection.query(
        query_texts=[query],                              
        n_results=TOP_K,
        include=["distances", "metadatas", "documents"]
    )

    results_ids   = results.get("ids", [[]])[0]
    results_dists = results.get("distances", [[]])[0]
    results_metas = results.get("metadatas", [[]])[0]

    if not results_ids:
        return {"answer":"Sorry, no answer was found for your question.", "answered": False, "distance": None}

    # 3: BM25 scoring                 
    query_tokens = query.split()
    RRF_K = 60

    semantic_rank = {doc_id: rank for rank, doc_id in enumerate(results_ids)}

    bm25_scores = {}
    for doc_id in results_ids:
        if doc_id in BM25_IDS:
            idx = BM25_IDS.index(doc_id)
            bm25_scores[doc_id] = bm25_score(query_tokens, idx)

    # ranking the BM25 candidates to get the best answer
    bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_ranked)}

    
    rrf_scores = defaultdict(float)
    for doc_id in results_ids:
        s_rank = semantic_rank.get(doc_id, TOP_K)
        b_rank = bm25_rank.get(doc_id, TOP_K)
        rrf_scores[doc_id] += SEMANTIC_WEIGHT * (1 / (RRF_K + s_rank + 1))
        rrf_scores[doc_id] += BM25_WEIGHT     * (1 / (RRF_K + b_rank + 1))

    # 4: sort by RRF score, take top 5 for reranking
    top_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_ids = [doc_id for doc_id, _ in top_candidates]

    # 5: Cross-encoder reranking                         
    candidate_metas = {
        results_ids[i]: results_metas[i]
        for i in range(len(results_ids))
        if results_ids[i] in top_ids
    }

    pairs = [(query, candidate_metas[doc_id].get("question", "")) for doc_id in top_ids]
    rerank_scores = cross_encoder.predict(pairs)

    best_local_idx = int(rerank_scores.argmax())
    best_id = top_ids[best_local_idx]

    # get original semantic distance for threshold check
    best_distance = results_dists[results_ids.index(best_id)] if best_id in results_ids else 1.0
    best_metadata = candidate_metas.get(best_id, {})

    #  6: Threshold check
    if best_distance > MAX_DISTANCE:
        return {
            "answer":"Sorry, no answer was found for your question. Try another question",
            "answered": False,
            "distance": best_distance,
            "matched_id": best_id
        }
#  ------------------- The Final response ----------------------

    return {
        "answer": best_metadata.get("answer", "Sorry, no answer was found for your question. Try another question"),
        "answered": True,
        "distance": best_distance,
        "matched_id": best_id,
        "matched_question": best_metadata.get("question", ""),
        "rerank_score": float(rerank_scores[best_local_idx]),        
    }
