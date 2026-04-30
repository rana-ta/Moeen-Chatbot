import json                                                        
import pandas as pd
import chromadb

from collections import defaultdict 
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

#  ------------------- Configurations -----------------------

CSV_PATH = "FAQ_Moeen.csv"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "faq_collection"
BM25_INDEX_PATH = "bm25_index.json"                                

#  ------------------- BM25 Index Builder -----------------------

def build_bm25_index(questions: list[str], ids: list[str], path: str):
    # build a BM25 Index from the Questions dataset FAQ_Moeen.csv and stores it as a JSON.
    tokenized = [q.lower().split() for q in questions]

    from collections import defaultdict
    df = defaultdict(int)
    for doc in tokenized:
        for term in set(doc):
            df[term] += 1

    index = {
        "ids": ids,
        "tokenized_docs": tokenized,
        "df": dict(df),
        "doc_count": len(tokenized),
        "avgdl": sum(len(d) for d in tokenized) / max(len(tokenized), 1),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"BM25 index saved to {path}")

#  ------------------- The pipeline for the main indexing -----------------------

def main():
    #  1. Load the dataset -----------

    df = pd.read_csv(CSV_PATH).fillna("")

    required = {"id", "question", "answer"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    #  2. Initilize Chroma DB -----------

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    try: # delete existing collection if exist 
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    #  3. Preparing the data for indexing -----------

    ids = []
    docs = []
    metas = []
    raw_questions = []                                               

    for _, row in df.iterrows():
        doc_id = str(row["id"]).strip()
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        if not doc_id or not question or not answer:
            continue


        ids.append(doc_id)
        docs.append(question)                           
        raw_questions.append(question)                              
        metas.append({
            "question": question,                                  
            "answer": answer
        })

    #  4. Store embeddings in Chroma -----------

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas
    )

    #  5. Building the BM25 index-----------

    build_bm25_index(raw_questions, ids, BM25_INDEX_PATH)           

    print(f"Indexed {len(ids)} questions successfully into Chroma.")

if __name__ == "__main__":
    main()
