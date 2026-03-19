from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_retriever():
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index("faiss_index.bin")
    with open("chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

def retrieve(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    results = []
    for idx in indices[0]:
        results.append(chunks[idx])
    return results

if __name__ == "__main__":
    print("Loading retriever...")
    model, index, chunks = load_retriever()

    query = "What are the symptoms of diabetes?"
    print(f"\n🔍 Query: {query}\n")

    results = retrieve(query, model, index, chunks)
    for i, r in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"📄 Source: {r['source']} | Page {r['page']}")
        print(r['text'][:300])
        print()