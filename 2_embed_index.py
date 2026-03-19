from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np, pickle

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_index(chunks_path="chunks.json"):
    with open(chunks_path) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    print("⏳ Loading embedding model (first time downloads ~80MB)...")
    model = SentenceTransformer(MODEL_NAME)

    print("⏳ Embedding 21 chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("⏳ Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.bin")
    with open("chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Done! Indexed {len(chunks)} chunks.")
    print(f"✅ Files saved: faiss_index.bin, chunks_metadata.pkl")

if __name__ == "__main__":
    build_index()