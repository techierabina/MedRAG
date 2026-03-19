from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"

def load_retriever():
    model = SentenceTransformer(EMBED_MODEL)
    index = faiss.read_index("faiss_index.bin")
    with open("chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

def retrieve(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[idx] for idx in indices[0]]

def build_prompt(query, context_chunks):
    context = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    ])
    return f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {query}
Answer:"""

def answer(query):
    # Load retriever
    embed_model, index, chunks = load_retriever()
    top_chunks = retrieve(query, embed_model, index, chunks, top_k=3)
    prompt = build_prompt(query, top_chunks)

    # Load Flan-T5 directly (correct way for seq2seq models)
    print("⏳ Loading Flan-T5 model...")
    tokenizer = T5Tokenizer.from_pretrained(GEN_MODEL)
    model     = T5ForConditionalGeneration.from_pretrained(GEN_MODEL)

    # Tokenize and generate
    inputs  = tokenizer(prompt, return_tensors="pt",
                        max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100) 
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response, top_chunks

if __name__ == "__main__":
    query = "What percentage of people with diabetes have good control of glycaemia?"
    print(f"\n🔍 Question: {query}\n")

    response, sources = answer(query)

    print(f"\n💬 Answer: {response}")
    print("\n📚 Sources used:")
    for s in sources:
        print(f"  - {s['source']}, Page {s['page']}")