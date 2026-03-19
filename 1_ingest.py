import fitz #PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, json

#Open every PDF in the data folder, read every page, and save the text along with where it came from.
def load_pdfs(data_folder="data/"):
    all_text = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(data_folder, filename))
            for page in doc:
                all_text.append({
                    "text": page.get_text(),
                    "source": filename,
                    "page": page.number + 1
                })
    return all_text



def chunk_documents(pages, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "source": page["source"],
                "page": page["page"]
            })
    return chunks

if __name__ == "__main__":
    print("Loading PDFs from data/ folder...")
    pages = load_pdfs("data/")
    print(f"Loaded {len(pages)} pages.")

    chunks = chunk_documents(pages)
    with open("chunks.json", "w") as f:
        json.dump(chunks, f)
    print(f"✅ Created {len(chunks)} chunks. Saved to chunks.json")