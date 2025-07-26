import os
import fitz  # For PDF text extraction
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# === CONFIG ===
pdf_path = r"C:\pdfs\vbl.pdf"
index_name = "vbl-local"

# === STEP 1: Extract Text from PDF ===
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

if not os.path.exists(pdf_path):
    print("PDF file not found.")
    exit()

text = extract_text_from_pdf(pdf_path)
print("PDF text extracted.")

# === STEP 2: Chunk the Text ===
def chunk_text(text, chunk_size=800):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

text_chunks = chunk_text(text)
print(f"Total chunks created: {len(text_chunks)}")

# === STEP 3: Setup Local Embedding Model ===
print("Loading local embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# === STEP 4: Setup ChromaDB (Persistent Storage) ===
client = chromadb.PersistentClient(path="./vbl_chromaa")

if index_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=index_name)

collection = client.create_collection(name=index_name, embedding_function=embedding_function)
print("Local ChromaDB collection created.")

# === STEP 5: Add Chunks to ChromaDB ===
for i, chunk in enumerate(text_chunks):
    collection.add(
        documents=[chunk],
        ids=[f"id-{i}"]
    )
print("All chunks embedded and added to local DB.")
