import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer

# === Step 1: Load Chroma Collection ===
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./vbl_chroma")
collection = client.get_collection("vbl-local", embedding_function=embedding_fn)

# === Step 2: Ask a Question ===
def ask(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    scores = results.get("distances", [[0]*top_k])[0]  # get similarity scores

    # Print results
    print("\n🔍 Top Matches:")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"\n📌 [{i+1}] (Score: {score:.4f})\n{doc[:400]}...\n")

    return docs, scores

# === Step 3: Simple rule-based response ===
def answer_from_results(query, docs):
    print("🤖 Answering based on context:")
    print("="*80)
    print(f"Q: {query}")
    print("-"*80)
    print("A:")
    print(docs[0])
    print("="*80)

# === Main Execution ===
if __name__ == "__main__":
    query = input("Enter your question: ")

    docs, scores = ask(query)

    # If there's no document or score is not confident (threshold), reject
    if not docs or docs[0].strip() == "" or scores[0] > 1.0:
        print("❌ Sorry, this question is outside the knowledge of the VBL 2024 Annual Report.")
    else:
        answer_from_results(query, docs)
