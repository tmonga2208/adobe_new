import os
import fitz  # For PDF text extraction
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Optional # For type hinting

# --- Configuration ---
class Config:
    def __init__(self):
        # Input directory for PDF files
        self.input_dir: str = r"C:\Users\hp\Desktop\adobe_new\input" # Changed from single pdf_path
        # Name for the ChromaDB collection
        self.index_name: str = "vbl-local"
        # Path for ChromaDB persistent storage
        self.chromadb_persist_path: str = "./vbl_chromaa"
        # Chunk size for text splitting
        self.chunk_size: int = 800
        # Embedding model name
        self.embedding_model_name: str = "all-MiniLM-L6-v2"

# --- Step 1: PDF Text Extraction ---
def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extracts text from a single PDF file.
    Returns the extracted text or None if an error occurs.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        print(f"  Successfully extracted text from: {os.path.basename(pdf_path)}")
        return text
    except Exception as e:
        print(f"  ERROR: Could not extract text from {os.path.basename(pdf_path)}: {e}")
        return None

def get_pdf_paths(input_directory: str) -> List[str]:
    """
    Scans the input directory for PDF files and returns their full paths.
    """
    if not os.path.exists(input_directory):
        print(f"ERROR: Input directory not found: {input_directory}")
        return []

    pdf_files = [f for f in os.listdir(input_directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in the directory: {input_directory}")
        return []

    return [os.path.join(input_directory, f) for f in pdf_files]

# --- Step 2: Text Chunking ---
def chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    Splits a given text into smaller chunks of specified size.
    """
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# --- Step 3: Embedding Model Setup ---
def setup_embedding_model(model_name: str):
    """
    Loads the Sentence Transformer model and creates the embedding function.
    """
    print(f"Loading embedding model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        print("Embedding model loaded successfully.")
        return model, embedding_function
    except Exception as e:
        print(f"ERROR: Failed to load embedding model {model_name}: {e}")
        exit() # Exit if model cannot be loaded, as it's a critical dependency

# --- Step 4: ChromaDB Setup ---
def setup_chromadb_collection(
    persist_path: str,
    collection_name: str,
    embedding_func: SentenceTransformerEmbeddingFunction
):
    """
    Sets up a persistent ChromaDB client and collection.
    Deletes existing collection if it already exists.
    """
    print(f"Setting up ChromaDB collection: '{collection_name}' at '{persist_path}'...")
    try:
        client = chromadb.PersistentClient(path=persist_path)

        if collection_name in [c.name for c in client.list_collections()]:
            print(f"  Collection '{collection_name}' already exists. Deleting and recreating.")
            client.delete_collection(name=collection_name)

        collection = client.create_collection(name=collection_name, embedding_function=embedding_func)
        print("ChromaDB collection created/recreated successfully.")
        return client, collection
    except Exception as e:
        print(f"ERROR: Failed to set up ChromaDB collection: {e}")
        exit() # Exit if DB cannot be set up

# --- Step 5: Add Chunks to ChromaDB ---
def add_chunks_to_chromadb(
    collection,
    chunks: List[str],
    pdf_file_name: str, # To ensure unique IDs per document
    start_chunk_id_idx: int = 0
) -> int:
    """
    Adds a list of text chunks to the ChromaDB collection.
    Generates unique IDs based on the PDF file name.
    Returns the number of chunks added.
    """
    added_count = 0
    document_ids = []
    document_chunks = []

    for i, chunk in enumerate(chunks):
        # Create a unique ID that includes the PDF file name and chunk index
        # Use os.path.splitext to get the base name without extension
        base_file_name = os.path.splitext(pdf_file_name)[0]
        unique_id = f"{base_file_name}_chunk-{start_chunk_id_idx + i}"
        document_ids.append(unique_id)
        document_chunks.append(chunk)

    if document_chunks:
        try:
            collection.add(
                documents=document_chunks,
                ids=document_ids
            )
            added_count = len(document_chunks)
            print(f"  Added {added_count} chunks from {pdf_file_name} to ChromaDB.")
        except Exception as e:
            print(f"  ERROR: Failed to add chunks from {pdf_file_name} to ChromaDB: {e}")
    return added_count


# --- Main Pipeline Execution ---
def run_ingestion_pipeline():
    """
    Executes the full PDF text ingestion pipeline.
    """
    config = Config() # Load configuration

    print("--- Starting PDF Ingestion Pipeline ---")
    print(f"Input Directory: {config.input_dir}")
    print(f"ChromaDB Path: {config.chromadb_persist_path}")
    print(f"ChromaDB Collection: {config.index_name}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Embedding Model: {config.embedding_model_name}")

    # Step 3: Setup Embedding Model
    _, embedding_function = setup_embedding_model(config.embedding_model_name)

    # Step 4: Setup ChromaDB
    chroma_client, chroma_collection = setup_chromadb_collection(
        config.chromadb_persist_path,
        config.index_name,
        embedding_function
    )

    # Step 1: Get PDF paths from the input directory
    pdf_paths = get_pdf_paths(config.input_dir)
    if not pdf_paths:
        print("No PDFs found or input directory issue. Exiting pipeline.")
        return

    total_chunks_processed = 0
    # Process each PDF sequentially
    for i, pdf_path in enumerate(pdf_paths):
        pdf_file_name = os.path.basename(pdf_path) # Get just the file name
        print(f"\n--- Processing Document {i+1}/{len(pdf_paths)}: {pdf_file_name} ---")

        # Step 1: Extract Text
        text = extract_text_from_pdf(pdf_path)
        if text is None:
            continue # Skip to the next PDF if extraction failed

        # Step 2: Chunk Text
        chunks = chunk_text(text, config.chunk_size)
        print(f"  Generated {len(chunks)} chunks.")

        # Step 5: Add Chunks to ChromaDB
        chunks_added = add_chunks_to_chromadb(
            chroma_collection,
            chunks,
            pdf_file_name,
            start_chunk_id_idx=0
        )
        total_chunks_processed += chunks_added

    print(f"\n--- Pipeline Completed ---")
    print(f"Total PDFs processed: {len(pdf_paths)}")
    print(f"Total chunks embedded and added to ChromaDB: {total_chunks_processed}")
    print(f"ChromaDB collection '{config.index_name}' contains {chroma_collection.count()} items.")

# --- Execute the Pipeline ---
if __name__ == "__main__":
    run_ingestion_pipeline()