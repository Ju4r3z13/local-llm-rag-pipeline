import os
import chromadb
from ollama import Client

DOCUMENTS_DIR = "./documents"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Connect to Ollama and ChromaDB
ollama = Client(host="http://localhost:11434")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or open collection
collection = chroma.get_or_create_collection(
    name="workshop_documents"
)

# Read and process documents
document_files = [
    file for file in os.listdir(DOCUMENTS_DIR)
    if file.endswith(".txt")
]
if not document_files:
    print("No .txt files found in the documents folder.")
    exit()
print(f"Found {len(document_files)} document(s).")
for filename in document_files:
    filepath = os.path.join(DOCUMENTS_DIR, filename)
    print(f"\nProcessing: {filename}")
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
        
    # Split document into chunks
    chunk_size = 1000
    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]
    print(f"Created {len(chunks)} chunks.")

    # Embed and store each chunk
    for i, chunk in enumerate(chunks):
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunk
        )
        embedding = response["embeddings"][0]
        collection.upsert(
            ids=[f"{filename}-{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": filename,
                "chunk": i
            }]
        )
print("\nAll documents successfully added to ChromaDB.\n")
