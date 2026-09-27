import os
import chromadb
from ollama import Client

'''
This Python script is a core component of the RAG process and it works as follows:
Takes documents -> breaks them into chunks -> converts them into embeddings -> stores them in ChromaDB
'''

# If you were to be using this as a template, here you would replace with:
DOCUMENTS_DIR = "./documents"                                                # The directory of your source docs
CHROMA_DIR = "./chroma_db"                                                   # The directory of your vector DB stored data
EMBEDDING_MODEL = "qwen3-embedding:0.6b"                                     # The embedding model you chose

# Connect to Ollama and ChromaDB
ollama = Client(host="http://localhost:11434")                               # Communicates your script to the local Ollama server 
chroma = chromadb.PersistentClient(path=CHROMA_DIR)                          # PersistentClient stores the data on disk (to retain the embeddings)

# Create or open collection
# A collection of related documents and chunks inside the DB
collection = chroma.get_or_create_collection(                                # If it exists it will use it, otherwise it will create it
    name="workshop_documents"
)

# Read and process documents
document_files = [                                                           # Loop to get all .txt files. Just a simple file storing into a list loop
    file for file in os.listdir(DOCUMENTS_DIR)
    if file.endswith(".txt")                                                 # Most likely you won't be only reading .txt files so you would definitely have to modify this. EZ :)
]
if not document_files:
    print("No .txt files found in the documents folder.")
    exit()
print(f"Found {len(document_files)} document(s).")
for filename in document_files:                                              # File reading loop. Again, this is some freshman year stuff, no need to explain each line
    filepath = os.path.join(DOCUMENTS_DIR, filename)
    print(f"\nProcessing: {filename}")
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()
        
    # Split document into chunks
    chunk_size = 1000                                                        # How many characters do we want in each chunk? (chunking is very useful to process large files)
    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]
    print(f"Created {len(chunks)} chunks.")

    # Embed and store each chunk
    for i, chunk in enumerate(chunks):                                       
        response = ollama.embed(                                             # Convert each chunk into a numerical representation 
            model=EMBEDDING_MODEL,                                           # Semantically similar text gets represented by vectors that are relatively close together
            input=chunk                                                      # That's embedding in a nutshell
        )
        embedding = response["embeddings"][0]
        collection.upsert(                                                   # Store in ChromaDB
            ids=[f"{filename}-{i}"],                                         # All of this is below is metadata for the DB
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": filename,
                "chunk": i
            }]
        )
print("\nAll documents successfully added to ChromaDB.\n")
