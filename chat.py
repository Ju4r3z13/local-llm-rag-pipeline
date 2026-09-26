import chromadb
from ollama import Client

# Configuration

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Replace this with the exact name of your model
LLM_MODEL = "qwen3:4b-instruct"

# Connect to Ollama and ChromaDB

ollama = Client(host="http://localhost:11434")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection(
    name="workshop_documents"
)

# Chatbot :)

print("Local RAG Chatbot")
print("Type 'exit' to quit.\n")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    # Create embedding for the question
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )
    query_embedding = response["embeddings"][0]
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    # Get retrieved documents
    retrieved_documents = results["documents"][0]
    context = "\n\n".join(retrieved_documents)
    # Build prompt
    prompt = f"""
        Rules:
        - Only answer questions that are directly related to information contained in the provided context.
        - If the question cannot be answered using the provided context, say:
          "I can only answer questions related to the provided documents."
        - Do not use outside knowledge to answer the question.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """
    # Ask the LLM
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response["message"]["content"]
    print(f"\nAssistant: {answer}\n")
