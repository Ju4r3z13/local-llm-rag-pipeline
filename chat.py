import chromadb
from ollama import Client


# THIS WHOLE SECTION HAS ALREADY BEEN EXPLAINED IN THE ingest.py FILE SO I WILL SKIP THE EXPLANATION UNTIL THE CHATBOT SECTION
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:4b-instruct"                                        # Replace this with the name of your model
ollama = Client(host="http://localhost:11434")
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection(
    name="workshop_documents"
)

# Chatbot :)
print("Local RAG Chatbot")
print("Type 'exit' to quit.\n")
while True:                                                            # User input
    question = input("You: ")
    if question.lower() == "exit":
        break
    # Create embedding for the question
    response = ollama.embed(                                           # Turn the question into a vector
        model=EMBEDDING_MODEL,                                         # Why? Makes the question comparable to the numerical vectors made from the documents
        input=question
    )
    query_embedding = response["embeddings"][0]
    # Search ChromaDB
    results = collection.query(                                        # Retrieve the chunks whose embeddings are the most relevant to the question
        query_embeddings=[query_embedding],                            # (this is why we turned the input into a vector :b)
        n_results=3
    )
    # Get retrieved documents
    retrieved_documents = results["documents"][0]
    context = "\n\n".join(retrieved_documents)
    # Build prompt                                                     # Prompt the LLM for a response using the context from the chunks
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
