**RAG IMPLEMENTATION WORKSHOP**  
**(STEP BY STEP)**   
Eduardo Juarez 9/26/2026

**Connect to UTD Linux machine session (using UNIX terminal or PuTTY)**  
ssh \[NETID\]@[giant.utdallas.edu](http://giant.utdallas.edu)  
(complete sign in steps…)

**Clone workshop repository**   
git clone [https://github.com/Ju4r3z13/local-llm-rag-pipeline](https://github.com/Ju4r3z13/local-llm-rag-pipeline)

**Download and unzip Ollama binary (workaround sudo access at UTD linux machine)**  
mkdir \-p \~/ollama  
cd \~/ollama  
curl \-L https://ollama.com/download/ollama-linux-amd64.tar.zst \-o ollama.tar.zst  
tar \--use-compress-program=zstd \-xf ollama.tar.zst

**Get Ollama running (keep this terminal running)**  
\~/ollama/bin/ollama serve

**(ON A SEPARATE TERMINAL)**  
**Pull models**  
\~/ollama/bin/ollama pull qwen3:4b-instruct  
\~/ollama/bin/ollama pull qwen3-embedding:0.6b

**Setup and run Python environment**  
python3 \-m venv .venv  
source .venv/bin/activate

**Get required Python packages**  
pip install chromadb ollama

**Process/Ingest documents (from where we will get our external context for RAG)**  
python [ingest.py](http://ingest.py)

**Run chatbot**  
python [chat.py](http://chat.py)

**IMPORTANT NOTES**

The LLM should never be responsible for deciding which client records it is allowed to see. Access control must happen in the application/database layer before information is sent to the LLM.

RAG does not automatically provide data isolation. Documents should be associated with appropriate metadata, and retrieval should be filtered according to the user's authorization.

The embedding model and the LLM have different jobs. The embedding model finds relevant information; the LLM uses that information to generate the response.

The LLM does not directly search ChromaDB. The application retrieves relevant information from ChromaDB and then provides that information to the LLM as context.  
