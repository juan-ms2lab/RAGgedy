Here’s a step-by-step guide to building a Retrieval-Augmented Generation (RAG) system for a set of 1,000 documents
Below is a complete, minimal example for your use case, integrating Ollama as the LLM backend, ChromaDB for retrieval, and a Streamlit UI for an interactive experience.

***

## 1. Prerequisites: Install Required Tools

```bash
# Ollama (for GPT-OSS-20B local inference)
brew install ollama
ollama pull gpt-oss:20b

# Python libraries for RAG system
pip install chromadb pdfplumber sentence-transformers streamlit
```

***

## 2. Update Your Directory Structure

```
rag_project/
├── extract_and_chunk.py    # no change
├── build_index.py          # no change
├── app.py                  # NEW (Streamlit UI)
├── docs/                   # your PDF and TXT files
└── db/                     # chroma vector db storage
```

You will reuse the scripts from earlier for extraction, chunking, and building the index.
Now, create a new file called `app.py`:

***

## 3. `app.py` (Streamlit + Ollama RAG Interface)

```python
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess
import json

DB_DIR = "db"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"

def get_top_chunks(query, top_k=5):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    embedder = SentenceTransformer(EMBED_MODEL)
    query_emb = embedder.encode([query])
    result = collection.query(query_embeddings=query_emb, n_results=top_k)
    return result['documents'][0]

def ollama_infer(prompt, model="gpt-oss:20b"):
    # Run Ollama from command line and capture output
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=120
        )
        return proc.stdout.decode("utf-8")
    except Exception as e:
        return str(e)

def build_prompt(query, contexts):
    context = "\n".join(f"- {c.strip()}" for c in contexts)
    sys_prompt = "You are a helpful assistant. Use only the provided context to answer."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query.strip()}\nAnswer:"
    return f"{sys_prompt}\n{user_prompt}"

st.title("Local RAG with GPT-OSS-20B (Ollama)")

query = st.text_area("Ask a question about your documents:", height=100)
top_k = st.slider("Number of chunks to retrieve", 3, 10, 5)

if st.button("Get Answer") and query.strip():
    with st.spinner("Retrieving context and generating answer..."):
        contexts = get_top_chunks(query, top_k=top_k)
        full_prompt = build_prompt(query, contexts)
        answer = ollama_infer(full_prompt)
        st.markdown("### Answer")
        st.write(answer)
        st.markdown("### Retrieved Context")
        for idx, context in enumerate(contexts, 1):
            st.markdown(f"**Chunk {idx}:** {context}")

st.markdown("---")
st.markdown("Powered by [Ollama](https://ollama.com/) + [ChromaDB](https://www.trychroma.com/) + [SentenceTransformers](https://www.sbert.net/)")
```

***

## 4. How to Run the System

1. Start the Ollama LLM server:
   ```bash
   ollama run gpt-oss:20b
   ```
   (Keep this terminal running; Ollama will listen for model requests.)

2. Open another terminal and launch your Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Navigate to the provided `localhost` URL shown by Streamlit, enter your question, and wait for answers.

***

## 5. Notes and Tips

- Ollama will use the downloaded GGUF model, running inference fully locally.
- If you want a faster, smaller model for initial testing, try `llama2:7b` or `mistral:7b` (just run `ollama pull llama2` or `ollama pull mistral` and change the `model` argument).
- With 1,000 documents, retrieval is quick and everything runs natively on M1 without cloud or API keys.

***

You now have an end-to-end local RAG system with an interactive web interface and LLM answering, all running on your machine! Let me know if you want further tweaks (such as support for document upload or result exporting).

[1](https://huggingface.co/unsloth/gpt-oss-20b-GGUF)