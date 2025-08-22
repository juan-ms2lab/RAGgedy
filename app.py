#!/usr/bin/env python3
"""
Streamlit RAG Interface Application
Frontend Developer Agent Implementation

Interactive web interface for the RAG system with Ollama LLM integration.
Provides document querying, result visualization, and system monitoring.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# Configuration constants
DB_DIR = "db"
COLLECTION_NAME = "rag_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MODEL = "phi3:mini"


class RAGInterface:
    """Main RAG interface class handling backend operations."""
    
    def __init__(self):
        """Initialize RAG interface with caching."""
        self._embedder = None
        self._collection = None
        self._client = None
    
    @st.cache_resource
    def get_embedder(_self):
        """Get cached embedding model."""
        if _self._embedder is None:
            _self._embedder = SentenceTransformer(EMBED_MODEL)
        return _self._embedder
    
    @st.cache_resource
    def get_chroma_client(_self):
        """Get cached ChromaDB client."""
        if _self._client is None:
            _self._client = chromadb.PersistentClient(path=DB_DIR)
        return _self._client
    
    def get_collection(self):
        """Get ChromaDB collection."""
        if self._collection is None:
            client = self.get_chroma_client()
            try:
                self._collection = client.get_collection(COLLECTION_NAME)
            except Exception as e:
                st.error(f"Could not connect to vector database: {e}")
                st.error("Please run 'python build_index.py build docs/' first")
                return None
        return self._collection
    
    def get_top_chunks(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve top-k most relevant document chunks.
        
        Args:
            query: User query text
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        collection = self.get_collection()
        if not collection:
            return []
        
        try:
            embedder = self.get_embedder()
            query_emb = embedder.encode([query])
            
            result = collection.query(
                query_embeddings=query_emb.tolist(),
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            return result
        except Exception as e:
            st.error(f"Error retrieving chunks: {e}")
            return {}
    
    def ollama_infer(self, prompt: str, model: str = DEFAULT_MODEL) -> str:
        """
        Generate response using Ollama LLM.
        
        Args:
            prompt: Input prompt for the model
            model: Ollama model name
            
        Returns:
            Generated response text
        """
        try:
            # Run Ollama with timeout and proper encoding
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                capture_output=True,
                timeout=120
            )
            
            if proc.returncode == 0:
                return proc.stdout.decode("utf-8").strip()
            else:
                error_msg = proc.stderr.decode("utf-8").strip()
                return f"Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            return "Error: Request timed out. Please try again."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def build_prompt(self, query: str, contexts: List[str], custom_prompt: str = None) -> str:
        """
        Build RAG prompt with context and query.
        
        Args:
            query: User question
            contexts: Retrieved context chunks
            custom_prompt: Optional custom system prompt
            
        Returns:
            Formatted prompt for LLM
        """
        if not contexts:
            return f"Question: {query}\nAnswer: I don't have enough information to answer this question."
        
        context = "\n".join(f"- {c.strip()}" for c in contexts if c.strip())
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            sys_prompt = custom_prompt
        else:
            sys_prompt = """You are a helpful AI assistant. Use only the provided context to answer the question. 
If the context doesn't contain enough information, say so clearly.
Be concise and accurate in your response."""
        
        user_prompt = f"""Context:
{context}

Question: {query.strip()}
Answer:"""
        
        return f"{sys_prompt}\n\n{user_prompt}"


def setup_streamlit_page():
    """Configure Streamlit page settings and styling."""
    st.set_page_config(
        page_title="RAGgedy - Local AI Assistant",
        page_icon="🧸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .context-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def check_system_status() -> Dict[str, bool]:
    """Check the status of system components."""
    status = {
        'ollama': False,
        'model': False,
        'vector_db': False,
        'docs': False
    }
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        status['ollama'] = result.returncode == 0
        
        # Check if model is available
        if status['ollama']:
            status['model'] = DEFAULT_MODEL in result.stdout
    except:
        pass
    
    # Check vector database
    try:
        client = chromadb.PersistentClient(path=DB_DIR)
        collection = client.get_collection(COLLECTION_NAME)
        status['vector_db'] = collection.count() > 0
    except:
        pass
    
    # Check docs directory
    docs_path = Path("docs")
    status['docs'] = docs_path.exists() and any(docs_path.glob("*"))
    
    return status


def display_system_status():
    """Display system status in sidebar."""
    st.sidebar.header("System Status")
    
    status = check_system_status()
    
    # Status indicators
    indicators = {
        'ollama': ('Ollama Server', '🟢' if status['ollama'] else '🔴'),
        'model': (f'Model ({DEFAULT_MODEL})', '🟢' if status['model'] else '🔴'),
        'vector_db': ('Vector Database', '🟢' if status['vector_db'] else '🔴'),
        'docs': ('Documents', '🟢' if status['docs'] else '🔴')
    }
    
    for key, (label, icon) in indicators.items():
        st.sidebar.write(f"{icon} {label}")
    
    # System readiness
    all_ready = all(status.values())
    readiness_icon = '🟢' if all_ready else '🔴'
    readiness_text = 'Ready' if all_ready else 'Not Ready'
    st.sidebar.write(f"**System: {readiness_icon} {readiness_text}**")
    
    if not all_ready:
        st.sidebar.warning("Some components are not ready. Check the setup instructions.")
    
    return status


def display_collection_stats(rag_interface: RAGInterface):
    """Display vector database statistics."""
    st.sidebar.header("Database Stats")
    
    collection = rag_interface.get_collection()
    if collection:
        try:
            count = collection.count()
            st.sidebar.metric("Total Chunks", count)
            
            # Get sample for analysis
            if count > 0:
                sample = collection.get(limit=min(20, count), include=['metadatas'])
                if sample.get('metadatas'):
                    files = set()
                    for metadata in sample['metadatas']:
                        if metadata and 'file_name' in metadata:
                            files.add(metadata['file_name'])
                    st.sidebar.metric("Source Files", len(files))
        except Exception as e:
            st.sidebar.error(f"Error getting stats: {e}")


def main_query_interface(rag_interface: RAGInterface):
    """Main query interface."""
    st.markdown('<div class="main-header"><h1>🧸 RAGgedy - Your Local AI Assistant</h1></div>', 
                unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Ask a question about your documents:",
        height=100,
        placeholder="Enter your question here..."
    )
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Number of chunks to retrieve", 1, 50, 5)
    with col2:
        # Get available models from Ollama
        available_models = []
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                available_models = [line.split()[0] for line in lines if line.strip()]
            
            # If no models found, show default options
            if not available_models:
                available_models = [DEFAULT_MODEL, "tinyllama", "gemma2:2b", "llama2:7b", "mistral:7b"]
                st.warning("⚠️ No models detected. Install models with: `ollama pull <model-name>`")
        except:
            available_models = [DEFAULT_MODEL, "tinyllama", "gemma2:2b", "llama2:7b", "mistral:7b"]
            st.warning("⚠️ Could not connect to Ollama. Make sure Ollama is running.")
        
        # Ensure default model is in the list
        if DEFAULT_MODEL not in available_models:
            available_models.insert(0, DEFAULT_MODEL)
        
        model_option = st.selectbox(
            "Model",
            available_models,
            help="Select from your installed Ollama models"
        )
    with col3:
        show_context = st.checkbox("Show retrieved context", value=True)
    
    # System prompt customization
    with st.expander("🎯 Customize System Prompt", expanded=False):
        st.write("**Default prompt:**")
        default_prompt = """You are a helpful AI assistant. Use only the provided context to answer the question. 
If the context doesn't contain enough information, say so clearly.
Be concise and accurate in your response."""
        st.code(default_prompt, language="text")
        
        custom_prompt = st.text_area(
            "**Custom system prompt** (leave empty to use default):",
            height=150,
            placeholder="Enter your custom system prompt here...",
            help="This prompt will guide how the AI responds to your questions. Use {context} and {question} as placeholders if needed."
        )
        
        if custom_prompt:
            st.info("Using custom system prompt")
        else:
            st.info("Using default system prompt")
    
    # Query processing
    if st.button("Get Answer", type="primary") and query.strip():
        with st.spinner("Processing your question..."):
            # Retrieve context
            with st.status("Retrieving relevant documents...") as status:
                result = rag_interface.get_top_chunks(query, top_k=top_k)
                
                if result and result.get('documents') and result['documents'][0]:
                    contexts = result['documents'][0]
                    distances = result.get('distances', [[]])[0]
                    metadatas = result.get('metadatas', [[]])[0]
                    
                    status.update(label=f"Found {len(contexts)} relevant chunks", state="complete")
                else:
                    contexts = []
                    st.warning("No relevant documents found.")
            
            if contexts:
                # Generate answer
                with st.status("Generating answer...") as status:
                    full_prompt = rag_interface.build_prompt(query, contexts, custom_prompt if custom_prompt.strip() else None)
                    answer = rag_interface.ollama_infer(full_prompt, model_option)
                    status.update(label="Answer generated", state="complete")
                
                # Display answer
                st.markdown("### 🤖 Answer")
                st.markdown(answer)
                
                # Display context if requested
                if show_context and contexts:
                    st.markdown("### 📚 Retrieved Context")
                    
                    for i, (context, distance, metadata) in enumerate(zip(contexts, distances, metadatas)):
                        similarity = 1 - distance if distance is not None else 0
                        
                        with st.expander(f"Chunk {i+1} - Similarity: {similarity:.3f}"):
                            st.markdown(f"**Source:** {metadata.get('file_name', 'Unknown') if metadata else 'Unknown'}")
                            st.markdown("**Content:**")
                            st.markdown(f'<div class="context-box">{context}</div>', 
                                      unsafe_allow_html=True)
                
                # Analysis section
                if len(contexts) > 1:
                    st.markdown("### 📊 Retrieval Analysis")
                    
                    # Similarity scores chart
                    similarities = [1 - d for d in distances] if distances else []
                    if similarities:
                        fig = px.bar(
                            x=[f"Chunk {i+1}" for i in range(len(similarities))],
                            y=similarities,
                            title="Document Chunk Similarities",
                            labels={'x': 'Chunks', 'y': 'Similarity Score'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)


def document_management_tab():
    """Document management interface."""
    st.header("📁 Document Management")
    
    # Upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        docs_dir = Path("docs")
        docs_dir.mkdir(exist_ok=True)
        
        for file in uploaded_files:
            file_path = docs_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"Saved: {file.name}")
        
        st.info("After uploading files, rebuild the index using: `python build_index.py rebuild docs/`")
    
    # Existing documents
    st.subheader("Existing Documents")
    docs_dir = Path("docs")
    
    if docs_dir.exists():
        doc_files = list(docs_dir.glob("*"))
        if doc_files:
            for file_path in doc_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {file_path.name}")
                with col2:
                    if st.button(f"Delete", key=f"del_{file_path.name}"):
                        file_path.unlink()
                        st.rerun()
        else:
            st.info("No documents found. Upload some documents to get started.")
    else:
        st.info("Documents directory not found.")


def system_management_tab():
    """System management interface."""
    st.header("⚙️ System Management")
    
    # Index management
    st.subheader("Vector Index Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Rebuild Index"):
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_container = st.container()
            
            try:
                # Start the rebuild process
                process = subprocess.Popen(
                    ["python3", "build_index.py", "rebuild", "docs/"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                output_lines = []
                progress_value = 0
                
                # Read output line by line
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.strip()
                        output_lines.append(line)
                        
                        # Update progress based on output
                        if "Loading embedding model" in line:
                            progress_value = 0.1
                            status_text.text("Loading embedding model...")
                        elif "Creating new collection" in line or "Using existing collection" in line:
                            progress_value = 0.2
                            status_text.text("Setting up collection...")
                        elif "Processing batch" in line:
                            # Extract batch info for progress
                            if "/" in line:
                                try:
                                    batch_info = line.split("batch ")[1]
                                    current, total = map(int, batch_info.split("/"))
                                    progress_value = 0.3 + (current / total) * 0.6
                                    status_text.text(f"Processing batch {current}/{total}...")
                                except:
                                    pass
                        elif "Successfully added" in line:
                            progress_value = 0.95
                            status_text.text("Finalizing index...")
                        elif "Index rebuilt successfully" in line:
                            progress_value = 1.0
                            status_text.text("Index rebuilt successfully!")
                        
                        progress_bar.progress(progress_value)
                        
                        # Show recent log lines
                        with log_container:
                            if len(output_lines) > 10:
                                st.text("\n".join(output_lines[-10:]))
                            else:
                                st.text("\n".join(output_lines))
                
                # Wait for process to complete
                return_code = process.wait()
                
                if return_code == 0:
                    st.success("Index rebuilt successfully!")
                    progress_bar.progress(1.0)
                else:
                    st.error(f"Index rebuild failed with code {return_code}")
                    
            except Exception as e:
                st.error(f"Error: {e}")
                status_text.text("Error occurred during rebuild")
    
    with col2:
        # Clear index with confirmation
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False
            
        if not st.session_state.confirm_clear:
            if st.button("Clear Index", type="secondary"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("⚠️ This will permanently delete the vector index!")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Confirm Clear", type="primary"):
                    try:
                        result = subprocess.run(
                            ["python3", "build_index.py", "clear"],
                            input="y\n",
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            st.success("Index cleared successfully!")
                        else:
                            st.error("Error clearing index:")
                            st.text(result.stderr)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        st.session_state.confirm_clear = False
                        st.rerun()
                        
            with col_b:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear = False
                    st.rerun()
    
    # Model management
    st.subheader("Model Management")
    
    # Get detailed model information
    available_models = []
    model_details = {}
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:  # Has header
                for line in lines[1:]:  # Skip header
                    parts = line.split()
                    if len(parts) >= 3:
                        model_name = parts[0]
                        model_id = parts[1] if len(parts) > 1 else ""
                        model_size = parts[2] if len(parts) > 2 else ""
                        modified = " ".join(parts[3:]) if len(parts) > 3 else ""
                        available_models.append(model_name)
                        model_details[model_name] = {
                            'id': model_id,
                            'size': model_size,
                            'modified': modified
                        }
    except:
        pass
    
    if available_models:
        st.write("**Installed Models:**")
        for model in available_models:
            details = model_details.get(model, {})
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if model == DEFAULT_MODEL:
                    st.write(f"🌟 **{model}** (default)")
                else:
                    st.write(f"📦 **{model}**")
            with col2:
                st.write(f"Size: {details.get('size', 'Unknown')}")
            with col3:
                st.write(f"Modified: {details.get('modified', 'Unknown')}")
    else:
        st.warning("No Ollama models found.")
    
    # Quick model installation
    st.write("**Quick Install Popular Models:**")
    col1, col2, col3, col4 = st.columns(4)
    
    recommended_models = [
        ("phi3:mini", "Recommended (4GB)"),
        ("tinyllama", "Ultra-small (1GB)"),
        ("gemma2:2b", "Efficient (3GB)"),
        ("llama3.2:3b", "Capable (2GB)")
    ]
    
    for i, (model_name, description) in enumerate(recommended_models):
        with [col1, col2, col3, col4][i]:
            if st.button(f"Install {model_name}", key=f"install_{model_name}"):
                with st.spinner(f"Installing {model_name}..."):
                    try:
                        # Show installation command to user
                        st.code(f"ollama pull {model_name}")
                        st.info(f"Run this command in your terminal to install {model_name}")
                    except Exception as e:
                        st.error(f"Note: {e}")
            st.caption(description)


def main():
    """Main application entry point."""
    setup_streamlit_page()
    
    # Initialize RAG interface
    rag_interface = RAGInterface()
    
    # Sidebar
    status = display_system_status()
    display_collection_stats(rag_interface)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🧸 Chat", "📁 Documents", "⚙️ System"])
    
    with tab1:
        if status['vector_db'] and status['model']:
            main_query_interface(rag_interface)
        else:
            st.warning("⚠️ System not ready. Please ensure all components are set up:")
            if not status['ollama']:
                st.error("- Ollama server is not running")
            if not status['model']:
                st.error(f"- Model {DEFAULT_MODEL} is not available")
            if not status['vector_db']:
                st.error("- Vector database is empty or not accessible")
    
    with tab2:
        document_management_tab()
    
    with tab3:
        system_management_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🧸 **RAGgedy** - Powered by [Ollama](https://ollama.com/) + "
        "[ChromaDB](https://www.trychroma.com/) + "
        "[SentenceTransformers](https://www.sbert.net/) + "
        "[Streamlit](https://streamlit.io/)"
    )


if __name__ == "__main__":
    main()