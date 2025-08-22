# RAG System Deployment Guide
## Claude Flow Swarm Implementation

### System Overview
This RAG (Retrieval-Augmented Generation) system combines local document processing with the GPT-OSS-20B model via Ollama, ChromaDB vector storage, and a Streamlit web interface. The system is designed for fully local operation without requiring API keys or cloud services.

### Architecture Components
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │  Vector Index   │    │   Web UI        │
│   (PDF/TXT)     │───▶│   (ChromaDB)    │───▶│  (Streamlit)    │
│   docs/         │    │   db/           │    │   app.py        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Document        │    │ Embedding       │    │ LLM Generation  │
│ Processing      │    │ Generation      │    │ (Ollama)        │
│ extract_and_    │    │ SentenceTransf. │    │ gpt-oss:20b     │
│ chunk.py        │    │ all-MiniLM-L6-v2│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows with WSL
- **Memory**: Minimum 16GB RAM (recommended 32GB for optimal performance)
- **Storage**: At least 20GB free space for models and data
- **Python**: Version 3.8 or higher

### Core Dependencies
1. **Ollama**: Local LLM inference server
2. **Python Packages**: ChromaDB, SentenceTransformers, Streamlit, pdfplumber

## Installation Steps

### 1. Install Ollama
```bash
# macOS (using Homebrew)
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (WSL)
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Download the GPT-OSS-20B Model
```bash
# This will download approximately 13GB
ollama pull gpt-oss:20b

# Verify installation
ollama list
```

### 3. Install Python Dependencies
```bash
# Install required packages
pip install chromadb pdfplumber sentence-transformers streamlit

# Verify installation
python -c "import chromadb, sentence_transformers, streamlit, pdfplumber; print('All packages installed successfully')"
```

### 4. Verify Project Structure
Ensure your project directory has the following structure:
```
RAG/
├── app.py                    # Streamlit web interface
├── extract_and_chunk.py      # Document processing
├── build_index.py           # Vector index management
├── docs/                    # Document storage
│   └── sample_document.txt  # Sample document
├── db/                      # ChromaDB storage (created automatically)
└── DEPLOYMENT_GUIDE.md      # This file
```

## System Setup and Configuration

### 1. Process Documents and Build Index
```bash
# Process all documents in the docs/ folder and build vector index
python build_index.py build docs/

# Alternative: Rebuild index from scratch (clears existing data)
python build_index.py rebuild docs/
```

### 2. Start Ollama Server
```bash
# Start Ollama in the background
ollama serve

# Or test the model directly (in a separate terminal)
ollama run gpt-oss:20b
```

### 3. Launch Streamlit Interface
```bash
# Start the web interface
streamlit run app.py

# The interface will be available at: http://localhost:8501
```

## Usage Guide

### Document Management
1. **Upload Documents**: Place PDF or TXT files in the `docs/` directory
2. **Rebuild Index**: Run `python build_index.py rebuild docs/` after adding new documents
3. **Query System**: Use the web interface to ask questions about your documents

### Web Interface Features
- **Query Tab**: Ask questions and get AI-generated answers with source citations
- **Documents Tab**: Upload and manage document files
- **System Tab**: Monitor system status and manage the vector index

### Command Line Tools

#### Document Processing
```bash
# Process a specific directory
python extract_and_chunk.py docs/

# Process with custom chunk settings (modify the script as needed)
python extract_and_chunk.py docs/
```

#### Index Management
```bash
# Build index from documents
python build_index.py build docs/

# Query the index directly
python build_index.py query "What is machine learning?"

# View collection statistics
python build_index.py stats

# Clear the entire index
python build_index.py clear
```

## Configuration Options

### Chunk Size Optimization
Modify the `DocumentProcessor` parameters in your scripts:
```python
# In extract_and_chunk.py and build_index.py
processor = DocumentProcessor(
    chunk_size=800,      # Characters per chunk
    chunk_overlap=150    # Overlap between chunks
)
```

### Embedding Model Selection
Change the embedding model in `build_index.py` and `app.py`:
```python
# Options: all-MiniLM-L6-v2 (default), all-mpnet-base-v2, multi-qa-MiniLM-L6-cos-v1
EMBED_MODEL = "all-MiniLM-L6-v2"
```

### LLM Model Selection
Modify the model in `app.py`:
```python
# Available models: gpt-oss:20b (default), llama2:7b, mistral:7b
DEFAULT_MODEL = "gpt-oss:20b"
```

## Troubleshooting

### Common Issues

#### 1. Ollama Model Not Found
```bash
# Symptom: "Model not found" error
# Solution: Download the model
ollama pull gpt-oss:20b
```

#### 2. ChromaDB Connection Error
```bash
# Symptom: "Could not connect to vector database"
# Solution: Rebuild the index
python build_index.py rebuild docs/
```

#### 3. Out of Memory Error
```bash
# Symptom: System runs out of memory during processing
# Solution: Reduce batch size in build_index.py
# Modify: batch_size=50 (default is 100)
```

#### 4. Slow Query Performance
- **Check System Resources**: Ensure sufficient RAM and CPU availability
- **Optimize Chunk Count**: Reduce `top_k` parameter in queries
- **Update Hardware**: Consider upgrading to faster storage (SSD)

### Performance Optimization

#### System Settings
```bash
# Increase file descriptor limits (Linux/macOS)
ulimit -n 4096

# Monitor system resources
top -p $(pgrep -f ollama)
```

#### Model Optimization
```python
# Use smaller models for faster responses (modify app.py)
DEFAULT_MODEL = "llama2:7b"  # Faster, less accurate
# or
DEFAULT_MODEL = "mistral:7b" # Good balance
```

## Security Considerations

### Data Privacy
- All processing is performed locally
- No data is sent to external services
- Documents remain on your local system

### Network Security
- Streamlit runs on localhost by default
- Ollama API is local-only by default
- No external network access required for operation

### File Security
```bash
# Set appropriate permissions for document directory
chmod 750 docs/
chmod 640 docs/*

# Secure the database directory
chmod 750 db/
```

## Monitoring and Maintenance

### Log Management
```bash
# Monitor Ollama logs
tail -f ~/.ollama/logs/server.log

# Monitor Streamlit logs
streamlit run app.py --logger.level debug
```

### Regular Maintenance
1. **Update Models**: Regularly check for model updates
2. **Clean Database**: Periodically rebuild the index for optimal performance
3. **Monitor Disk Usage**: Vector databases can grow large with many documents

### Backup Procedures
```bash
# Backup documents
tar -czf documents_backup.tar.gz docs/

# Backup vector database
tar -czf database_backup.tar.gz db/

# Restore procedures
tar -xzf documents_backup.tar.gz
tar -xzf database_backup.tar.gz
python build_index.py rebuild docs/
```

## Advanced Configuration

### Custom Preprocessing
Modify `extract_and_chunk.py` to add custom preprocessing:
```python
def custom_preprocess(self, text):
    # Add domain-specific preprocessing
    text = re.sub(r'specific_pattern', 'replacement', text)
    return text
```

### Multi-Language Support
```python
# Use multilingual embedding models
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
```

### Scaling for Large Document Sets
```python
# Increase batch sizes for bulk processing
batch_size = 500  # Process more documents at once
```

## API Integration

### REST API Wrapper (Optional)
Create a simple API wrapper for programmatic access:
```python
# api.py (example implementation)
from flask import Flask, request, jsonify
from build_index import VectorIndexBuilder

app = Flask(__name__)
builder = VectorIndexBuilder()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    results = builder.query_index(data['query'], data.get('top_k', 5))
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
```

## Support and Resources

### Documentation
- [Ollama Documentation](https://ollama.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [SentenceTransformers Documentation](https://www.sbert.net/)

### Community Resources
- [Ollama GitHub](https://github.com/ollama/ollama)
- [ChromaDB GitHub](https://github.com/chroma-core/chroma)

### System Status Monitoring
The Streamlit interface provides real-time system status monitoring including:
- Ollama server connectivity
- Model availability
- Vector database status
- Document collection statistics

---

## Quick Start Summary

1. **Install Prerequisites**: `brew install ollama && pip install chromadb pdfplumber sentence-transformers streamlit`
2. **Download Model**: `ollama pull gpt-oss:20b`
3. **Build Index**: `python build_index.py build docs/`
4. **Start System**: `streamlit run app.py`
5. **Access Interface**: Navigate to `http://localhost:8501`

Your RAG system is now ready to answer questions about your documents!