# 🧸 RAGgedy - Your Local AI Assistant

*"Here's to the crazy ones. The misfits. The rebels. The troublemakers. The round pegs in the square holes... Because they change things."*

RAGgedy is for the makers, the tinkerers, and the curious minds who want to chat with their documents without sending data to the cloud. It's a fully local RAG (Retrieval-Augmented Generation) system that runs entirely on your machine.

## ✨ What Makes RAGgedy Special

- 🏠 **Completely Local** - Your documents never leave your computer
- 🧸 **Friendly Interface** - Clean, intuitive web interface built with Streamlit
- 🎯 **Customizable Personality** - Edit the AI's system prompt to match your style
- 📚 **Smart Document Processing** - Handles PDFs with robust error handling
- 🚀 **Real-time Progress** - See exactly what's happening during indexing
- 🔍 **Powerful Search** - Retrieve up to 50 document chunks for comprehensive answers

## 🚀 Quick Start

### 1. Install the Basics
```bash
# Install Ollama (your local LLM server)
brew install ollama  # macOS
# or visit https://ollama.com for other platforms

# Download a language model (this will take a few minutes)
ollama pull gpt-oss:20b

# Install Python dependencies
pip install chromadb pdfplumber sentence-transformers streamlit
```

### 2. Set Up RAGgedy
```bash
# Clone or download this project
git clone [your-repo-url]
cd RAGgedy

# Add your documents to the docs folder
mkdir -p docs
# Copy your PDF or TXT files to docs/

# Build the search index
python build_index.py rebuild docs/

# Launch RAGgedy
streamlit run app.py
```

### 3. Start Chatting!
Open your browser to `http://localhost:8501` and start asking questions about your documents!

## 🎯 Features

### Smart Document Chat
Ask questions about your documents and get answers with source citations. RAGgedy will find relevant passages and use them to provide informed responses.

### Customizable AI Personality
Don't like how RAGgedy responds? Click "🎯 Customize System Prompt" and change how it talks:
- Make it more formal or casual
- Add domain expertise
- Change the response style
- Set specific constraints

### Real-time Indexing
Watch your documents get processed in real-time with progress bars and status updates. No more wondering if it's working!

### Multiple Models
Choose from different AI models based on your needs:
- `gpt-oss:20b` - Most capable (default)
- `llama2:7b` - Faster responses
- `mistral:7b` - Good balance

## 📁 Project Structure

```
RAGgedy/
├── 🧸 app.py                 # Main web interface
├── 📄 extract_and_chunk.py   # Document processing magic
├── 🗃️ build_index.py         # Vector database management
├── 📚 docs/                  # Put your documents here
├── 🔍 db/                    # Search index (auto-created)
└── 📖 README.md              # You are here
```

## 🛠️ Advanced Usage

### Command Line Tools

```bash
# Process documents and build index
python build_index.py rebuild docs/

# Query from command line
python build_index.py query "What is machine learning?"

# Check system stats
python build_index.py stats

# Clear the index
python build_index.py clear
```

### Customization

Want to tweak how documents are processed? Edit the settings in the Python files:

```python
# Chunk size (characters per section)
chunk_size = 800

# Overlap between chunks
chunk_overlap = 150

# Number of results to retrieve
top_k = 5
```

## 🔒 Privacy First

RAGgedy was built with privacy in mind:
- ✅ Everything runs locally on your machine
- ✅ No data sent to external services
- ✅ No API keys required
- ✅ Your documents stay private

## 🐛 Troubleshooting

### "Model not found" error
```bash
ollama pull gpt-oss:20b
```

### "No vector database" error
```bash
python build_index.py rebuild docs/
```

### System running slow?
- Try a smaller model: `llama2:7b`
- Reduce chunk retrieval count in the interface
- Make sure you have enough RAM

## 🤝 Contributing

RAGgedy is for the makers and tinkerers. If you've got ideas, improvements, or just want to make it better, contributions are welcome!

## 💝 Acknowledgments

Built with love using:
- [Ollama](https://ollama.com/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web interface
- [SentenceTransformers](https://www.sbert.net/) - Text embeddings

---

*RAGgedy: Because sometimes you need an AI that's as curious about your documents as you are.* 🧸