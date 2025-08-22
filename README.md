# 🧸 RAGgedy - Your Local AI Assistant

*"Here's to the crazy ones. The misfits. The rebels. The troublemakers. The round pegs in the square holes... Because they change things."*

RAGgedy is for the makers, the tinkerers, and the curious minds who want to chat with their documents without sending data to the cloud. It's a fully local RAG (Retrieval-Augmented Generation) system that runs entirely on your machine.

**Created by [Juan Santos](https://github.com/juan-ms2lab) at [Imagiro](https://github.com/imagiro)**

## ✨ What Makes RAGgedy Special

- 🏠 **Completely Local** - Your documents never leave your computer
- 🧸 **Friendly Interface** - Clean, intuitive web interface built with Streamlit
- 🎯 **Customizable Personality** - Edit the AI's system prompt to match your style
- 📚 **Smart Document Processing** - Handles PDFs with robust error handling
- 🚀 **Real-time Progress** - See exactly what's happening during indexing
- 🔍 **Powerful Search** - Retrieve up to 50 document chunks for comprehensive answers

## 💻 System Requirements

- **RAM**: 8GB minimum (4GB available for the AI model)
- **Storage**: 10GB free space 
- **OS**: macOS, Linux, or Windows
- **Python**: 3.8 or higher

*RAGgedy is designed to run on typical laptops and desktops - no GPU required!*

## 🚀 Quick Start

### 1. Install the Basics
```bash
# Install Ollama (your local LLM server)
brew install ollama  # macOS
# or visit https://ollama.com for other platforms

# Download a language model (this will take a few minutes)
ollama pull phi3:mini
```

### 2. Install RAGgedy
Choose one of these installation methods:

#### Option A: Install from GitHub (Recommended)
```bash
# Clone the repository
git clone https://github.com/juan-ms2lab/RAGgedy.git
cd RAGgedy

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

#### Option B: Install with pip (when available)
```bash
pip install raggedy
```

### 3. Set Up Your Documents
```bash
# Add your documents to the docs folder  
mkdir -p docs
# Copy your PDF or TXT files to docs/

# Build the search index
python build_index.py rebuild docs/

# Launch RAGgedy
streamlit run app.py
```

### 4. Start Chatting!
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
- `phi3:mini` - Best balance of performance & size (default, ~4GB)
- `tinyllama` - Ultra-fast, minimal resources (~2GB) 
- `gemma2:2b` - Google's efficient model (~3GB)
- `llama2:7b` - More capable but larger (~4.8GB)
- `gpt-oss:20b` - Most capable but requires 16GB+ RAM

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

If you installed RAGgedy as a package, you can use these convenient commands:

```bash
# Process documents and build index
raggedy-build-index rebuild docs/

# Query from command line  
raggedy-build-index query "What is machine learning?"

# Check system stats
raggedy-build-index stats

# Clear the index
raggedy-build-index clear

# Extract and process documents
raggedy-extract docs/
```

Or use the Python scripts directly:
```bash
python build_index.py rebuild docs/
python build_index.py query "What is machine learning?"
python build_index.py stats
python build_index.py clear
python extract_and_chunk.py docs/
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
ollama pull phi3:mini
```

### "No vector database" error
```bash
python build_index.py rebuild docs/
```

### System running slow?
- Try an even smaller model: `tinyllama` or `gemma2:2b`
- Reduce chunk retrieval count in the interface
- Make sure you have at least 8GB RAM available

## 🤝 Contributing

RAGgedy is for the makers and tinkerers. If you've got ideas, improvements, or just want to make it better, contributions are welcome!

## 📄 License

RAGgedy is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- ✅ Share and redistribute for non-commercial purposes
- ✅ Adapt, remix, and build upon the material for non-commercial purposes
- ✅ Use for personal projects, education, and research

**Attribution required:** Please credit Juan Santos (Imagiro) and link to this repository.

For commercial licensing, please contact [juan@imagiro.com](mailto:juan@imagiro.com).

## 💝 Acknowledgments

Built with love using:
- [Ollama](https://ollama.com/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web interface
- [SentenceTransformers](https://www.sbert.net/) - Text embeddings

**Created with assistance from [Claude AI](https://claude.ai/) by Anthropic**

---

*RAGgedy: Because sometimes you need an AI that's as curious about your documents as you are.* 🧸