#!/usr/bin/env python3
"""
RAGgedy Startup Script

A simple launcher that bypasses Streamlit's email prompt and starts RAGgedy
in the most user-friendly way possible.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def setup_streamlit_config():
    """Set up Streamlit config to skip email prompt."""
    config_dir = Path.home() / ".streamlit"
    config_file = config_dir / "config.toml"
    
    if not config_file.exists():
        config_dir.mkdir(exist_ok=True)
        with open(config_file, "w") as f:
            f.write("[browser]\n")
            f.write("gatherUsageStats = false\n")
        print("📝 Created Streamlit config to skip email prompts")

def main():
    """Launch RAGgedy with optimal settings."""
    print("🧸 Starting RAGgedy - Your Local AI Assistant")
    print("=" * 50)
    
    # Check if Ollama is available
    if not check_ollama():
        print("⚠️  Warning: Ollama not found or not running")
        print("   Install Ollama from https://ollama.com")
        print("   Then run: ollama pull phi3:mini")
        print()
    
    # Setup Streamlit config
    setup_streamlit_config()
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: app.py not found")
        print("   Make sure you're in the RAGgedy directory")
        sys.exit(1)
    
    print("🚀 Launching RAGgedy...")
    print("   Local URL will be: http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        # Launch Streamlit with optimal settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "true",
            "--browser.serverAddress", "localhost",
            "--server.enableCORS", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 RAGgedy stopped. Thanks for using your local AI assistant!")
    except Exception as e:
        print(f"❌ Error starting RAGgedy: {e}")
        print("\nTry manually with:")
        print("streamlit run app.py --server.headless true")

if __name__ == "__main__":
    main()