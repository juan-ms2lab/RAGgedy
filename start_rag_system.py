#!/usr/bin/env python3
"""
RAG System Startup Script
DevOps Engineer Agent Implementation

Comprehensive startup script that initializes all components and
provides system health monitoring for the RAG system.
"""

import subprocess
import time
import os
import sys
from pathlib import Path
import requests
import json
from typing import Dict, List, Tuple, Optional


class RAGSystemManager:
    """Manages the complete RAG system startup and health monitoring."""
    
    def __init__(self):
        """Initialize system manager."""
        self.base_dir = Path(__file__).parent
        self.docs_dir = self.base_dir / "docs"
        self.db_dir = self.base_dir / "db"
        self.required_files = [
            "extract_and_chunk.py",
            "build_index.py", 
            "app.py",
            "optimize_performance.py"
        ]
        self.ollama_process = None
        self.streamlit_process = None
    
    def check_system_requirements(self) -> Dict[str, bool]:
        """
        Check if all system requirements are met.
        
        Returns:
            Dictionary with requirement status
        """
        requirements = {
            "python": False,
            "ollama": False,
            "required_files": False,
            "python_packages": False,
            "directories": False
        }
        
        # Check Python version
        try:
            version = sys.version_info
            requirements["python"] = version.major >= 3 and version.minor >= 8
        except:
            pass
        
        # Check Ollama installation
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            requirements["ollama"] = result.returncode == 0
        except:
            pass
        
        # Check required files
        missing_files = []
        for file in self.required_files:
            if not (self.base_dir / file).exists():
                missing_files.append(file)
        requirements["required_files"] = len(missing_files) == 0
        
        # Check Python packages
        required_packages = [
            "chromadb", "sentence_transformers", 
            "streamlit", "pdfplumber"
        ]
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        requirements["python_packages"] = len(missing_packages) == 0
        
        # Check directories
        requirements["directories"] = (
            self.docs_dir.exists() and 
            (self.db_dir.exists() or True)  # db_dir can be created
        )
        
        return requirements
    
    def check_ollama_model(self, model_name: str = "gpt-oss:20b") -> bool:
        """
        Check if the required Ollama model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available
        """
        try:
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True, timeout=10)
            return model_name in result.stdout if result.returncode == 0 else False
        except:
            return False
    
    def start_ollama_server(self) -> bool:
        """
        Start Ollama server if not already running.
        
        Returns:
            True if server is running
        """
        # Check if already running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is already running")
                return True
        except:
            pass
        
        print("🚀 Starting Ollama server...")
        try:
            # Start Ollama server in background
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to be ready
            for attempt in range(30):  # 30 second timeout
                try:
                    response = requests.get("http://localhost:11434/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("✅ Ollama server started successfully")
                        return True
                except:
                    time.sleep(1)
            
            print("❌ Failed to start Ollama server (timeout)")
            return False
            
        except Exception as e:
            print(f"❌ Error starting Ollama server: {e}")
            return False
    
    def check_vector_database(self) -> Dict[str, any]:
        """
        Check vector database status and statistics.
        
        Returns:
            Database status and statistics
        """
        try:
            from build_index import VectorIndexBuilder
            builder = VectorIndexBuilder()
            stats = builder.get_collection_stats()
            
            return {
                "exists": True,
                "stats": stats,
                "has_data": stats.get("total_chunks", 0) > 0
            }
        except Exception as e:
            return {
                "exists": False,
                "error": str(e),
                "has_data": False
            }
    
    def build_vector_index(self, rebuild: bool = False) -> bool:
        """
        Build or rebuild the vector index.
        
        Args:
            rebuild: Whether to rebuild from scratch
            
        Returns:
            True if successful
        """
        if not list(self.docs_dir.glob("*")):
            print("⚠️  No documents found in docs/ directory")
            return False
        
        print(f"🔨 {'Rebuilding' if rebuild else 'Building'} vector index...")
        try:
            command = ["python3", "build_index.py", 
                      "rebuild" if rebuild else "build", str(self.docs_dir)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Vector index built successfully")
                return True
            else:
                print(f"❌ Error building vector index: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Vector index build timed out")
            return False
        except Exception as e:
            print(f"❌ Error building vector index: {e}")
            return False
    
    def start_streamlit_app(self) -> bool:
        """
        Start the Streamlit web application.
        
        Returns:
            True if successful
        """
        print("🚀 Starting Streamlit web application...")
        try:
            self.streamlit_process = subprocess.Popen(
                ["streamlit", "run", "app.py", "--server.address", "localhost", 
                 "--server.port", "8501", "--server.headless", "true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for Streamlit to be ready
            for attempt in range(20):  # 20 second timeout
                try:
                    response = requests.get("http://localhost:8501", timeout=2)
                    if response.status_code == 200:
                        print("✅ Streamlit app started successfully")
                        print("🌐 Access the RAG system at: http://localhost:8501")
                        return True
                except:
                    time.sleep(1)
            
            print("⚠️  Streamlit started but may still be loading...")
            print("🌐 Try accessing: http://localhost:8501")
            return True
            
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return False
    
    def print_system_status(self) -> None:
        """Print comprehensive system status."""
        print("\n" + "="*60)
        print("RAG SYSTEM STATUS")
        print("="*60)
        
        # Check requirements
        requirements = self.check_system_requirements()
        print("\n🔍 System Requirements:")
        for req, status in requirements.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {req.replace('_', ' ').title()}")
        
        # Check Ollama model
        model_available = self.check_ollama_model()
        icon = "✅" if model_available else "❌"
        print(f"  {icon} GPT-OSS-20B Model")
        
        # Check database
        db_status = self.check_vector_database()
        icon = "✅" if db_status["has_data"] else "⚠️"
        print(f"  {icon} Vector Database")
        if db_status.get("stats"):
            stats = db_status["stats"]
            print(f"    - Chunks: {stats.get('total_chunks', 0)}")
            print(f"    - Files: {stats.get('unique_source_files', 0)}")
        
        # Check services
        print("\n🌐 Running Services:")
        
        # Ollama service
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            ollama_running = response.status_code == 200
        except:
            ollama_running = False
        icon = "✅" if ollama_running else "❌"
        print(f"  {icon} Ollama Server (http://localhost:11434)")
        
        # Streamlit service
        try:
            response = requests.get("http://localhost:8501", timeout=2)
            streamlit_running = response.status_code == 200
        except:
            streamlit_running = False
        icon = "✅" if streamlit_running else "❌"
        print(f"  {icon} Streamlit App (http://localhost:8501)")
        
        print("\n" + "="*60)
        
        # Overall system status
        all_good = (
            all(requirements.values()) and 
            model_available and 
            db_status["has_data"] and
            ollama_running
        )
        
        if all_good:
            print("🎉 RAG System is fully operational!")
            print("🌐 Access your RAG system at: http://localhost:8501")
        else:
            print("⚠️  Some components need attention. See details above.")
    
    def shutdown(self) -> None:
        """Shutdown all managed processes."""
        print("\n🛑 Shutting down RAG system...")
        
        if self.streamlit_process:
            try:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
                print("✅ Streamlit app stopped")
            except:
                try:
                    self.streamlit_process.kill()
                except:
                    pass
        
        if self.ollama_process:
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=5)
                print("✅ Ollama server stopped")
            except:
                try:
                    self.ollama_process.kill()
                except:
                    pass
        
        print("👋 RAG system shutdown complete")
    
    def full_startup(self) -> bool:
        """
        Perform complete system startup sequence.
        
        Returns:
            True if all components started successfully
        """
        print("🚀 RAG System Initialization")
        print("="*50)
        
        # Check requirements
        requirements = self.check_system_requirements()
        if not all(requirements.values()):
            print("❌ System requirements not met:")
            for req, status in requirements.items():
                if not status:
                    print(f"  - Missing: {req.replace('_', ' ')}")
            return False
        
        # Check model
        if not self.check_ollama_model():
            print("❌ GPT-OSS-20B model not found")
            print("   Run: ollama pull gpt-oss:20b")
            return False
        
        # Start Ollama server
        if not self.start_ollama_server():
            return False
        
        # Check/build vector database
        db_status = self.check_vector_database()
        if not db_status["has_data"]:
            print("📚 No vector index found, building from documents...")
            if not self.build_vector_index():
                return False
        else:
            print("✅ Vector database is ready")
        
        # Start Streamlit app
        if not self.start_streamlit_app():
            return False
        
        # Final status
        self.print_system_status()
        return True


def main():
    """Main CLI interface."""
    manager = RAGSystemManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            manager.print_system_status()
            
        elif command == "start":
            success = manager.full_startup()
            if success:
                try:
                    print("\nPress Ctrl+C to shutdown the system...")
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    manager.shutdown()
            else:
                sys.exit(1)
                
        elif command == "build-index":
            rebuild = "--rebuild" in sys.argv
            manager.build_vector_index(rebuild=rebuild)
            
        elif command == "optimize":
            print("🔧 Running performance optimization...")
            try:
                result = subprocess.run(["python3", "optimize_performance.py", "--quick"])
                if result.returncode == 0:
                    print("✅ Optimization complete")
                else:
                    print("❌ Optimization failed")
            except Exception as e:
                print(f"❌ Error running optimization: {e}")
                
        elif command == "help":
            print("""
RAG System Manager Commands:

  status          - Show system status
  start           - Start all components (interactive mode)
  build-index     - Build vector index from documents
  build-index --rebuild - Rebuild index from scratch
  optimize        - Run performance optimization
  help            - Show this help message

Examples:
  python3 start_rag_system.py status
  python3 start_rag_system.py start
  python3 start_rag_system.py build-index --rebuild
            """)
        else:
            print(f"Unknown command: {command}")
            print("Use 'help' for available commands")
    else:
        # Default: show status and offer to start
        manager.print_system_status()
        
        response = input("\nWould you like to start the RAG system? (y/N): ")
        if response.lower() in ['y', 'yes']:
            success = manager.full_startup()
            if success:
                try:
                    print("\nPress Ctrl+C to shutdown the system...")
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    manager.shutdown()


if __name__ == "__main__":
    main()