"""
RAGgedy - Your Local AI Assistant

A fully local RAG (Retrieval-Augmented Generation) system for chatting with your documents
without sending data to the cloud. Built with Ollama, ChromaDB, and Streamlit.

Created by Juan Santos at Imagiro
License: Creative Commons Attribution-NonCommercial 4.0 International
"""

__version__ = "1.0.0"
__author__ = "Juan Santos"
__email__ = "juan@imagiro.com"
__license__ = "CC BY-NC 4.0"
__url__ = "https://github.com/juan-ms2lab/RAGgedy"

from .app import RAGInterface
from .build_index import VectorIndexBuilder  
from .extract_and_chunk import DocumentProcessor

__all__ = [
    "RAGInterface",
    "VectorIndexBuilder", 
    "DocumentProcessor",
]