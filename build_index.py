#!/usr/bin/env python3
"""
Vector Index Creation Component
Backend Developer Agent Implementation

This module builds and manages the ChromaDB vector database for the RAG system.
Handles embedding generation and efficient vector storage/retrieval.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from extract_and_chunk import DocumentProcessor


class VectorIndexBuilder:
    """Manages ChromaDB vector database for RAG system."""
    
    def __init__(
        self, 
        db_dir: str = "db",
        collection_name: str = "rag_docs",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector index builder.
        
        Args:
            db_dir: Directory for ChromaDB storage
            collection_name: Name of the document collection
            embedding_model: SentenceTransformers model for embeddings
        """
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Ensure db directory exists
        Path(self.db_dir).mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.db_dir)
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        print("Embedding model loaded successfully")
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            print(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document chunks with embeddings"}
            )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def add_chunks_to_index(self, chunks: List[Dict], batch_size: int = 100):
        """
        Add document chunks to the vector index.
        
        Args:
            chunks: List of chunk dictionaries from document processing
            batch_size: Number of chunks to process in each batch
        """
        if not chunks:
            print("No chunks to add to index")
            return
        
        print(f"Adding {len(chunks)} chunks to vector index...")
        
        # Process in batches to manage memory
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            # Prepare data for ChromaDB
            texts = [chunk['text'] for chunk in batch_chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Create unique IDs for chunks
            ids = [f"chunk_{chunk['source_file']}_{chunk['chunk_id']}" for chunk in batch_chunks]
            
            # Prepare metadata (ChromaDB requires all metadata values to be strings, numbers, or booleans)
            metadatas = []
            for chunk in batch_chunks:
                metadata = {
                    'source_file': chunk['source_file'],
                    'file_name': chunk['file_name'],
                    'file_type': chunk['file_type'],
                    'chunk_id': chunk['chunk_id'],
                    'start_pos': chunk['start_pos'],
                    'end_pos': chunk['end_pos'],
                    'total_chars': chunk['total_chars'],
                    'chunk_length': len(chunk['text'])
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        
        print(f"Successfully added {len(chunks)} chunks to the vector index")
    
    def query_index(
        self, 
        query_text: str, 
        n_results: int = 5,
        include_metadata: bool = True
    ) -> Dict:
        """
        Query the vector index for similar chunks.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            include_metadata: Whether to include chunk metadata
            
        Returns:
            Query results from ChromaDB
        """
        query_embedding = self.generate_embeddings([query_text])
        
        include_params = ['documents', 'distances']
        if include_metadata:
            include_params.append('metadatas')
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=include_params
        )
        
        return results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Get ALL documents to properly count unique files
        all_results = self.collection.get(include=['metadatas', 'documents'])
        
        stats = {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name
        }
        
        if all_results['documents']:
            avg_chunk_length = sum(len(doc) for doc in all_results['documents']) / len(all_results['documents'])
            stats['avg_chunk_length'] = round(avg_chunk_length, 1)
            
            # Count unique source files from ALL documents
            if all_results.get('metadatas'):
                unique_files = set()
                for metadata in all_results['metadatas']:
                    if metadata and 'source_file' in metadata:
                        unique_files.add(metadata['source_file'])
                stats['unique_source_files'] = len(unique_files)
                stats['source_files'] = sorted(list(unique_files))  # Show actual file list
        
        return stats
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")
    
    def rebuild_index_from_documents(self, docs_dir: str):
        """
        Rebuild the entire index from documents directory.
        
        Args:
            docs_dir: Directory containing documents to process
        """
        print("Rebuilding vector index from documents...")
        print(f"Processing documents from: {docs_dir}")
        
        # Clear existing collection
        print("Clearing existing collection...")
        self.clear_collection()
        
        # Process documents
        print("Processing documents and creating chunks...")
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
        chunks = processor.process_directory(docs_dir)
        
        print(f"Created {len(chunks)} chunks from documents")
        
        if chunks:
            # Add to index
            self.add_chunks_to_index(chunks)
            
            # Print statistics
            stats = self.get_collection_stats()
            print(f"\nIndex rebuilt successfully!")
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            print("No chunks created from documents")


def main():
    """Command line interface for vector index building."""
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <command> [args]")
        print("Commands:")
        print("  build <docs_dir>     - Build index from documents directory")
        print("  rebuild <docs_dir>   - Rebuild index from scratch")
        print("  query <query_text>   - Query the existing index")
        print("  stats                - Show collection statistics")
        print("  clear                - Clear the collection")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Initialize index builder
    builder = VectorIndexBuilder()
    
    if command == "build":
        if len(sys.argv) != 3:
            print("Usage: python build_index.py build <docs_dir>")
            sys.exit(1)
        
        docs_dir = sys.argv[2]
        
        # Process documents
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
        chunks = processor.process_directory(docs_dir)
        
        # Add to index
        builder.add_chunks_to_index(chunks)
        
        # Show stats
        stats = builder.get_collection_stats()
        print(f"\nBuild complete!")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    elif command == "rebuild":
        if len(sys.argv) != 3:
            print("Usage: python build_index.py rebuild <docs_dir>")
            sys.exit(1)
        
        docs_dir = sys.argv[2]
        builder.rebuild_index_from_documents(docs_dir)
    
    elif command == "query":
        if len(sys.argv) != 3:
            print("Usage: python build_index.py query '<query_text>'")
            sys.exit(1)
        
        query_text = sys.argv[2]
        results = builder.query_index(query_text, n_results=3)
        
        print(f"\nQuery: {query_text}")
        print(f"Found {len(results['documents'][0])} results:")
        
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"\n--- Result {i+1} (similarity: {1-distance:.3f}) ---")
            print(doc[:200] + "..." if len(doc) > 200 else doc)
            
            if results.get('metadatas') and results['metadatas'][0][i]:
                metadata = results['metadatas'][0][i]
                print(f"Source: {metadata.get('file_name', 'Unknown')}")
    
    elif command == "stats":
        stats = builder.get_collection_stats()
        print("Collection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    elif command == "clear":
        confirm = input("Are you sure you want to clear the collection? (y/N): ")
        if confirm.lower() == 'y':
            builder.clear_collection()
        else:
            print("Operation cancelled")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()