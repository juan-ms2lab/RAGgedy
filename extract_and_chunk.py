#!/usr/bin/env python3
"""
Document Processing and Text Chunking Component
Backend Developer Agent Implementation

This module handles document extraction and chunking for the RAG system.
Supports PDF and TXT files with configurable chunking strategies.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Generator
import pdfplumber
import re


class DocumentProcessor:
    """Handles document extraction and text chunking for RAG pipeline."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file using pdfplumber with robust error handling.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_content = []
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    try:
                        # Try to extract text from the page
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                        print(f"Processed page {i+1}/{total_pages} from {Path(pdf_path).name}")
                    except Exception as page_error:
                        print(f"Warning: Could not extract text from page {i+1} of {Path(pdf_path).name}: {page_error}")
                        # Try alternative extraction method for problematic pages
                        try:
                            # Extract text with simpler method that ignores graphics
                            simple_text = page.within_bbox((0, 0, page.width, page.height)).extract_text(
                                x_tolerance=3, y_tolerance=3
                            )
                            if simple_text:
                                text_content.append(simple_text)
                                print(f"Recovered text from page {i+1} using fallback method")
                        except Exception:
                            print(f"Could not recover text from page {i+1}, skipping...")
                            continue
                
                if text_content:
                    return "\n\n".join(text_content)
                else:
                    print(f"No text could be extracted from {pdf_path}")
                    return ""
                    
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}")
            # Try with a more basic approach using PyPDF2 as fallback
            try:
                import PyPDF2
                print(f"Attempting fallback extraction for {pdf_path}")
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text_content = []
                    for page in reader.pages:
                        try:
                            text_content.append(page.extract_text())
                        except:
                            continue
                    return "\n\n".join(text_content) if text_content else ""
            except ImportError:
                print("PyPDF2 not available for fallback extraction")
            except Exception as fallback_error:
                print(f"Fallback extraction also failed: {fallback_error}")
            
            return ""
    
    def extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text from TXT file.
        
        Args:
            txt_path: Path to TXT file
            
        Returns:
            File text content
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {txt_path}: {e}")
            return ""
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from file based on extension.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Extracted text content
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.txt':
            return self.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file type: {file_ext}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers patterns (basic cleanup)
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
        
        return text.strip()
    
    def chunk_text(self, text: str, metadata: dict = None) -> List[dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position for current chunk
            end = start + self.chunk_size
            
            # If not at end of text, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within reasonable distance
                sentence_end = text.rfind('.', start, end + 100)
                if sentence_end != -1 and sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Fallback to word boundary
                    word_boundary = text.rfind(' ', start, end)
                    if word_boundary != -1:
                        end = word_boundary
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_id': len(chunks)
                }
                
                # Add metadata if provided
                if metadata:
                    chunk_data.update(metadata)
                
                chunks.append(chunk_data)
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def process_document(self, file_path: str) -> List[dict]:
        """
        Process a single document: extract text and create chunks.
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of text chunks with metadata
        """
        print(f"Processing: {file_path}")
        
        # Extract text
        raw_text = self.extract_text_from_file(file_path)
        if not raw_text:
            print(f"No text extracted from {file_path}")
            return []
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Create metadata
        metadata = {
            'source_file': file_path,
            'file_name': Path(file_path).name,
            'file_type': Path(file_path).suffix.lower(),
            'total_chars': len(cleaned_text)
        }
        
        # Create chunks
        chunks = self.chunk_text(cleaned_text, metadata)
        
        print(f"Created {len(chunks)} chunks from {file_path}")
        return chunks
    
    def process_directory(self, docs_dir: str) -> List[dict]:
        """
        Process all supported documents in a directory.
        
        Args:
            docs_dir: Directory containing documents
            
        Returns:
            List of all text chunks from all documents
        """
        supported_extensions = {'.pdf', '.txt'}
        all_chunks = []
        
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            print(f"Directory not found: {docs_dir}")
            return []
        
        # Find all supported files
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(docs_path.glob(f"*{ext}"))
        
        if not doc_files:
            print(f"No supported documents found in {docs_dir}")
            return []
        
        print(f"Found {len(doc_files)} documents to process")
        
        # Process each file
        for file_path in doc_files:
            chunks = self.process_document(str(file_path))
            all_chunks.extend(chunks)
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


def main():
    """Command line interface for document processing."""
    if len(sys.argv) != 2:
        print("Usage: python extract_and_chunk.py <docs_directory>")
        sys.exit(1)
    
    docs_dir = sys.argv[1]
    
    # Initialize processor with optimized settings for RAG
    processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
    
    # Process all documents
    chunks = processor.process_directory(docs_dir)
    
    if chunks:
        print(f"\nProcessing complete!")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(len(chunk['text']) for chunk in chunks) / len(chunks):.0f} characters")
    else:
        print("No chunks created. Please check your documents directory.")


if __name__ == "__main__":
    main()