#!/usr/bin/env python3
"""
RAGgedy Setup Script
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="raggedy",
    version="1.0.0",
    author="Juan Santos",
    author_email="juan@imagiro.com",
    description="🧸 Your Local AI Assistant - Chat with your documents privately",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/juan-ms2lab/RAGgedy",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Free for non-commercial use",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "raggedy=app:main",
            "raggedy-build-index=build_index:main",
            "raggedy-extract=extract_and_chunk:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "LICENSE"],
    },
    keywords="ai, rag, local-ai, ollama, streamlit, document-chat, privacy, llm",
    project_urls={
        "Bug Reports": "https://github.com/juan-ms2lab/RAGgedy/issues",
        "Source": "https://github.com/juan-ms2lab/RAGgedy",
        "Documentation": "https://github.com/juan-ms2lab/RAGgedy#readme",
    },
)