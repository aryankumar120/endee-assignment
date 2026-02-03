# Talk-Endee
**Talk-Endee is a compact RAG demo that uses Endee for vector search and Groq for answer generation, delivering grounded responses over local documents.**

![Build](https://img.shields.io/badge/build-manual-lightgrey)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)

## Table of Contents
- Overview
- Problem Statement
- System Design
- How Endee Is Used
- Tech Stack
- Setup
- Usage
- License

## Overview
Talk-Endee demonstrates a full Retrieval-Augmented Generation workflow: ingest documents, retrieve the best matches, and generate answers grounded in retrieved context.

## Problem Statement
LLMs alone can hallucinate and lack access to private data. This project addresses that by combining fast vector search with controlled generation so answers are grounded in your local documents.

## System Design
- Ingestion pipeline: chunking → embeddings → upsert to Endee
- Retrieval: query embedding → Endee search → metadata lookup
- Generation: Groq chat completions using retrieved context
- Output: ranked sources + final answer

## How Endee Is Used
- Endee stores vector embeddings for document chunks
- Search uses Endee similarity retrieval to return top-k matches
- Results are mapped back to local metadata for source attribution

## Tech Stack
- Endee (C++)
- Python 3.9+
- Groq API
- Requests, Sentence-Transformers, MessagePack

## Setup
1) Start Endee server in a new terminal:
```
cd /Users/aryankumar/Downloads/endee
export NDD_DATA_DIR=$(pwd)/data
./build/ndd-neon-darwin
```

2) Configure Groq:
- Add GROQ_API_KEY in talk-endee/.env

3) Install dependencies:
```
cd /Users/aryankumar/Downloads/endee/talk-endee
pip3 install -r requirements.txt
```

## Usage
Ingest sample documents:
```
python3 main.py ingest --directory data/sample_docs
```

Search example:
```
python3 main.py query "What is semantic search?" --search-only
```

Full RAG answer:
```
python3 main.py query "What is semantic search?"
```

## License
Apache-2.0. See LICENSE.
