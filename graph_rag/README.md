# graph_rag: Retrieval-Augmented Generation for DEDuCTv3.0

This module provides a retrieval-augmented chatbot and CLI for the DEDuCTv3.0 knowledge graph and supporting data.

## Features

- Parses and indexes:
  - DEDuCTv3.0 knowledge graph (nodes/edges)
  - CGPD tetramers
  - Chemical similarity and target similarity tables
  - All project README files as reference documents
- Provides retrieval-augmented answers using GroqAPI LLM (or fallback evidence summary)
- Streamlit and CLI interfaces

## Usage

### 1. Build the index

```bash
python graph_rag/build_graph_index.py --data-root . --output-dir artifacts
```

### 2. Run the Streamlit chatbot

```bash
streamlit run graph_rag/chat_app.py
```

### 3. Run the CLI chatbot

```bash
python graph_rag/chat_cli.py
```

## Requirements

- Python 3.9+
- See requirements.txt for dependencies

## Environment

- Place your GroqAPI key in a `.env` file as `GROQ_API_KEY=...`

## Authors

- Adapted for DEDuCTv3.0 by [your name here]
