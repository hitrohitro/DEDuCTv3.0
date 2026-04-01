# DEDuCT-GraphRAG: Conversational AI for Endocrine Disruptors

## Overview

This repository is an independent, proof-of-concept Retrieval-Augmented Generation (RAG) architecture. The software pipeline and GraphRAG chatbot were custom-built by Rohit B K, but the underlying toxicology data and graph structure were developed entirely by the Computational Biology Group at IMSc. The project demonstrates advanced conversational AI capabilities for exploring the DEDuCTv3.0 knowledge graph and its supporting data.

## Key Features

- **Knowledge Graph (KG):** Contains 7 primary node types and 31 edge types mapping Chemical-Gene-Disease-Phenotype interactions.
- **Supporting Data:** Includes CGPD tetramers, Chemical Similarity (Tanimoto), and Target Similarity (Jaccard) networks.
- **Global Hybrid Search Architecture:**
  - Overcomes the traditional "Lexical Gap" in pure Vector RAG pipelines.
  - Concurrently queries across all 8 internal node labels (`Chemical`, `Gene`, `Disease`, `Phenotype`, `Generic`, etc.).
  - Combines **Semantic Vector Similarity** (via `all-MiniLM-L6-v2`) and **BM25 Exact Text Search** to perfectly capture highly specific chemical names (like _Octylphenol_) alongside conceptual questions.
- **Smart Graph Context Injection:** Extracts multi-hop relational evidence (e.g., `- [affects_expression_of] -> Gene ABC (Gene)`) and directly embeds it into the LLM context block, securely scaling up to 25 edges per node to respect maximum strict token constraints.
- **Interfaces:** Retrieval-augmented Chatbot (Streamlit) powered by GroqAPI.

---

## 🛠 Installation & Setup

### 1. Requirements

Ensure you have Python 3.9+ installed.

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory with your Neo4j credentials and Groq API key:

```env
NEO4J_URI=neo4j+s://your-database-id.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
GROQ_API_KEY=your-groq-api-key
```

### 3. Build the Database (Run Once)

The pipeline requires a two-step data ingestion process to perfectly configure the Neo4j backend:

**Step A: Ingest Nodes & Create Vector Embeddings**
This script parses the supporting dataset files, embeds their semantic text representations, inserts them into Neo4j, and creates Vector Indexes.

```bash
python graph_rag/build_graph_index.py --data-root . --output-dir artifacts
```

**Step B: Ingest Edges & Create BM25 Indexes**
This script parses all 36 relationship TSV files, builds the multi-hop `RELATED_TO` graph network between your nodes, and configures the Full-Text search capabilities for the Hybrid pipeline.
_(Note: Can take 30-90 minutes depending on your Neo4j cloud tier bandwidth)_

```bash
python graph_rag/build_relationships.py
```

---

## 💬 Usage

### Run the Streamlit Chatbot UI

```bash
streamlit run graph_rag/chat_app.py
```

### Features to Try

Ask complex graph-traversal questions such as:

- _"Explain octylphenol, its uses and chemical composition"_
- _"What genes does Benzene interact with, and does it increase or decrease their expression?"_
- _"Are there any diseases associated with both Bisphenol A and Diethylstilbestrol?"_

## Credits & Acknowledgements

**GraphRAG Architecture & Codebase Author:**

- Rohit B K

**Data Source & Original Authors:**
The toxicology data and graph structure are the work of the Computational Biology Group at IMSc. Please cite the following preprint for the original DEDuCT 3.0 resource:

Nikhil Chivukula, Shrish Vashishth, Pavithra Kandasamy, Shreyes Rajan Madgaonkar, Areejit Samal\*, DEDuCT 3.0: An enhanced and expanded FAIR-compliant resource and toxicology knowledge graph for endocrine disrupting chemicals. bioRxiv 2026.01.23.701267; doi: https://doi.org/10.64898/2026.01.23.701267
