# DEDuCTv3.0

## Overview

DEDuCTv3.0 is an enhanced and expanded FAIR-compliant resource and toxicology knowledge graph for endocrine disrupting chemicals (EDCs). This repository provides the datasets and a retrieval-augmented chatbot pipeline for exploring the knowledge graph and supporting data.

## Features

- Knowledge graph (KG) with 7 node types and 31 edge types
- Additional supporting data: CGPD tetramers, chemical similarity, target similarity
- Retrieval-augmented chatbot (Streamlit) and CLI, powered by GroqAPI LLM
- All project README files indexed for reference

## Usage

### 1. Build the retrieval index

```
python graph_rag/build_graph_index.py --data-root . --output-dir artifacts
```

### 2. Run the Streamlit chatbot

```
streamlit run graph_rag/chat_app.py
```

### 3. Run the CLI chatbot

```
python graph_rag/chat_cli.py
```

### 4. Environment

Create a `.env` file with your GroqAPI key:

```
GROQ_API_KEY=your-groq-api-key-here
```

## Data Contents

- `Supporting_Data/DEDuCT_KG/`: Node and edge tables for the KG
- `Supporting_Data/CGPD_tetramers_from_CTD.tsv`: Chemical-gene-phenotype-disease tetramers
- `Supporting_Data/chemical_similarity_network_edge_table.tsv`: Chemical similarity (Tanimoto)
- `Supporting_Data/target_similarity_network_edge_table.tsv`: Target similarity (Jaccard)
- All README files for documentation

## Reference

This repository provides the datasets associated with the following research article:

Nikhil Chivukula, Shrish Vashishth, Pavithra Kandasamy, Shreyes Rajan Madgaonkar, Areejit Samal*, [<i>DEDuCT 3.0: An enhanced and expanded FAIR-compliant resource and toxicology knowledge graph for endocrine disrupting chemicals</i>](https://www.biorxiv.org/content/10.64898/2026.01.23.701267), bioRxiv 2026.01.23.701267 (2026).
(* Corresponding Author)

## Contributors

- [Rohit B K](https://github.com/hitrohitro)
- [Nikhil Chivukula](https://github.com/NikC99)
