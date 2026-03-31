from __future__ import annotations
import json
import pickle
import numpy as np
import os

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv


@dataclass
class Document:
	doc_id: str
	kind: str
	text: str
	metadata: dict[str, Any]


def row_to_text(row: pd.Series, prefix: str) -> str:
	parts: list[str] = []
	for col, value in row.items():
		if pd.isna(value):
			continue
		sval = str(value).strip()
		if not sval:
			continue
		parts.append(f"{col}: {sval}")
	return f"{prefix}. " + "; ".join(parts)


def parse_edge_filename(file_path: Path) -> tuple[str, str, str]:
	# Expected format: NodeType.relation.NodeType.edges.tsv
	pieces = file_path.name.split(".")
	if len(pieces) < 5:
		raise ValueError(f"Unexpected edge filename format: {file_path.name}")
	source_type = pieces[0]
	target_type = pieces[-3]
	relation = ".".join(pieces[1:-3])
	return source_type, relation, target_type


def build_graph_and_documents(data_root: Path) -> tuple[nx.MultiDiGraph, list[Document]]:
	node_dir = data_root / "Supporting_Data" / "DEDuCT_KG" / "node_tables"
	edge_dir = data_root / "Supporting_Data" / "DEDuCT_KG" / "edge_tables"
	support_dir = data_root / "Supporting_Data"

	graph = nx.MultiDiGraph(name="DEDuCT-KG")
	docs: list[Document] = []
	node_type_count: dict[str, int] = {}

	# --- KG nodes ---
	for node_file in sorted(node_dir.glob("*.nodes.tsv")):
		df = pd.read_csv(node_file, sep="\t", dtype=str, keep_default_na=False)
		for _, row in df.iterrows():
			node_id = str(row.get("id", "")).strip()
			if not node_id:
				continue
			attributes = {k: str(v) for k, v in row.to_dict().items() if str(v).strip()}
			node_type = attributes.get("node_type", "Unknown")
			node_type_count[node_type] = node_type_count.get(node_type, 0) + 1
			graph.add_node(node_id, **attributes)
			docs.append(
				Document(
					doc_id=f"node::{node_id}",
					kind="node",
					text=row_to_text(row, prefix=f"Node {node_id}"),
					metadata={"node_id": node_id, "node_type": node_type, "source_file": node_file.name},
				)
			)

	# --- KG edges ---
	relation_count: dict[str, int] = {}
	for edge_file in sorted(edge_dir.glob("*.edges.tsv")):
		source_type, relation, target_type = parse_edge_filename(edge_file)
		df = pd.read_csv(edge_file, sep="\t", dtype=str, keep_default_na=False)
		for idx, row in df.iterrows():
			source = str(row.get("source", "")).strip()
			target = str(row.get("target", "")).strip()
			if not source or not target:
				continue
			edge_attrs = {k: str(v) for k, v in row.to_dict().items() if str(v).strip()}
			edge_attrs.update({"relation": relation, "source_type": source_type, "target_type": target_type})
			graph.add_edge(source, target, key=f"{relation}:{idx}", **edge_attrs)
			relation_count[relation] = relation_count.get(relation, 0) + 1
			source_name = graph.nodes[source].get("name", source) if source in graph.nodes else source
			target_name = graph.nodes[target].get("name", target) if target in graph.nodes else target
			edge_text = (
				f"Edge {source} ({source_name}) -[{relation}]-> {target} ({target_name}). "
				+ row_to_text(row, prefix="Edge metadata")
			)
			docs.append(
				Document(
					doc_id=f"edge::{edge_file.stem}::{idx}",
					kind="edge",
					text=edge_text,
					metadata={
						"source": source,
						"target": target,
						"relation": relation,
						"source_file": edge_file.name,
					},
				)
			)

	# --- CGPD tetramers ---
	cgpd_path = support_dir / "CGPD_tetramers_from_CTD.tsv"
	if cgpd_path.exists():
		df = pd.read_csv(cgpd_path, sep="\t", dtype=str, keep_default_na=False)
		for idx, row in df.iterrows():
			text = (
				f"CGPD tetramer: Chemical {row.get('Chemical Identifier','')} (Gene {row.get('Gene Symbol','')}) "
				f"affects phenotype {row.get('Phenotype Name','')} and disease {row.get('Disease Name','')} "
				f"(IDs: {row.get('Chemical Identifier','')}, {row.get('Gene Identifier','')}, {row.get('Phenotype Identifier','')}, {row.get('Disease Identifier','')})"
			)
			docs.append(
				Document(
					doc_id=f"cgpd::{idx}",
					kind="cgpd_tetramer",
					text=text,
					metadata=row.to_dict(),
				)
			)

	# --- Chemical similarity ---
	chem_sim_path = support_dir / "chemical_similarity_network_edge_table.tsv"
	if chem_sim_path.exists():
		df = pd.read_csv(chem_sim_path, sep="\t", dtype=str, keep_default_na=False)
		for idx, row in df.iterrows():
			text = (
				f"Chemical similarity: {row.get('Chemical 1','')} and {row.get('Chemical 2','')} "
				f"have Tanimoto coefficient {row.get('Tanimoto Coefficient','')}"
			)
			docs.append(
				Document(
					doc_id=f"chemsim::{idx}",
					kind="chemical_similarity",
					text=text,
					metadata=row.to_dict(),
				)
			)

	# --- Target similarity ---
	target_sim_path = support_dir / "target_similarity_network_edge_table.tsv"
	if target_sim_path.exists():
		df = pd.read_csv(target_sim_path, sep="\t", dtype=str, keep_default_na=False)
		for idx, row in df.iterrows():
			text = (
				f"Target similarity: {row.get('Chemical 1','')} and {row.get('Chemical 2','')} "
				f"have Jaccard similarity {row.get('Jaccard Similarity','')}"
			)
			docs.append(
				Document(
					doc_id=f"targetsim::{idx}",
					kind="target_similarity",
					text=text,
					metadata=row.to_dict(),
				)
			)

	# --- README files as reference ---
	readme_paths = [
		data_root / "README.md",
		support_dir / "README.md",
		data_root / "Supporting_Data" / "DEDuCT_KG" / "README.md",
	]
	for rpath in readme_paths:
		if rpath.exists():
			with open(rpath, encoding="utf-8") as f:
				text = f.read()
			docs.append(
				Document(
					doc_id=f"readme::{rpath.name}",
					kind="readme",
					text=text,
					metadata={"source_file": rpath.name},
				)
			)

	graph.graph["node_type_count"] = node_type_count
	graph.graph["relation_count"] = relation_count
	return graph, docs


def store_embeddings_in_neo4j(docs: list[Document], embedding_dim: int = 384):
	"""
	Store documents as nodes in Neo4j, using node_type as label, and auto-create vector index per label.
	"""
	load_dotenv()
	uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
	user = os.getenv("NEO4J_USER", "neo4j")
	password = os.getenv("NEO4J_PASSWORD", "password")


	# --- Fallback: Load or compute embeddings ---
	if os.path.exists("docs.pkl") and os.path.exists("embeddings.npy"):
		print("Loading cached docs and embeddings...")
		with open("docs.pkl", "rb") as f:
			docs = pickle.load(f)
		embeddings = np.load("embeddings.npy")
	else:
		model = SentenceTransformer('all-MiniLM-L6-v2')
		texts = [d.text for d in docs]
		embeddings = model.encode(texts, show_progress_bar=True)
		# Save for resume
		with open("docs.pkl", "wb") as f:
			pickle.dump(docs, f)
		np.save("embeddings.npy", embeddings)

	# Group docs by label
	label_to_docs = {}
	for doc, emb in zip(docs, embeddings):
		label = doc.metadata.get("node_type", "Generic")
		if not label.isidentifier():
			label = "Generic"
		label_to_docs.setdefault(label, []).append((doc, emb))

	driver = GraphDatabase.driver(uri, auth=(user, password))
	with driver.session() as session:
		for label, doc_emb_list in label_to_docs.items():
			# Upsert nodes
			for doc, emb in doc_emb_list:
				session.run(
					f"MERGE (n:{label} {{doc_id: $doc_id}}) SET n.text = $text, n.embedding = $embedding, n.kind = $kind, n.metadata = $metadata",
					doc_id=doc.doc_id,
					text=doc.text,
					embedding=emb.tolist(),
					kind=doc.kind,
					metadata=json.dumps(doc.metadata)  # Serialize metadata to JSON string
				)
			# Create vector index if not exists (Neo4j Aura/5.x+)
			index_name = f"{label.lower()}_embedding_index"
			cypher_check = f"SHOW INDEXES YIELD name WHERE name = '{index_name}' RETURN count(*) AS count"
			exists = session.run(cypher_check).single()["count"]
			if not exists:
				cypher_create = (
					f"CREATE VECTOR INDEX {index_name} FOR (n:{label}) ON (n.embedding) "
					f"OPTIONS {{indexConfig: {{`vector.dimensions`: {embedding_dim}, `vector.similarity_function`: 'cosine'}}}}"
				)
				session.run(cypher_create)
				print(f"Created vector index: {index_name} for label: {label}")
	driver.close()



def main() -> None:
	parser = argparse.ArgumentParser(description="Build Graph RAG artifacts from DEDuCTv3.0 TSV files and store embeddings in Neo4j with automatic label and index management.")
	parser.add_argument("--data-root", type=Path, default=Path("."), help="Path containing Supporting_Data/DEDuCT_KG/")
	parser.add_argument("--embedding-dim", type=int, default=384, help="Embedding dimension (default 384 for all-MiniLM-L6-v2)")
	args = parser.parse_args()

	graph, docs = build_graph_and_documents(args.data_root)
	store_embeddings_in_neo4j(docs, embedding_dim=args.embedding_dim)

	print("Embeddings stored in Neo4j with automatic node labels and vector indexes.")


if __name__ == "__main__":
	main()
