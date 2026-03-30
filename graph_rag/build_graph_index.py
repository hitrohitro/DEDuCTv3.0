from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


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


def persist_index(output_dir: Path, graph: nx.MultiDiGraph, docs: list[Document]) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	graph_path = output_dir / "deduct_kg_graph.joblib"
	joblib.dump(graph, graph_path)
	stale_graphml = output_dir / "deduct_kg.graphml"
	if stale_graphml.exists():
		stale_graphml.unlink()

	corpus = [d.text for d in docs]
	vectorizer = TfidfVectorizer(
		lowercase=True,
		stop_words="english",
		ngram_range=(1, 2),
		max_features=250_000,
	)
	matrix = vectorizer.fit_transform(corpus)

	index_obj = {
		"vectorizer": vectorizer,
		"matrix": matrix,
		"documents": [asdict(d) for d in docs],
	}
	joblib.dump(index_obj, output_dir / "tfidf_index.joblib")

	stats = {
		"num_nodes": graph.number_of_nodes(),
		"num_edges": graph.number_of_edges(),
		"node_type_count": graph.graph.get("node_type_count", {}),
		"relation_count": graph.graph.get("relation_count", {}),
		"num_documents": len(docs),
	}
	(output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description="Build Graph RAG artifacts from DEDuCTv3.0 TSV files")
	parser.add_argument("--data-root", type=Path, default=Path("."), help="Path containing Supporting_Data/DEDuCT_KG/")
	parser.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Directory for graph and index artifacts")
	args = parser.parse_args()

	graph, docs = build_graph_and_documents(args.data_root)
	persist_index(args.output_dir, graph, docs)

	print("Build complete")
	print(f"Nodes: {graph.number_of_nodes()}")
	print(f"Edges: {graph.number_of_edges()}")
	print(f"Documents: {len(docs)}")
	print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
	main()
