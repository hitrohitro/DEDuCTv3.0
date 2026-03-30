from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

try:
	from dotenv import load_dotenv
except Exception:
	load_dotenv = None

try:
	from groq import Groq
except Exception:
	Groq = None


@dataclass
class RetrievedItem:
	score: float
	text: str
	metadata: dict[str, Any]


class GraphRAG:
	def __init__(self, artifacts_dir: Path) -> None:
		if load_dotenv is not None:
			load_dotenv()
		self.artifacts_dir = artifacts_dir
		graph_joblib = artifacts_dir / "deduct_kg_graph.joblib"
		graph_graphml = artifacts_dir / "deduct_kg.graphml"
		if graph_joblib.exists():
			self.graph = joblib.load(graph_joblib)
		else:
			self.graph = nx.read_graphml(graph_graphml)
		index_obj = joblib.load(artifacts_dir / "tfidf_index.joblib")
		self.vectorizer = index_obj["vectorizer"]
		self.matrix = index_obj["matrix"]
		self.documents = index_obj["documents"]

	def retrieve(self, query: str, top_k: int = 12) -> list[RetrievedItem]:
		qv = self.vectorizer.transform([query])
		sims = cosine_similarity(qv, self.matrix).flatten()
		top_idx = sims.argsort()[::-1][:top_k]

		results: list[RetrievedItem] = []
		for i in top_idx:
			doc = self.documents[int(i)]
			results.append(
				RetrievedItem(
					score=float(sims[int(i)]),
					text=doc["text"],
					metadata=doc["metadata"],
				)
			)
		return results

	def _collect_context_nodes(self, retrievals: list[RetrievedItem], max_nodes: int = 40) -> list[str]:
		node_ids: list[str] = []
		seen: set[str] = set()

		for item in retrievals:
			meta = item.metadata
			for key in ("node_id", "source", "target"):
				value = meta.get(key)
				if value and value in self.graph and value not in seen:
					seen.add(value)
					node_ids.append(value)
					if len(node_ids) >= max_nodes:
						return node_ids
		return node_ids

	def _summarize_subgraph(self, seed_nodes: list[str], max_triples: int = 30) -> list[str]:
		triples: list[str] = []
		seen: set[tuple[str, str, str]] = set()

		for source in seed_nodes:
			if source not in self.graph:
				continue

			for target in self.graph.successors(source):
				edges = self.graph.get_edge_data(source, target)
				if not edges:
					continue
				for _, attrs in edges.items():
					relation = attrs.get("relation", "related_to")
					triple = (str(source), str(relation), str(target))
					if triple in seen:
						continue
					seen.add(triple)

					s_name = self.graph.nodes[source].get("name", source)
					t_name = self.graph.nodes[target].get("name", target)
					triples.append(f"{source} ({s_name}) -[{relation}]-> {target} ({t_name})")
					if len(triples) >= max_triples:
						return triples
		return triples

	def build_context(self, query: str, top_k: int = 12) -> tuple[list[RetrievedItem], list[str]]:
		retrievals = self.retrieve(query=query, top_k=top_k)
		seeds = self._collect_context_nodes(retrievals)
		triples = self._summarize_subgraph(seeds)
		return retrievals, triples

	def answer(self, query: str, model: str = "llama-3.3-70b-versatile") -> str:
		retrievals, triples = self.build_context(query)

		context_lines: list[str] = []
		for idx, item in enumerate(retrievals[:8], start=1):
			context_lines.append(f"[{idx}] score={item.score:.3f} | {item.text}")

		triple_lines = [f"- {t}" for t in triples[:20]]
		context_blob = "\n".join(context_lines)
		triple_blob = "\n".join(triple_lines) if triple_lines else "- No graph triples found"

		groq_key = os.getenv("GROQ_API_KEY", "").strip()
		if groq_key and Groq is not None:
			client = Groq(api_key=groq_key)
			system_prompt = (
				"You are a biomedical knowledge graph assistant. "
				"Answer using only the provided retrieval context and graph triples. "
				"If evidence is insufficient, clearly say so."
			)
			user_prompt = (
				f"Question: {query}\n\n"
				f"Retrieved context:\n{context_blob}\n\n"
				f"Graph triples:\n{triple_blob}\n\n"
				"Provide a concise answer with a short evidence section citing [index] items."
			)
			response = client.chat.completions.create(
				model=model,
				temperature=0.1,
				messages=[
					{"role": "system", "content": system_prompt},
					{"role": "user", "content": user_prompt},
				],
			)
			return response.choices[0].message.content or "No response generated."

		fallback = [
			"No LLM provider configured. Returning evidence-grounded retrieval summary.",
			"",
			f"Question: {query}",
			"",
			"Top retrieved evidence:",
		]
		fallback.extend([f"- [{i+1}] {r.text}" for i, r in enumerate(retrievals[:6])])
		fallback.append("")
		fallback.append("Relevant graph triples:")
		fallback.extend(triple_lines[:12] if triple_lines else ["- No triples available"])
		fallback.append("")
		fallback.append("Set GROQ_API_KEY to enable natural language answer generation.")
		return "\n".join(fallback)
