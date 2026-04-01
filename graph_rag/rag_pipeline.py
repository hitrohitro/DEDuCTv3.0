from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import SessionExpired, ServiceUnavailable
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

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
	def __init__(self):
		load_dotenv()
		self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
		self.user = os.getenv("NEO4J_USER", "neo4j")
		self.password = os.getenv("NEO4J_PASSWORD", "password")
		self.driver = GraphDatabase.driver(
			self.uri,
			auth=(self.user, self.password),
			max_connection_lifetime=300,   # keep connections alive <=5 min
			keep_alive=True,
		)
		self.model = SentenceTransformer('all-MiniLM-L6-v2')

	def _reconnect(self) -> None:
		"""Close the existing driver and create a fresh connection."""
		try:
			self.driver.close()
		except Exception:
			pass
		self.driver = GraphDatabase.driver(
			self.uri,
			auth=(self.user, self.password),
			max_connection_lifetime=300,
			keep_alive=True,
		)

	def _run_hybrid_query(self, label: str, index_name: str, text_index_name: str, query: str, embedding: list, top_k: int) -> list:
		"""Execute both vector and full-text searches, merge nodes, and fetch neighbors."""
		import re
		with self.driver.session() as session:
			# 1. Vector Search
			vec_res = session.run(
				f"""
				WITH $embedding AS embedding
				CALL db.index.vector.queryNodes('{index_name}', $top_k, embedding) YIELD node, score
				RETURN node.doc_id AS id, score
				""",
				embedding=embedding, top_k=top_k
			).data()

			# 2. Full-Text BM25 Search
			# Clean query for Lucene syntax, extracting relevant keyword tokens
			clean_q = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
			lucene_q = " OR ".join([w for w in clean_q.split() if len(w) > 3 and w.lower() not in ('explain', 'what', 'how', 'the', 'and', 'for')])
			if not lucene_q:
				lucene_q = clean_q

			text_res = []
			try:
				text_res = session.run(
					f"""
					CALL db.index.fulltext.queryNodes('{text_index_name}', $lucene_q, {{limit: $top_k}}) YIELD node, score
					RETURN node.doc_id AS id, score
					""",
					lucene_q=lucene_q, top_k=top_k
				).data()
			except Exception as e:
				print(f"Full-text search fallback / error: {e}")

			# 3. Merge and rank
			score_map = {}
			for r in vec_res:
				score_map[r['id']] = max(score_map.get(r['id'], 0), r['score'])
			for r in text_res:
				# Full-text scores can be > 1.0, so normalize or cap them high to guarantee retrieval
				score_map[r['id']] = max(score_map.get(r['id'], 0), 1.0)

			combined_ids = list(score_map.keys())
			if not combined_ids:
				return []

			# 4. Fetch the actual context and neighbors using the doc_id index
			result = session.run(
				f"""
				UNWIND $node_ids AS doc_id
				MATCH (node:{label} {{doc_id: doc_id}})
				OPTIONAL MATCH (node)-[r]->(neighbor)
				WITH node, collect(CASE WHEN r IS NOT NULL THEN {{relation: type(r), name: coalesce(neighbor.name, neighbor.text, ''), kind: head(labels(neighbor))}} ELSE NULL END) AS relations
				RETURN node.doc_id AS id, node.text AS node_text, relations
				""",
				node_ids=combined_ids
			)
			
			final_rows = []
			for r in result:
				data = r.data()
				text_block = data.get('node_text') or ""
				relations = data.get('relations') or []
				
				rel_lines = []
				# Cap at 25 relationships per node to prevent blowing up the LLM token limits
				for rel in relations[:25]:
					if rel and rel.get('relation'):
						# e.g. - [affects_expression_of] -> Gene ABC
						rel_lines.append(f"- [{rel['relation']}] -> {rel['name']} ({rel['kind']})")
				
				if rel_lines:
					text_block += "\nDirect Graph Relationships:\n" + "\n".join(rel_lines)
					if len(relations) > 25:
						text_block += f"\n...and {len(relations) - 25} more graph connections omitted for brevity."
				
				final_rows.append({
					'id': data.get('id'),
					'score': dict.get(score_map, data.get('id'), 0.0),
					'text': text_block
				})
			
			# Return the top K combined results
			final_rows.sort(key=lambda x: x['score'], reverse=True)
			return final_rows[:top_k]

	def retrieve(self, query: str, top_k: int = 8, labels: list[str] = None):
		if labels is None:
			# Global search across all relevant node types
			labels = ["Chemical", "Gene", "Disease", "AOP", "Phenotype", "KeyEvent", "DEDuCT_Endpoint", "Generic"]
		
		embedding = self.model.encode([query])[0].tolist()
		
		all_results = []
		for label in labels:
			index_name = f"{label.lower()}_embedding_index"
			text_index_name = f"{label.lower()}_text_index"
			try:
				res = self._run_hybrid_query(label, index_name, text_index_name, query, embedding, top_k)
				all_results.extend(res)
			except (SessionExpired, ServiceUnavailable):
				# Connection went stale — reconnect once and retry this label
				self._reconnect()
				try:
					res = self._run_hybrid_query(label, index_name, text_index_name, query, embedding, top_k)
					all_results.extend(res)
				except Exception as e:
					print(f"Failed to query {label} after reconnect: {e}")
			except Exception as e:
				# Log and skip if an index doesn't exist for a label (e.g. Generic)
				print(f"Skipping {label} due to query error: {e}")
				
		# Deduplicate and rank globally
		unique_results = {}
		for r in all_results:
			uid = r['id']
			if uid not in unique_results or r['score'] > unique_results[uid]['score']:
				unique_results[uid] = r
				
		final_results = list(unique_results.values())
		final_results.sort(key=lambda x: x['score'], reverse=True)
		
		return final_results[:top_k]

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

	def build_context(self, query: str, top_k: int = 8, labels: list[str] = None):
		retrievals = self.retrieve(query=query, top_k=top_k, labels=labels)
		# For now, just return the retrieved nodes and their neighbors as context
		return retrievals

	def answer(self, query: str, model: str = "llama-3.3-70b-versatile", labels: list[str] = None) -> str:
		retrievals = self.build_context(query, labels=labels)
		context_lines = []
		for idx, item in enumerate(retrievals, start=1):
			context_lines.append(f"[{idx}] score={item['score']:.3f} | {item['text']}")

		context_blob = "\n".join(context_lines)

		groq_key = os.getenv("GROQ_API_KEY", "").strip()
		if groq_key and Groq is not None:
			client = Groq(api_key=groq_key)
			system_prompt = (
				"You are a biomedical knowledge graph assistant. "
				"Answer graph-related questions using ONLY the provided retrieval context. "
				"If the question is just a general greeting or generic, you may answer normally."
			)
			user_prompt = (
				f"Question: {query}\n\n"
				f"Retrieved context:\n{context_blob}\n\n"
				"Provide a concise answer with a short evidence section citing [index] items. "
				"If the question is generic or unrelated to the context, answer it naturally without an evidence section."
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
		fallback.extend([f"- [{i+1}] {item['text']}" for i, item in enumerate(retrievals[:6])])
		fallback.append("")
		fallback.append("Set GROQ_API_KEY to enable natural language answer generation.")
		return "\n".join(fallback)
