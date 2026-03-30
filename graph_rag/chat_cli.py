from __future__ import annotations

import argparse
from pathlib import Path

from rag_pipeline import GraphRAG

def main() -> None:
	parser = argparse.ArgumentParser(description="CLI chatbot for DEDuCTv3.0 Graph RAG")
	parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
	parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile")
	args = parser.parse_args()

	rag = GraphRAG(args.artifacts_dir)

	print("DEDuCTv3.0 Graph RAG chat")
	print("Type 'exit' to quit")

	while True:
		user_query = input("\nYou: ").strip()
		if not user_query:
			continue
		if user_query.lower() in {"exit", "quit"}:
			break

		answer = rag.answer(user_query, model=args.model)
		print("\nAssistant:")
		print(answer)

if __name__ == "__main__":
	main()
