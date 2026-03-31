from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

try:
	from graph_rag.rag_pipeline import GraphRAG
	from neo4j.exceptions import SessionExpired, ServiceUnavailable
except ModuleNotFoundError:
	sys.path.append(str(Path(__file__).resolve().parent))
	from rag_pipeline import GraphRAG
	from neo4j.exceptions import SessionExpired, ServiceUnavailable


st.set_page_config(page_title="DEDuCTv3.0 GraphRAG Chatbot", page_icon="🧬", layout="wide")
st.title("DEDuCTv3.0 GraphRAG Chat")
st.caption("Ask biomedical graph questions over DEDuCTv3.0 KG")


model_name = st.sidebar.text_input("Model", value="llama-3.3-70b-versatile")

if "rag" not in st.session_state:
	st.session_state.rag = GraphRAG()
if "messages" not in st.session_state:
	st.session_state.messages = []

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

user_prompt = st.chat_input("Ask a question about DEDuCTv3.0 KG")
if user_prompt:
	st.session_state.messages.append({"role": "user", "content": user_prompt})
	with st.chat_message("user"):
		st.markdown(user_prompt)

	with st.spinner("Retrieving graph evidence and generating answer..."):
		try:
			response = st.session_state.rag.answer(user_prompt, model=model_name)
		except (SessionExpired, ServiceUnavailable) as e:
			# Auto-reconnect failed twice — reset the RAG object so the
			# driver is re-created fresh on the next query.
			st.session_state.rag = GraphRAG()
			st.warning(
				f"⚠️ Lost connection to Neo4j (`{e}`). "
				"The connection has been reset — please re-send your question."
			)
			response = None
		except Exception as e:
			st.error(f"❌ Unexpected error: {e}")
			response = None

	if response is not None:
		st.session_state.messages.append({"role": "assistant", "content": response})
		with st.chat_message("assistant"):
			st.markdown(response)
