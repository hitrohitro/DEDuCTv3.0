from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

try:
	from graph_rag.rag_pipeline import GraphRAG
except ModuleNotFoundError:
	sys.path.append(str(Path(__file__).resolve().parent))
	from rag_pipeline import GraphRAG


st.set_page_config(page_title="DEDuCTv3.0 GraphRAG Chatbot", page_icon="🧬", layout="wide")
st.title("DEDuCTv3.0 GraphRAG Chat")
st.caption("Ask biomedical graph questions over DEDuCTv3.0 KG")

artifacts_default = Path("artifacts")
artifacts_dir = Path(st.sidebar.text_input("Artifacts directory", value=str(artifacts_default)))
model_name = st.sidebar.text_input("Model", value="llama-3.3-70b-versatile")

if "rag" not in st.session_state:
	st.session_state.rag = None
if "messages" not in st.session_state:
	st.session_state.messages = []

if st.sidebar.button("Load Artifacts"):
	st.session_state.rag = GraphRAG(artifacts_dir)
	st.sidebar.success("Artifacts loaded")

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

user_prompt = st.chat_input("Ask a question about DEDuCTv3.0 KG")
if user_prompt:
	st.session_state.messages.append({"role": "user", "content": user_prompt})
	with st.chat_message("user"):
		st.markdown(user_prompt)

	if st.session_state.rag is None:
		response = "Load artifacts first using the sidebar button."
	else:
		with st.spinner("Retrieving graph evidence and generating answer..."):
			response = st.session_state.rag.answer(user_prompt, model=model_name)

	st.session_state.messages.append({"role": "assistant", "content": response})
	with st.chat_message("assistant"):
		st.markdown(response)
