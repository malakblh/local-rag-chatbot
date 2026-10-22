"""Interface Streamlit : streamlit run ui.py"""
import streamlit as st

import config
from core import get_client
from ingest import ingest_pdf, reset_collection
from rag import rag_stream
from rag import rag_stream, page_label

st.set_page_config(page_title="Chatbot RAG", page_icon="🤖")
st.title("🤖 Chatbot RAG local")
st.caption(f"Qdrant + Ollama ({config.LLM_MODEL}) – 100 % hors ligne")

# Vérifie que Qdrant répond
try:
    client = get_client()
    has_collection = client.collection_exists(config.COLLECTION_NAME)
except Exception:
    st.error("Qdrant ne répond pas. Lancez-le avec : `docker compose up -d`")
    st.stop()

# ----- Barre latérale : documents -----
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader("Ajouter un PDF", type="pdf")
    if uploaded and st.button("Indexer ce PDF"):
        with st.spinner("Indexation en cours..."):
            try:
                n = ingest_pdf(uploaded, uploaded.name)
                st.success(f"{n} passages indexés.")
                has_collection = True
            except ValueError as e:
                st.error(str(e))

    if has_collection:
        st.caption(f"{client.count(config.COLLECTION_NAME).count} passages dans la base")
    else:
        st.info("Aucun document indexé pour l'instant.")

    if st.button("🗑️ Vider la base"):
        reset_collection()
        st.session_state.messages = []
        st.rerun()


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s['source']}, {page_label(s)}** (score {s['score']:.2f})")
            st.caption(s["text"][:300] + "…")


# ----- Chat -----
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Posez votre question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche dans le document..."):
                sources, stream = rag_stream(question)
            answer = st.write_stream(stream)
            if sources:
                show_sources(sources)
        except Exception as e:
            answer, sources = "", []
            st.error(f"Erreur : {e}\n\nOllama est-il lancé et le modèle `{config.LLM_MODEL}` téléchargé ?")

    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
