"""Streamlit entry point: python -m streamlit run app.py."""
import hashlib
import json

import streamlit as st

from rag.config import ROOT, Settings
from rag.embeddings import GeminiService, RAGError
from rag.generator import ask, citation
from rag.vector_store import VectorStore


def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("cited_sources"):
            st.caption("Cited references")
            for source in message["cited_sources"]:
                st.text(citation(source))
        if message.get("query"):
            with st.expander("Search query"):
                st.text(message["query"])
        if message.get("sources"):
            with st.expander(f"Retrieved passages ({len(message['sources'])})"):
                st.caption("Search results may include passages that were not cited in the answer. Scores measure similarity, not confidence.")
                for source in message["sources"]:
                    st.text(f"{source['source']} · Page {source['page']} · Similarity {source['score']:.3f}")
                    st.text(source["text"])
                    st.divider()


def main():
    st.set_page_config(page_title="Mini RAG | Chat with PDFs", page_icon="📚", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1100px; padding-top: 2rem;}
        [data-testid="stChatMessage"] {border-radius: 12px; padding: 1.2rem;}
        [data-testid="stMetric"] {background: var(--secondary-background-color); padding: .8rem; border-radius: 10px;}
        </style>""", unsafe_allow_html=True)
    st.title("📚 Mini RAG")
    st.caption("CHAT WITH YOUR PDF LIBRARY · Answers with document evidence")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("database", None)
    st.session_state.setdefault("selection", None)
    service = None
    try:
        service = GeminiService(Settings())
    except RAGError as exc:
        st.warning(str(exc))

    with st.sidebar:
        st.header("Your documents")
        files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                                 help="Text PDFs work best. Remove a file using its × button.")
        use_sample = st.checkbox("Use included document.pdf", value=False) if (ROOT / "document.pdf").exists() else False
        with st.expander("Retrieval settings"):
            chunk_size = st.number_input("Chunk size (characters)", min_value=100, max_value=2000, value=500, step=100)
            overlap = st.number_input("Chunk overlap (characters)", min_value=0, max_value=int(chunk_size) - 1, value=min(100, int(chunk_size) - 1), step=10)
            top_k = st.slider("Retrieved passages", 3, 5, 4)
        settings = Settings(chunk_size=int(chunk_size), chunk_overlap=int(overlap))
        documents = [(f.name, f.getvalue()) for f in files]
        if use_sample:
            documents.append(("document.pdf", (ROOT / "document.pdf").read_bytes()))
        selection = hashlib.sha256(json.dumps({"files": sorted((n, hashlib.sha256(b).hexdigest()) for n, b in documents),
                                               "settings": settings.index_config()}, sort_keys=True).encode()).hexdigest()
        if selection != st.session_state.selection:
            st.session_state.database = None
            st.session_state.messages = []
            st.session_state.selection = selection
        if st.button("Build document library", type="primary", disabled=not documents or service is None, use_container_width=True):
            progress = st.progress(0, text="Reading documents and preparing the index…")
            try:
                service.settings = settings
                with st.spinner("Preparing your library…"):
                    st.session_state.database = VectorStore(ROOT / "cache", settings).build(documents, service, progress.progress)
                st.session_state.messages = []
            except RAGError as exc:
                st.error(str(exc))
            except OSError:
                st.error("Could not read or save the library. Check available disk space and cache folder permissions.")
            except Exception:
                st.error("The library could not be built. Check the PDFs and try again.")
            finally:
                progress.empty()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
        st.caption("PDF text and questions are sent to Google Gemini. Indexes stay in this project's cache folder.")

    database = st.session_state.database
    if database:
        columns = st.columns(3)
        columns[0].metric("Documents", len(database.documents))
        columns[1].metric("Pages", sum(d["pages"] for d in database.documents))
        columns[2].metric("Indexed chunks", len(database.chunks))
        with st.sidebar:
            st.success("Library loaded from disk" if database.cached else "Library ready")
            for doc in database.documents:
                st.text(f"{doc['source']} · {doc['pages']} pages · {doc.get('chunks', 0)} chunks")
        for warning in database.warnings:
            st.warning(warning)
    else:
        st.info("Upload your lecture notes or select the included PDF, then click **Build document library**.")
        st.markdown("Ask a question, explore a follow-up, and inspect the passages behind each answer.")

    for message in st.session_state.messages:
        display_message(message)
    question = st.chat_input("Ask about your documents…", disabled=database is None or service is None)
    if question:
        if not question.strip():
            st.warning("Please enter a non-empty question.")
            return
        display_message({"role": "user", "content": question})
        try:
            service.settings = settings
            with st.spinner("Searching your documents and preparing an answer…"):
                answer = ask(question, st.session_state.messages, database, service, top_k)
            st.session_state.messages.extend([{"role": "user", "content": question}, answer])
            display_message(answer)
        except RAGError as exc:
            st.error(str(exc))
        except Exception:
            st.error("The question could not be completed. Please retry or rebuild the library.")


if __name__ == "__main__":
    main()
