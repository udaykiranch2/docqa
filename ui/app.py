"""Streamlit UI for the RAG Document Q&A system."""

import streamlit as st

import ui.api_client as api

SUPPORTED_EXTENSIONS = ["pdf", "txt", "csv", "docx", "py"]


def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "backend_ok" not in st.session_state:
        try:
            health = api.health_check()
            st.session_state.backend_ok = health.get("status") == "ok"
            st.session_state.index_ready = health.get("index_ready", False)
        except Exception:
            st.session_state.backend_ok = False
            st.session_state.index_ready = False


def _render_sidebar():
    with st.sidebar:
        st.header("Documents")

        # Show current stats.
        if st.session_state.backend_ok:
            try:
                stats = api.get_stats()
                st.metric("Indexed chunks", stats.get("chunk_count", 0))
                sources = stats.get("source_files", [])
                if sources:
                    with st.expander("Source files", expanded=False):
                        for s in sources:
                            st.text(s)
            except Exception:
                pass

        # File upload.
        st.divider()
        uploaded = st.file_uploader(
            "Upload documents",
            type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingest", type="primary"):
            files = [(f.name, f.getvalue()) for f in uploaded]
            with st.spinner("Ingesting documents..."):
                try:
                    result = api.upload_documents(files)
                    if result["success"]:
                        st.success(result["message"])
                        st.session_state.index_ready = True
                    else:
                        st.error(result["message"])
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        # Clear index.
        st.divider()
        if st.button("Clear index", type="secondary"):
            try:
                result = api.clear_index()
                st.success(result["message"])
                st.session_state.index_ready = False
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Clear failed: {e}")


def _render_chat():
    st.title("RAG Document Q&A")
    st.caption("Ask questions about your ingested documents.")

    if not st.session_state.backend_ok:
        st.error(
            "Cannot reach the FastAPI backend. "
            "Make sure it's running: `python run_api.py`"
        )
        return

    # Render existing messages.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        label = src["source_file"]
                        if src.get("page"):
                            label += f" (page {src['page']})"
                        st.markdown(f"**{label}**")
                        st.code(src["content_preview"][:300])

    # Chat input.
    if not st.session_state.index_ready:
        st.info("Upload documents in the sidebar to get started.")
        return

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = api.ask_question(prompt)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                label = src["source_file"]
                                if src.get("page"):
                                    label += f" (page {src['page']})"
                                st.markdown(f"**{label}**")
                                st.code(src["content_preview"][:300])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


def main():
    _init_session_state()
    _render_sidebar()
    _render_chat()


if __name__ == "__main__":
    main()
