"""
app.py
======
Streamlit front-end for the AI-Powered Adaptive Learning Chatbot.

Run with:
    streamlit run app.py
"""

import uuid
from typing import Any, Dict, List

import streamlit as st

import config
from rag_engine import (
    build_rag_chain,
    clear_session_history,
    is_vector_store_ready,
    stream_answer,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Learning Chatbot",
    page_icon="🎓",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and vector store...")
def load_chain(model_name: str):
    """Build the RAG chain once per model (cached across reruns)."""
    return build_rag_chain(model_name)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Create the keys we rely on the first time the page loads."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        # Each item: {"role": "user"|"assistant", "content": str, "sources": list}
        st.session_state.messages = []


def reset_chat() -> None:
    """Clear both the visible chat and the LangChain memory."""
    clear_session_history(st.session_state.session_id)
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())


def render_sources(sources: List[Dict[str, Any]]) -> None:
    """Show the retrieved sources in a collapsible section."""
    if not sources:
        return
    with st.expander("📚 Sources used"):
        for src in sources:
            page = f", page {src['page']}" if src.get("page") else ""
            st.markdown(f"- **{src['source']}**{page}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> Dict[str, str]:
    """Draw the sidebar controls and return the chosen settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        level = st.select_slider(
            "Your learning level",
            options=config.STUDENT_LEVELS,
            value="Beginner",
            help="The tutor adapts the depth and style of its explanations.",
        )
        model = st.selectbox(
            "Local LLM (Ollama)",
            options=config.AVAILABLE_MODELS,
            index=config.AVAILABLE_MODELS.index(config.DEFAULT_MODEL),
            help="The model must already be pulled with `ollama pull <name>`.",
        )
        if st.button("🗑️ New conversation", use_container_width=True):
            reset_chat()
            st.rerun()

        st.divider()
        st.caption(
            "Answers are generated locally with Ollama and grounded in the PDFs "
            "stored in the `data/` folder."
        )
    return {"level": level, "model": model}


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    settings = render_sidebar()

    st.title("🎓 AI-Powered Adaptive Learning Chatbot")
    st.caption(
        f"Level: **{settings['level']}**  |  Model: **{settings['model']}**  |  "
        "Local RAG with ChromaDB"
    )

    # Stop early with a helpful message if ingestion was not run.
    if not is_vector_store_ready():
        st.error(
            "No knowledge base found. Put your PDFs in the `data/` folder and "
            "run `python ingest.py` in the terminal, then refresh this page."
        )
        st.stop()

    # Build (or fetch from cache) the chain.
    try:
        chain = load_chain(settings["model"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not initialise the chatbot: {exc}")
        st.stop()

    # Replay the conversation stored in session state.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    # Welcome hint on an empty chat.
    if not st.session_state.messages:
        st.info(
            "👋 Ask me anything about your study material. "
            "Try: *\"Explain the main idea of chapter 1\"*."
        )

    # Handle a new question.
    question = st.chat_input("Ask a question about your study material...")
    if not question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources: List[Dict[str, Any]] = []
        try:
            answer = st.write_stream(
                stream_answer(
                    chain,
                    question=question,
                    level=settings["level"],
                    session_id=st.session_state.session_id,
                    sources_out=sources,
                )
            )
            render_sources(sources)
        except Exception as exc:  # noqa: BLE001
            answer = (
                "⚠️ I could not reach the local language model. Make sure Ollama "
                f"is running and the model **{settings['model']}** is pulled "
                "(`ollama pull " f"{settings['model']}`).\n\n"
                f"Technical details: `{exc}`"
            )
            sources = []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
