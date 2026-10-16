"""
rag_engine.py
=============
Core backend of the AI-Powered Adaptive Learning Chatbot.

Responsibilities
----------------
* Load the local LLM served by Ollama (Llama 3 / Mistral).
* Load the persistent ChromaDB vector store and expose it as a retriever.
* Define the prompt templates (question rewriting + adaptive tutoring).
* Assemble an LCEL chain and wrap it with ``RunnableWithMessageHistory``
  so that multi-turn follow-up questions work.

Chain flow (per user turn)
--------------------------
    input + chat_history
        -> [condense follow-up into a standalone question]
        -> [retrieve top-k chunks from ChromaDB]
        -> [format chunks into a context string]
        -> [adaptive tutor prompt -> LLM -> text answer]
"""

from operator import itemgetter
from typing import Any, Dict, Iterator, List, Optional

from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

import config

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Step 1: turn a context-dependent follow-up ("explain that more simply")
# into a self-contained query suitable for vector search.
CONDENSE_SYSTEM_PROMPT = (
    "Given the chat history and the student's latest message, rewrite the "
    "latest message as a single, standalone question that can be understood "
    "without the chat history. Do NOT answer the question. If the message is "
    "already standalone, return it unchanged. Output only the question."
)

# Step 2: the adaptive tutoring prompt (the heart of the project).
ADAPTIVE_SYSTEM_PROMPT = """\
You are an expert, friendly AI tutor inside an adaptive learning platform.
The current student level is: {level}.

GROUNDING RULES
- Answer ONLY using the information inside the <context> block below.
- If the context does not contain the answer, say clearly: "I could not find
  this in the study material you uploaded." Then suggest how the student could
  rephrase the question. Do NOT invent facts and do NOT add a concept check
  in that case.
- Never mention the words "context" or "retrieved documents" to the student;
  refer to it naturally as "your study material" or "your notes".

ADAPTATION RULES (match the student level)
- Beginner: use plain everyday language, define every technical term, keep
  sentences short, and rely heavily on simple real-life analogies.
- Intermediate: assume basic familiarity, use correct terminology with brief
  reminders, and connect ideas to one another with one clear analogy.
- Advanced: be concise and technically precise, discuss nuances, trade-offs
  and edge cases, and use an analogy only when it adds real insight.

RESPONSE FORMAT (follow this structure)
1. A short, direct answer in 1-2 sentences.
2. A clear explanation using bullet points.
3. An "Analogy:" line that compares the idea to something familiar.
4. A final line starting with "Concept Check:" containing ONE question that
   tests understanding and can be answered strictly from the material in the
   context. Do not reveal the answer.

<context>
{context}
</context>
"""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

ADAPTIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ADAPTIVE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------------------------
# Building blocks: embeddings, vector store, retriever, LLM
# ---------------------------------------------------------------------------


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the HuggingFace embedding model (must match ``ingest.py``)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def is_vector_store_ready() -> bool:
    """True if the ChromaDB directory exists and is not empty."""
    return config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir())


def get_vector_store() -> Chroma:
    """Open the persistent ChromaDB collection created by ``ingest.py``.

    Raises:
        FileNotFoundError: if ingestion has not been run yet.
    """
    if not is_vector_store_ready():
        raise FileNotFoundError(
            f"Vector store not found at '{config.CHROMA_DIR}'. "
            "Run 'python ingest.py' first."
        )
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def get_retriever():
    """Return a similarity retriever returning the top-k chunks."""
    return get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K},
    )


def get_llm(model_name: str = config.DEFAULT_MODEL) -> ChatOllama:
    """Create the local chat model served by Ollama."""
    return ChatOllama(
        model=model_name,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.LLM_TEMPERATURE,
    )


# ---------------------------------------------------------------------------
# Conversation memory (one history object per chat session)
# ---------------------------------------------------------------------------

_SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (creating if needed) the message history for ``session_id``."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def clear_session_history(session_id: str) -> None:
    """Forget the conversation memory of ``session_id``."""
    _SESSION_STORE.pop(session_id, None)


# ---------------------------------------------------------------------------
# LCEL chain assembly
# ---------------------------------------------------------------------------


def format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into one context string with source tags."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, p.{page + 1}" if isinstance(page, int) else source
        parts.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(model_name: str = config.DEFAULT_MODEL) -> Runnable:
    """Build the conversational adaptive RAG chain.

    The returned runnable expects an input dict::

        {"input": "<student question>", "level": "Beginner"}

    and must be invoked with
    ``config={"configurable": {"session_id": "<id>"}}``.
    Its output dict contains the keys ``answer`` (str) and ``docs``
    (the retrieved ``Document`` objects, useful for showing sources).
    """
    llm = get_llm(model_name)
    retriever = get_retriever()
    parser = StrOutputParser()

    # Sub-chain that rewrites follow-up questions using chat history.
    condense_chain = CONDENSE_PROMPT | llm | parser

    def condense_question(inputs: Dict[str, Any]) -> str:
        """Skip the extra LLM call on the very first turn."""
        if not inputs.get("chat_history"):
            return inputs["input"]
        return condense_chain.invoke(inputs).strip() or inputs["input"]

    core_chain = (
        RunnablePassthrough.assign(
            standalone_question=RunnableLambda(condense_question)
        )
        | RunnablePassthrough.assign(
            docs=itemgetter("standalone_question") | retriever
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: format_docs(x["docs"]))
        )
        | RunnablePassthrough.assign(answer=ADAPTIVE_PROMPT | llm | parser)
    )

    # Add automatic multi-turn memory around the chain.
    return RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def stream_answer(
    chain: Runnable,
    question: str,
    level: str,
    session_id: str,
    sources_out: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[str]:
    """Stream the answer token by token.

    Args:
        chain: Chain returned by :func:`build_rag_chain`.
        question: The student's message.
        level: One of ``config.STUDENT_LEVELS``.
        session_id: Identifier used to look up conversation memory.
        sources_out: Optional list that will be filled with unique source
            dicts (``{"source": str, "page": int | None}``) after retrieval.

    Yields:
        Pieces of the answer text as they are generated.
    """
    seen = set()
    stream = chain.stream(
        {"input": question, "level": level},
        config={"configurable": {"session_id": session_id}},
    )
    for chunk in stream:
        if sources_out is not None and "docs" in chunk:
            for doc in chunk["docs"]:
                page = doc.metadata.get("page")
                key = (doc.metadata.get("source"), page)
                if key not in seen:
                    seen.add(key)
                    sources_out.append(
                        {
                            "source": doc.metadata.get("source", "unknown"),
                            "page": page + 1 if isinstance(page, int) else None,
                        }
                    )
        if "answer" in chunk:
            yield chunk["answer"]
