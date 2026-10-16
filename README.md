# 🎓 AI-Powered Adaptive Learning Chatbot

A fully local, privacy-friendly study assistant built with **Retrieval-Augmented Generation (RAG)**. It reads your PDF notes, retrieves the most relevant passages, and explains them at the student's level using analogies, bullet points, and a closing concept-check question.

No paid APIs. Everything runs on your own machine.

## Features

- **Local LLM** via [Ollama](https://ollama.com) (Llama 3 or Mistral)
- **Open-source embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **ChromaDB** persistent vector store
- **LangChain LCEL** chain with `RunnableWithMessageHistory` for multi-turn follow-ups
- **Adaptive prompting**: Beginner / Intermediate / Advanced explanations
- **Streamlit** chat UI with streaming answers and source citations

## Architecture

```
PDFs (data/) -> PyPDFLoader -> RecursiveCharacterTextSplitter (800 / 150)
             -> all-MiniLM-L6-v2 embeddings -> ChromaDB (chroma_db/)

Question + chat history -> condense to standalone question -> retriever (top-4)
        -> adaptive tutor prompt -> Ollama LLM -> streamed answer in Streamlit
```

## Project Structure

```
adaptive-learning-chatbot/
├── data/              # Put your PDF files here
├── chroma_db/         # Auto-generated vector store
├── config.py          # Central settings
├── ingest.py          # PDF -> chunks -> embeddings -> ChromaDB
├── rag_engine.py      # LLM, retriever, prompts, LCEL chain, memory
├── app.py             # Streamlit UI
├── requirements.txt
└── README.md
```

## Installation

### 1. Install Ollama and pull a model

Download Ollama from <https://ollama.com/download>, then:

```bash
ollama pull llama3        # or: ollama pull mistral
ollama list               # confirm the model is installed
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add study material and build the knowledge base

```bash
mkdir data                 # if it does not exist
# copy your PDF files into data/
python ingest.py
```

The first run downloads the embedding model (~90 MB) once.

### 4. Launch the app

Make sure Ollama is running, then:

```bash
streamlit run app.py
```

Open <http://localhost:8501> in your browser.

## Usage

1. Choose your **learning level** and **model** in the sidebar.
2. Ask a question about your PDFs.
3. Ask follow-ups such as *"explain that more simply"*; the chat memory resolves them.
4. Add new PDFs later and re-run `python ingest.py` to refresh the knowledge base.

## Configuration

All tunable values (chunk size, `k`, model names, paths) live in `config.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` / model error | Start Ollama (`ollama serve`) and check `ollama list` |
| "No knowledge base found" | Add PDFs to `data/` and run `python ingest.py` |
| Scanned PDF skipped | The PDF has no text layer; use an OCR'd version |
| Slow answers | Use a smaller model, e.g. `ollama pull llama3.2:3b` and add it to `AVAILABLE_MODELS` |
| Cannot rebuild store on Windows | Stop Streamlit first, then run `python ingest.py` |

## Tech Stack

Python · LangChain · Ollama · HuggingFace Sentence-Transformers · ChromaDB · Streamlit

## License

For academic use. Add your preferred license (e.g., MIT) before publishing.
