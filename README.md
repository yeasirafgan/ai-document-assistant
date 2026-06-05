# AI Document Assistant

A RAG (Retrieval Augmented Generation) API that lets you upload PDF documents and ask questions about them. Built with FastAPI, LangGraph, Claude, and pgvector.

## Demo

Upload a PDF → Ask questions → Get answers with source page citations.

```
POST /ingest   →  Upload a PDF, chunks are embedded and stored in pgvector
POST /chat     →  Ask a question, get an answer with page sources
GET  /docs     →  Interactive API docs (Swagger UI)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + Python |
| LLM | Claude (claude-haiku-4-5) via Anthropic API |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | pgvector on PostgreSQL |
| Orchestration | LangGraph (StateGraph workflow) |
| Database Driver | psycopg v3 |
| Infrastructure | Docker |
| Package Manager | uv |

## Architecture

```
PDF Upload
    │
    ▼
[ingest.py]
pypdf → RecursiveCharacterTextSplitter → OpenAI Embeddings → pgvector

Question
    │
    ▼
[rag.py — LangGraph workflow]
retrieve node → similarity_search (pgvector) → generate node → Claude API → Answer + Sources
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone and install

```bash
git clone https://github.com/yeasirafgan/ai-document-assistant.git
cd ai-document-assistant
uv sync
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
ANTHROPIC_API_KEY=your-anthropic-api-key
OPENAI_API_KEY=your-openai-api-key
```

### 3. Start the database

```bash
docker compose up -d
```

### 4. Run the API

```bash
uv run uvicorn app.main:app --reload --port 8000
```

### 5. Open the interactive docs

```
http://localhost:8000/docs
```

## API Usage

### Upload a PDF

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your-document.pdf"
```

Response:
```json
{"message": "Successfully ingested 'your-document.pdf'", "chunks": 12}
```

### Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

Response:
```json
{
  "answer": "The document covers...\n\nSources: Page 2",
  "sources": [
    {"page": 2, "source": "uploads/your-document.pdf"},
    {"page": 3, "source": "uploads/your-document.pdf"}
  ]
}
```

## Project Structure

```
ai-document-assistant/
├── app/
│   ├── config.py       # Pydantic settings (API keys, model config)
│   ├── database.py     # pgvector connection and embeddings
│   ├── ingest.py       # PDF → chunks → embeddings → pgvector
│   ├── rag.py          # LangGraph workflow: retrieve → generate
│   └── main.py         # FastAPI endpoints
├── uploads/            # Uploaded PDF files (git-ignored)
├── docker-compose.yml  # PostgreSQL + pgvector
├── pyproject.toml      # Dependencies (managed by uv)
└── .env.example        # Environment variable template
```

## Key Design Decisions

- **pgvector over Pinecone** — Free, runs in the same PostgreSQL database, widely used in UK companies
- **OpenAI embeddings** — text-embedding-3-small chosen for cost-efficiency and Intel Mac compatibility
- **LangGraph StateGraph** — Explicit retrieve → generate workflow; easy to extend with reranking or query rewriting
- **Lazy vector store init** — `get_vector_store()` is a function, not a module-level object, so database connection only happens when needed
- **langchain-community avoided** — All imports use standalone packages (`langchain-openai`, `langchain-postgres`, `langchain-anthropic`) as `langchain-community` is being sunset
