# 📖 Complete User & Developer Guide: Autonomous arXiv Research Assistant

This guide covers **everything you need to run, configure, and understand** the Autonomous arXiv Research Assistant.

---

## 📑 Table of Contents
1. [Virtual Environment Activation](#-1-virtual-environment-activation)
2. [Environment Configuration (.env)](#-2-environment-configuration-env)
3. [Gradio Web Interface (Primary UI)](#-3-gradio-web-interface-primary-ui)
4. [Command-Line Interface (CLI)](#-4-command-line-interface-cli)
5. [FastAPI REST API (/docs)](#-5-fastapi-rest-api-docs)
6. [Telegram Bot (Mobile Access)](#-6-telegram-bot-mobile-access)
7. [Which File is for What? (Project Architecture Map)](#-7-which-file-is-for-what-project-architecture-map)
8. [Agentic Reasoning Engine & Workflow Architecture](#-8-agentic-reasoning-engine--workflow-architecture)
9. [Aiven OpenSearch Cluster Management](#-9-aiven-opensearch-cluster-management)
10. [Troubleshooting & FAQs](#-10-troubleshooting--faqs)

---

## 🐍 1. Virtual Environment Activation

Before running any commands, activate the Python virtual environment so all required libraries (`fastapi`, `langchain-groq`, `opensearch-py`, `gradio`, `docling`, etc.) are accessible.

### On Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1

# (If PowerShell blocks scripts, run this once: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
```

### On Windows (Command Prompt / CMD):
```cmd
.venv\Scripts\activate.bat
```

### On macOS / Linux:
```bash
source .venv/bin/activate
```

> **💡 Shortcut with UV:**  
> If you have `uv` installed, you can prefix any command with `uv run` to execute inside the virtual environment without manual activation:
> ```bash
> uv run python -m src.interfaces.cli.main ui --port 7860
> ```

---

## ⚙️ 2. Environment Configuration (`.env`)

All active settings live inside `agentic_research_assistant/.env`. Ensure the following variables are present:

```ini
# Relational Metadata Storage
DATABASE_TYPE=sqlite
DATABASE_URL=sqlite:///./data/metadata.db

# Vector Storage Engine
VECTOR_STORE_TYPE=opensearch
OPENSEARCH_HOST=https://your-user:your-password@your-opensearch-cluster.aivencloud.com:14442
OPENSEARCH_INDEX_NAME=arxiv_papers_hybrid
OPENSEARCH_VERIFY_CERTS=true

# LLM Provider (Groq Cloud - Fast Inference)
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.0

# Dense Vector Embeddings (Jina Cloud Embeddings v3 - 1024 dimensions)
JINA_API_KEY=your_jina_api_key_here
EMBEDDING_MODEL=jina-embeddings-v3
EMBEDDING_DIMENSION=1024

# Caching Layer (In-Memory or Redis)
CACHE_TYPE=memory
CACHE_TTL_SECONDS=3600

# Langfuse Observability & Tracing (Optional)
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com

# Telegram Bot Interface (Optional)
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

> **🔑 Get a Free Groq API Key**: Visit [console.groq.com/keys](https://console.groq.com/keys) (Takes 30 seconds).  
> **🔑 Get a Free Jina API Key**: Visit [jina.ai/embeddings](https://jina.ai/embeddings/).

---

## 🌐 3. Gradio Web Interface (Primary UI)

The Gradio Web UI provides an interactive visual dashboard for paper ingestion, multi-PDF document uploads, conversational research chat, and step-by-step reasoning transparency.

### Launch Command:
```bash
python -m src.interfaces.cli.main ui --port 7860
```
Open your browser at: **`http://localhost:7860`**

---

### Step-by-Step Walkthrough:

#### Step A: Ingest Papers (Tab: "📥 Ingest Papers")
1. Navigate to the **"📥 Ingest Papers"** tab.
2. Choose your ingestion method:
   - **Fetch from arXiv API**: Enter a topic or category query (e.g., `cat:cs.AI OR cat:cs.LG`, `chain of thought reasoning`, `mixture of experts routing`) and select the paper limit slider.
   - **Upload Local PDFs**: Upload one or multiple research PDF files directly using the file uploader.
3. Click **"Start Ingestion"** or **"Process Uploaded PDFs"**.
4. The status log updates in real time showing the number of papers fetched, parsed, and chunks indexed into the vector store.

#### Step B: Chat with Citations (Tab: "💬 Research Chat")
1. Switch to the **"💬 Research Chat"** tab.
2. Type your question in the prompt box and hit **Enter** (or click **"Ask"**).

#### 💡 Realistic Query Examples to Try:
- *"How does chain-of-thought monitoring detect evasion in large language models?"*
- *"What are the architectural differences between dense transformers and sparse Mixture of Experts?"*
- *"What loss functions and regularization techniques are used in diffusion model training?"*

#### What the UI Displays:
- **Chat Window**: The final synthesized answer grounded in the indexed papers.
- **📚 Sources Box**: The exact papers cited with arXiv IDs, titles, section headings, and excerpt previews.
- **🧠 Reasoning Trace Box**: Shows the agent's internal thought process:
  - Guardrail Domain Score (e.g. `95/100`).
  - Retrieval attempts made.
  - Document relevance grading verdict (`Relevant` or `Irrelevant`).
  - Total round-trip execution latency.

---

## 💻 4. Command-Line Interface (CLI)

The CLI lets you manage ingestion, search, and research queries directly in your terminal.

```bash
# Display help & all available commands
python -m src.interfaces.cli.main --help
```

### Ingestion Commands
```bash
# Ingest 5 papers in CS.AI
python -m src.interfaces.cli.main ingest --query "cat:cs.AI" --limit 5

# Ingest 3 papers on a specific keyword
python -m src.interfaces.cli.main ingest --query "reinforcement learning from human feedback" --limit 3

# Fast metadata-only indexing (skips full PDF downloads)
python -m src.interfaces.cli.main ingest --query "cat:cs.CL" --limit 5 --no-download
```

### Search Commands
```bash
# Hybrid Search (Keyword + Dense Vector Fusion - Recommended)
python -m src.interfaces.cli.main search "attention mechanism" --top-k 3

# BM25 Keyword Search Only
python -m src.interfaces.cli.main search "LoRA fine-tuning" --mode bm25 --top-k 3

# k-NN Dense Vector Search Only
python -m src.interfaces.cli.main search "diffusion probabilistic models" --mode knn --top-k 3
```

### Agentic Q&A Command
```bash
python -m src.interfaces.cli.main ask "How does chain-of-thought monitoring prevent AI misalignment?"
```

---

## 🔌 5. FastAPI REST API (`/docs`)

Start the production backend server to connect the assistant to custom apps, dashboards, or webhooks.

### Launch Command:
```bash
python -m src.interfaces.cli.main serve --port 8000
```
Open your browser at: **`http://localhost:8000/docs`** for interactive Swagger documentation.

### Core API Endpoints:

#### 1. `POST /api/v1/ask` (Agentic Question Answering)
```bash
curl -X POST "http://localhost:8000/api/v1/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is chain of thought monitoring?", "top_k": 5, "use_hybrid": true}'
```

#### 2. `POST /api/v1/search` (Hybrid Search)
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "transformer attention", "top_k": 3, "mode": "hybrid"}'
```

#### 3. `POST /api/v1/ingest` (Trigger Ingestion)
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{"query": "cat:cs.AI", "max_results": 5, "download_pdf": true}'
```

#### 4. `GET /api/v1/health` (Health Status)
```bash
curl "http://localhost:8000/api/v1/health"
```

---

## 📱 6. Telegram Bot (Mobile Access)

Interact with your research assistant from your phone via Telegram.

### 1. Get a Bot Token:
1. Open Telegram and search for `@BotFather`.
2. Send `/newbot` and follow the prompts to choose a name and username.
3. Copy the HTTP API token provided by BotFather.

### 2. Configure in `.env`:
```ini
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 3. Start the Bot:
```bash
python -m src.interfaces.bot.bot
```
*(Runs via long-polling on localhost — no public IP or ngrok required!)*

---

## 📁 7. Which File is for What? (Project Architecture Map)

Here is a complete reference of every file and folder in the project:

```text
agentic_research_assistant/
├── .env                              # Active environment variables & API keys (gitignored)
├── .env.example                      # Sanitized environment template
├── .gitignore                        # Git exclusion rules
├── pyproject.toml                    # Dependencies & build configuration
├── USER_GUIDE.md                     # Complete user & developer manual
├── README.md                         # Project overview & architectural summary
├── data/                             # Local persistence
│   ├── downloads/                    # Downloaded arXiv paper PDFs
│   ├── metadata.db                   # SQLite relational database
│   └── vector_store.json             # Local fallback vector store
│
└── src/
    ├── __init__.py                   # Top-level package initializer
    │
    ├── config/                       # Application Configuration
    │   ├── __init__.py
    │   └── settings.py               # Pydantic v2 Settings (parses .env & validates types)
    │
    ├── domain/                       # Core Entities & Data Contracts
    │   ├── __init__.py
    │   ├── models.py                 # Entities: Paper, DocumentChunk, SearchHit
    │   └── schemas.py                # DTOs: AskRequest, SearchRequest, IngestRequest, etc.
    │
    ├── storage/                      # Storage Implementations
    │   ├── database/
    │   │   ├── base.py               # Abstract BasePaperRepository interface
    │   │   ├── sqlite_repo.py        # SQLite repository via SQLAlchemy
    │   │   └── factory.py            # Database factory selector
    │   └── vector_store/
    │       ├── base.py               # Abstract BaseVectorStore interface
    │       ├── opensearch_store.py   # OpenSearch Hybrid & k-NN vector store
    │       ├── memory_store.py       # Local persistent vector store fallback
    │       └── factory.py            # Vector store factory with auto-fallback
    │
    ├── ingestion/                    # Automated Ingestion Pipeline
    │   ├── __init__.py
    │   ├── arxiv_client.py           # Queries arXiv API & downloads PDFs with retries
    │   ├── pdf_parser.py             # Parses PDFs to Markdown using IBM Docling
    │   ├── chunker.py                # Section-aware text chunker with overlap
    │   └── pipeline.py               # Ingestion coordinator & batch indexer
    │
    ├── services/                     # External Provider Wrappers
    │   ├── __init__.py
    │   ├── llm.py                    # Unified LLM provider (Groq / Ollama / OpenAI)
    │   ├── embeddings.py             # Jina Embeddings API with deterministic fallback
    │   ├── cache.py                  # In-Memory & Redis exact-match caching
    │   └── observability.py          # Langfuse v3 SDK tracing & feedback manager
    │
    ├── agents/                       # LangGraph Agentic Reasoning Workflow
    │   ├── __init__.py
    │   ├── state.py                  # AgentState schema & structured outputs
    │   ├── prompts.py                # Prompts for Guardrail, Grading, Rewriting, Answering
    │   ├── workflow.py               # Compiled LangGraph StateGraph pipeline
    │   └── nodes/
    │       ├── guardrail.py          # Evaluates domain scope (0-100 score)
    │       ├── retrieve.py           # Executes OpenSearch hybrid retrieval
    │       ├── document_grader.py    # LLM-as-a-Judge document relevance grading
    │       ├── query_rewriter.py     # Refines search query if results are poor
    │       └── answer_generator.py   # Synthesizes final citation-grounded answer
    │
    └── interfaces/                   # User & Developer Entrypoints
        ├── __init__.py
        ├── cli/
        │   ├── __init__.py
        │   ├── commands.py           # Typer CLI subcommands (ingest, search, ask, serve, ui)
        │   └── main.py               # CLI entrypoint script
        ├── api/
        │   ├── app.py                # FastAPI application factory & CORS setup
        │   ├── dependencies.py       # Dependency injection providers
        │   └── routes/               # API route handlers (health, search, rag)
        ├── ui/
        │   └── app.py                # Gradio web interface application
        └── bot/
            └── bot.py                # Telegram bot service
```

---

## 🧠 8. Agentic Reasoning Engine & Workflow Architecture

The core of the assistant is an **Agentic / Corrective RAG (CRAG)** reasoning engine built using **LangGraph** (`StateGraph`), orchestrated in [`src/agents/workflow.py`](./src/agents/workflow.py).

Unlike traditional naive RAG pipelines that execute a fixed `Retrieve -> Generate` chain, this system uses **stateful evaluation, automated domain guardrails, and self-correcting query reformulation loops**.

### 🔄 Agent Workflow Graph

```mermaid
flowchart TD
    Start([User Query]) --> GuardrailNode[1. Guardrail Node]
    
    GuardrailNode -- Out of Scope (Score < 50) --> AnswerNode[5. Answer Generator Node]
    GuardrailNode -- In Scope (Score >= 50) --> RetrieveNode[2. Retrieve Node]
    
    RetrieveNode --> GradeNode[3. Document Grader Node]
    
    GradeNode -- Documents Relevant (Yes) --> AnswerNode
    GradeNode -- Documents Irrelevant & Attempts < Max --> RewriteNode[4. Query Rewriter Node]
    GradeNode -- Max Attempts Reached --> AnswerNode
    
    RewriteNode --> RetrieveNode
    AnswerNode --> End([Final Response + Citations])
```

### 🧩 The 5 Agent Nodes

| Node | Implementation File | Functionality & Logic |
| :--- | :--- | :--- |
| **1. Guardrail** | [`src/agents/nodes/guardrail.py`](./src/agents/nodes/guardrail.py) | **Domain Scope Gate**: Evaluates if the query relates to Computer Science / AI research (scores 0–100). If below threshold (`50`), routes directly to answer generation for a graceful refusal. |
| **2. Retrieve** | [`src/agents/nodes/retrieve.py`](./src/agents/nodes/retrieve.py) | **Hybrid Search**: Queries OpenSearch using reciprocal rank fusion between BM25 keyword search and Jina Cloud dense vector embeddings (`jina-embeddings-v3`). |
| **3. Document Grader** | [`src/agents/nodes/document_grader.py`](./src/agents/nodes/document_grader.py) | **Self-Reflection (LLM-as-a-Judge)**: Analyzes retrieved passages against the question to verify if the context actually contains relevant evidence (`yes`/`no`). |
| **4. Query Rewriter** | [`src/agents/nodes/query_rewriter.py`](./src/agents/nodes/query_rewriter.py) | **Adaptive Query Expansion**: If grading fails, reformulates the user query into optimized semantic search terms and loops back to `Retrieve` (up to `max_retrieval_attempts = 2`). |
| **5. Answer Generator** | [`src/agents/nodes/answer_generator.py`](./src/agents/nodes/answer_generator.py) | **Evidence-Grounded Synthesis**: Synthesizes the final answer citing paper titles, URLs, and arXiv IDs. |

### 📊 Agent State Management (`AgentState`)
Defined in [`src/agents/state.py`](./src/agents/state.py), the state object passes across all nodes:
- `original_query` & `rewritten_query`
- `retrieval_attempts` (loop counter)
- `guardrail_score` & `is_in_scope`
- `documents_relevant` (grader decision)
- `sources` & `retrieved_context`
- `reasoning_steps` (transparent execution trace)
- `final_answer` & `trace_id` (Langfuse integration)

---

## 💾 9. Aiven OpenSearch Cluster Management & Visual Dashboards

When using OpenSearch on Aiven or self-hosted infrastructure, vector indices scale efficiently.

### 📊 Capacity Math & Sizing:
- **1 Document Chunk**: ~500 words of text (~3 KB).
- **1 Vector Embedding (1024 dimensions)**: ~4 KB.
- **1 Average 10-Page Paper**: Generates ~10–25 chunks (~80–180 KB in OpenSearch).
- **A Standard 20 GB Cluster**: Can comfortably hold **over 100,000+ chunks (~5,000 to 10,000 full academic papers)**!

---

### 🖥️ Aiven Console & OpenSearch Dashboards Step-by-Step

#### 1. Accessing OpenSearch in the Aiven Console
1. Log in to [console.aiven.io](https://console.aiven.io).
2. Click on your active project and select your **OpenSearch** service.
3. On the **Overview** page, note:
   - **Service URI**: Your full connection string containing host, port (`14442`), and `avnadmin` credentials.
   - **OpenSearch Dashboards**: Click the direct **"OpenSearch Dashboards URL"** link.

#### 2. Logging into OpenSearch Dashboards
1. Open the Dashboards URL in your browser.
2. Log in using:
   - **Username**: `avnadmin`
   - **Password**: *(Found in your Aiven Console under Service Overview)*.

#### 3. Creating the Index Pattern in OpenSearch Dashboards
To explore and query your indexed paper chunks in the **Discover** tab:

1. Click the **Hamburger Menu (☰)** on the top-left $\rightarrow$ Scroll down to **Management** $\rightarrow$ Click **Stack Management**.
2. In the left sidebar under *OpenSearch Dashboards*, click **Index Patterns**.
3. Click the blue **"Create index pattern"** button.
4. **Step 1 of 2: Define Pattern**:
   - In the *Index pattern name* text box, type: `arxiv_papers_hybrid*`
   - It will display a matching index: `arxiv_papers_hybrid` (1 index matched).
   - Click **Next step**.
5. **Step 2 of 2: Configure Settings (⚠️ CRITICAL)**:
   - In the **Time field** dropdown: Select **"I don't want to use the time filter"**.
   - > **⚠️ Why do NOT select a time field?**  
     > If you assign a time field (like `published_date` or `@timestamp`), OpenSearch Dashboards defaults to filtering documents from the *Last 15 minutes*. Because academic papers carry original publication dates from previous years, **the Discover page will look completely empty** unless you manually change the global time filter to *Last 10 years*. Selecting *"I don't want to use the time filter"* ensures all indexed paper chunks are always visible immediately!
6. Click **"Create index pattern"**.

#### 4. Exploring Indexed Papers & Dense Vectors
1. Click the **Hamburger Menu (☰)** $\rightarrow$ **Discover**.
2. Select `arxiv_papers_hybrid*` from the index pattern selector dropdown.
3. You will see all indexed documents with their structured fields:
   - `title`: Academic paper title (e.g., *Attention Is All You Need*).
   - `arxiv_id`: Paper identifier (e.g., `1706.03762`).
   - `chunk_text`: Extracted content passage.
   - `section_title`: Section heading (e.g., `Abstract`, `Model Architecture`).
   - `embedding`: 1024-dimensional dense vector array generated by Jina AI.

#### 5. Using Dev Tools for Direct Queries
1. Click the **Hamburger Menu (☰)** $\rightarrow$ **Management** $\rightarrow$ **Dev Tools**.
2. Run standard OpenSearch REST commands to inspect your cluster directly:
   ```json
   // Check total indexed chunk count
   GET arxiv_papers_hybrid/_count

   // Check index settings and k-NN vector mapping
   GET arxiv_papers_hybrid/_mapping

   // Inspect the first 3 documents
   GET arxiv_papers_hybrid/_search
   {
     "size": 3
   }
   ```

---

### Best Practices:
1. **Batch Ingestions**: Ingest papers in focused batches (e.g. `--limit 5` or `--limit 10`) on specific topics you are currently researching.
2. **Check Cluster Health**:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```
3. **Local PDF Space**: If your disk fills up, you can safely delete PDFs in `data/downloads/`. All search text and vectors are already safely stored in your cloud vector store.

---

## ❓ 10. Troubleshooting & FAQs

| Issue | Solution |
|---|---|
| **PowerShell Script Execution Error** | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in PowerShell, then reactivate the venv. |
| **arXiv Rate Limit (HTTP 429)** | The client automatically backs off and retries. Ingest in smaller batches (e.g. `--limit 5`). |
| **OpenSearch Connection Fails** | Verify `OPENSEARCH_HOST` starts with `https://` in `.env` and `OPENSEARCH_VERIFY_CERTS=true`. |
| **No Answer / Blank Result** | Ensure `GROQ_API_KEY` is set in `.env` with a valid key from [console.groq.com](https://console.groq.com/keys). |
| **Port 7860 or 8000 Already in Use** | Specify a different port: `python -m src.interfaces.cli.main ui --port 7865`. |
