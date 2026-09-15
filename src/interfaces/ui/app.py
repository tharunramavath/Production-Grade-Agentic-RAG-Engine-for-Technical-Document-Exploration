import asyncio
import gradio as gr
from src.agents.workflow import AgenticRAGWorkflow
from src.config.settings import get_settings
from src.domain.schemas import IngestRequest
from src.ingestion.pipeline import IngestionPipeline
from src.services.cache import CacheService
from src.services.embeddings import EmbeddingsService
from src.services.llm import LLMService
from src.services.observability import ObservabilityService
from src.storage.database import create_database_repository
from src.storage.vector_store import create_vector_store


def create_ui():
    settings = get_settings()
    repo = create_database_repository(settings)
    vector_store = create_vector_store(settings)
    embeddings = EmbeddingsService(settings)
    llm = LLMService(settings)
    cache = CacheService(settings)
    observability = ObservabilityService(settings)

    pipeline = IngestionPipeline(settings, repo, vector_store, embeddings)
    workflow = AgenticRAGWorkflow(settings, llm, vector_store, embeddings, cache, observability)

    async def handle_ask(query: str, chat_history):
        if not query.strip():
            return "", chat_history, "", ""

        response = await workflow.execute(query)

        sources_text = "### 📚 Sources & Citations\n\n"
        if response.sources:
            for s in response.sources:
                sources_text += f"- **[{s.arxiv_id}] {s.title}** (Chunk {s.chunk_index})\n  > *{s.text_snippet[:200]}...*\n\n"
        else:
            sources_text += "_No specific paper citations used._"

        steps_text = "### 🧠 Agentic Reasoning Steps\n\n"
        for step in response.reasoning_steps:
            steps_text += f"- {step}\n"
        steps_text += f"\n**Execution Time**: `{response.execution_time:.2f}s` | **Guardrail Score**: `{response.guardrail_score}/100`"

        chat_history = chat_history or []
        chat_history.append((query, response.answer))
        return "", chat_history, sources_text, steps_text

    async def handle_ingest(category_query: str, max_papers: int):
        req = IngestRequest(query=category_query, max_results=int(max_papers), download_pdf=True)
        res = await pipeline.run(req)
        return f"Status: {res.message}\nPapers in DB: {repo.count()} | Chunks in Vector Store: {vector_store.count()}"

    async def handle_seed():
        res = await pipeline.ingest_sample_papers()
        return f"Status: {res.message}\nPapers in DB: {repo.count()} | Chunks in Vector Store: {vector_store.count()}"

    async def handle_upload_pdf(files):
        if not files:
            return "Please select or drop at least one PDF file."
        file_list = files if isinstance(files, list) else [files]

        processed = 0
        total_chunks = 0
        errors = 0
        details = []

        for f in file_list:
            file_path = f.name if hasattr(f, "name") else str(f)
            res = await pipeline.ingest_pdf_file(file_path)
            if res.errors == 0:
                processed += 1
                total_chunks += res.chunks_indexed
                details.append(f"✓ {res.message}")
            else:
                errors += 1
                details.append(f"✗ {res.message}")

        summary = f"Processed {processed}/{len(file_list)} PDFs ({total_chunks} chunks indexed, {errors} errors).\n" + "\n".join(details)
        return f"Status: {summary}\nPapers in DB: {repo.count()} | Chunks in Vector Store: {vector_store.count()}"

    async def handle_explore_chunks(search_term: str, top_k_val: int, mode_val: str):
        term = search_term.strip() or "transformer attention reasoning model"
        if mode_val == "bm25":
            hits = vector_store.search_bm25(term, top_k=int(top_k_val))
        elif mode_val == "knn":
            vec = await embeddings.embed_query(term)
            hits = vector_store.search_knn(vec, top_k=int(top_k_val))
        else:  # hybrid
            vec = await embeddings.embed_query(term)
            hits = vector_store.search_hybrid(term, vec, top_k=int(top_k_val))

        if not hits:
            return "### ⚠️ No chunks found matching your search term in OpenSearch."

        output_md = f"### 🧩 Found {len(hits)} Individual Document Chunks in OpenSearch\n\n"
        for idx, h in enumerate(hits, 1):
            section_name = h.section_title or "General"
            output_md += f"#### **{idx}. [{h.arxiv_id}] {h.title}**\n"
            output_md += f"- **Chunk Index**: `#{h.chunk_index}` | **Section**: `{section_name}` | **Relevance Score**: `{h.score:.4f}`\n\n"
            output_md += f"```text\n{h.chunk_text}\n```\n\n---\n\n"
        return output_md

    with gr.Blocks(title="arXiv Agentic Research Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 arXiv Agentic Research Assistant")
        gr.Markdown("Self-correcting, citation-grounded RAG over Computer Science & AI research papers.")

        with gr.Tab("💬 Research Chat"):
            chatbot = gr.Chatbot(height=480, type="tuples")
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask a question about AI/CS research papers (e.g. 'How does self-attention work in Transformers?')...", scale=9)
                send = gr.Button("Ask", variant="primary", scale=1)

            with gr.Row():
                sources_box = gr.Markdown("### 📚 Sources will appear here")
                steps_box = gr.Markdown("### 🧠 Reasoning trace will appear here")

            send.click(handle_ask, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_box, steps_box])
            msg.submit(handle_ask, inputs=[msg, chatbot], outputs=[msg, chatbot, sources_box, steps_box])

        with gr.Tab("📥 Ingest Papers"):
            gr.Markdown("### 🌐 Method 1: Fetch Live from arXiv")
            with gr.Row():
                ingest_query = gr.Textbox(label="arXiv Search Query", value="cat:cs.AI")
                ingest_limit = gr.Slider(minimum=1, maximum=20, value=2, step=1, label="Max Papers")
            ingest_btn = gr.Button("Fetch & Index from arXiv", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### ⚡ Method 2: Seed Foundational AI Benchmark Papers")
            gr.Markdown("Instantly index seminal papers (*Attention Is All You Need*, *RAG*, *Chain-of-Thought*) into OpenSearch without relying on arXiv API rate limits.")
            seed_btn = gr.Button("⚡ Index Benchmark AI Papers", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("### 📄 Method 3: Upload Local Research Paper PDF(s)")
            pdf_upload = gr.File(label="Upload Research Paper PDF(s) (Multiple Files Supported)", file_types=[".pdf"], file_count="multiple")
            upload_btn = gr.Button("Parse & Index Uploaded PDF(s)", variant="primary")

            gr.Markdown("---")
            ingest_output = gr.Textbox(label="Ingestion Progress & Storage Stats", lines=6)

            ingest_btn.click(handle_ingest, inputs=[ingest_query, ingest_limit], outputs=[ingest_output])
            seed_btn.click(handle_seed, outputs=[ingest_output])
            upload_btn.click(handle_upload_pdf, inputs=[pdf_upload], outputs=[ingest_output])

        with gr.Tab("🔍 Chunk Explorer"):
            gr.Markdown("### Inspect Individual Document Chunks Stored in OpenSearch")
            with gr.Row():
                chunk_query = gr.Textbox(label="Filter / Search Chunks by Keyword or Topic", value="attention matrix", scale=7)
                chunk_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of Chunks", scale=2)
                chunk_mode = gr.Dropdown(choices=["hybrid", "bm25", "knn"], value="hybrid", label="Search Mode", scale=2)
                explore_btn = gr.Button("🔍 Fetch Chunks", variant="primary", scale=1)

            chunks_display = gr.Markdown("### Click 'Fetch Chunks' to inspect individual chunk contents, indices, and metadata.")
            explore_btn.click(handle_explore_chunks, inputs=[chunk_query, chunk_k, chunk_mode], outputs=[chunks_display])

    return demo


def launch_ui(server_name="127.0.0.1", server_port=7860):
    ui = create_ui()
    ui.launch(server_name=server_name, server_port=server_port, share=False)
