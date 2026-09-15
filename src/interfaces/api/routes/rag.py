from fastapi import APIRouter, Depends, HTTPException
from src.agents.workflow import AgenticRAGWorkflow
from src.domain.schemas import (
    AgenticAskResponse,
    AskRequest,
    FeedbackRequest,
    FeedbackResponse,
    IngestRequest,
    IngestResponse,
)
from src.ingestion.pipeline import IngestionPipeline
from src.interfaces.api.dependencies import (
    get_agentic_workflow,
    get_ingestion_pipeline,
    get_observability_service,
)
from src.services.observability import ObservabilityService

router = APIRouter(tags=["Agentic RAG"])


@router.post("/ask", response_model=AgenticAskResponse)
async def ask_agentic(
    request: AskRequest,
    workflow: AgenticRAGWorkflow = Depends(get_agentic_workflow),
) -> AgenticAskResponse:
    """Ask a question using self-correcting LangGraph Agentic RAG."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")
    try:
        return await workflow.execute(query=request.query, user_id=request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic RAG execution error: {str(e)}")


@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(
    request: IngestRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> IngestResponse:
    """Programmatically trigger paper ingestion without Airflow."""
    try:
        return await pipeline.run(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    observability: ObservabilityService = Depends(get_observability_service),
) -> FeedbackResponse:
    """Submit rating or feedback for an answer trace in Langfuse."""
    success = observability.submit_feedback(
        trace_id=request.trace_id,
        score=request.score,
        comment=request.comment,
    )
    return FeedbackResponse(
        success=success,
        message="Feedback recorded successfully" if success else "Observability disabled or record failed",
    )
