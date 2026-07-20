"""Food Concierge chat router — LLM-powered with rules-based fallback.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from dabba.config import get_config
from dabba.llm.food_concierge import ConciergeTools, get_concierge_response
from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

config = get_config()

_tools: Optional[ConciergeTools] = None


def load_concierge_tools() -> None:
    """Load restaurant data for concierge tools."""
    global _tools
    data_path = config.data_processed_dir / "restaurants_processed.csv"
    if not data_path.exists():
        logger.warning("Processed data not found for concierge")
        _tools = ConciergeTools(pd.DataFrame(), config=config)
        return

    df = pd.read_csv(data_path)
    _tools = ConciergeTools(df, config=config)
    logger.info("Concierge tools loaded with %d restaurants", len(df))


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the Food Concierge.

    Args:
        request: ChatRequest with message and conversation history.

    Returns:
        ChatResponse with the concierge's reply.
    """
    if _tools is None:
        raise HTTPException(
            status_code=503,
            detail="Concierge tools not loaded. Run `make train` first.",
        )

    # Format conversation history
    history = []
    for msg in request.history or []:
        history.append({"role": msg.role, "content": msg.content})

    response = get_concierge_response(
        request.message,
        history,
        _tools,
        config=config,
    )

    return ChatResponse(reply=response)
