"""Food Concierge chat router — LLM-powered with rules-based fallback."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.limiter import limiter
from dabba.config import get_config
from dabba.llm.food_concierge import ConciergeTools, get_concierge_response
from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

config = get_config()

_tools: Optional[ConciergeTools] = None
_tools_lock = threading.Lock()


def load_concierge_tools() -> None:
    """Load restaurant data for concierge tools (thread-safe)."""
    global _tools
    with _tools_lock:
        if _tools is not None:
            return
        data_path = config.data_processed_dir / "restaurants_processed.csv"
        if not data_path.exists():
            logger.warning("Processed data not found for concierge")
            _tools = ConciergeTools(pd.DataFrame(), config=config)
            return

        df = pd.read_csv(data_path)
        _tools = ConciergeTools(df, config=config)
        logger.info("Concierge tools loaded with %d restaurants", len(df))


def get_tools() -> Optional[ConciergeTools]:
    """Thread-safe accessor for concierge tools."""
    global _tools
    with _tools_lock:
        return _tools


@router.post("", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """Send a message to the Food Concierge.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        body: ChatRequest with message and conversation history.

    Returns:
        ChatResponse with the concierge's reply.
    """
    tools = get_tools()
    if tools is None:
        raise HTTPException(
            status_code=503,
            detail="Concierge tools not loaded. Run `make train` first.",
        )

    # Format conversation history
    history = []
    for msg in body.history or []:
        history.append({"role": msg.role, "content": msg.content})

    response = get_concierge_response(
        body.message,
        history,
        tools,
        config=config,
    )

    return ChatResponse(reply=response)
