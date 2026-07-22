"""Food Concierge chat router — LLM-powered with rules-based fallback.

Tools (ConciergeTools) are loaded at app startup and stored in
``app.state``, then injected via FastAPI ``Depends()`` —
no module-level globals.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.limiter import limiter
from dabba.config import get_config
from dabba.llm.food_concierge import ConciergeTools, get_concierge_response
from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

config = get_config()


def _load_concierge_tools() -> Optional[ConciergeTools]:
    """Build and return ConciergeTools from processed data.

    Called once at app startup by ``api.main``. Returns an empty
    ConciergeTools if data hasn't been generated yet.

    Returns:
        ConciergeTools instance (never None — uses empty DF as fallback).
    """
    data_path = config.data_processed_dir / "restaurants_processed.csv"
    if not data_path.exists():
        logger.warning("Processed data not found for concierge")
        return ConciergeTools(pd.DataFrame(), config=config)

    df = pd.read_csv(data_path)
    tools = ConciergeTools(df, config=config)
    logger.info("Concierge tools loaded with %d restaurants", len(df))
    return tools


def get_tools(request: Request) -> Optional[ConciergeTools]:
    """FastAPI dependency: return ConciergeTools from ``app.state``.

    Usage:
        .. code-block:: python

            @router.post(...)
            async def chat(body: ChatRequest, tools = Depends(get_tools)):
                ...

    Returns:
        ConciergeTools instance, or ``None`` if not loaded.
    """
    return getattr(request.app.state, "concierge_tools", None)


@router.post("", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    tools: Optional[ConciergeTools] = Depends(get_tools),
) -> ChatResponse:
    """Send a message to the Food Concierge.

    Args:
        request: Incoming HTTP request (required by rate limiter).
        body: ChatRequest with message and conversation history.
        tools: ConciergeTools (injected via ``Depends``).

    Returns:
        ChatResponse with the concierge's reply.
    """
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
