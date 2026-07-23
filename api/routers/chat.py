"""Food Concierge chat router — LLM-powered with rules-based fallback.

Tools (ConciergeTools) are loaded at app startup and stored in
``app.state``, then injected via FastAPI ``Depends()`` —
no module-level globals.

Data is loaded from the database via repositories (not CSVs) — the
CSV→DB migration ensures the serving path never reads raw CSV files.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.limiter import limiter
from dabba.config import get_config
from dabba.database.repositories import get_all_restaurants_as_df
from dabba.llm.food_concierge import ConciergeTools, get_concierge_response
from api.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

config = get_config()


def _load_concierge_tools(
    eta_model: Any = None,
) -> Optional[ConciergeTools]:
    """Build and return ConciergeTools from the database.

    Called once at app startup by ``api.main``. Uses the repository
    layer to read restaurant data from Postgres/SQLite (not CSVs)
    and passes it to ConciergeTools for search/ETA/reliability tools.

    Returns an empty ConciergeTools if the database has no data yet.

    Args:
        eta_model: The loaded ETA model pipeline (optional).

    Returns:
        ConciergeTools instance (never None — uses empty DF as fallback).
    """
    from dabba.database.session import get_db

    try:
        with get_db() as db:
            df = get_all_restaurants_as_df(db, with_cuisine_features=False)
    except Exception as e:
        logger.warning("Failed to load restaurant data from DB: %s", e)
        df = pd.DataFrame()

    if df.empty:
        logger.warning("No restaurant data found in database for concierge")
        return ConciergeTools(pd.DataFrame(), eta_model=eta_model, config=config)

    tools = ConciergeTools(df, eta_model=eta_model, config=config)
    logger.info(
        "Concierge tools loaded from DB with %d restaurants (ETA model: %s)",
        len(df),
        "loaded" if eta_model is not None else "missing",
    )
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
