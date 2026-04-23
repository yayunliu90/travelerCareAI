"""Optional web search via Serper.dev (Google results JSON)."""

from __future__ import annotations

import json
from typing import Any

import httpx


async def serper_search(query: str, api_key: str, num: int = 6) -> dict[str, Any]:
    if not (query or "").strip() or not api_key:
        return {"error": "missing_query_or_api_key", "results": []}
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            content=json.dumps({"q": query.strip(), "num": min(max(num, 1), 10)}),
        )
        r.raise_for_status()
        data = r.json()
    organic = data.get("organic") or []
    out: list[dict[str, Any]] = []
    for row in organic[:num]:
        out.append(
            {
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "link": row.get("link"),
            }
        )
    return {"results": out, "query": query.strip()}
