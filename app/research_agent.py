"""OpenAI tool-calling loop: Google Places (nearby + review snippets) + Serper web search."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app import places_client, web_search

_PLACES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_nearby_medical_places",
        "description": (
            "Find nearby medical facilities using Google Places (ratings, review snippets, "
            "approximate straight-line distance in meters from the search origin). "
            "Does not provide reliable medical prices—use web_search for cost ballparks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Search origin latitude"},
                "longitude": {"type": "number", "description": "Search origin longitude"},
                "radius_meters": {
                    "type": "integer",
                    "description": "Search radius 200–50000",
                    "default": 4500,
                },
                "place_type": {
                    "type": "string",
                    "description": "Places type, e.g. hospital, doctor, pharmacy, dentist",
                    "default": "hospital",
                },
                "keyword": {
                    "type": "string",
                    "description": "Extra keyword e.g. emergency, urgent care, 24 hour",
                    "default": "",
                },
                "include_review_snippets": {
                    "type": "boolean",
                    "default": True,
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
}

_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web. **Prioritize official local or regional government and public-health sources** "
            "when travel_location is known: city/county/state health departments, ministry of health pages, "
            "national health-service visitor guides, official .gov (or equivalent) portals explaining how care works, "
            "emergency access, pharmacies, and visitor orientation. Include the destination name in queries "
            "(e.g. \"NYC Department of Health\", \"New York State health insurance visitor\"). "
            "Also usable for typical costs or clinic types—treat costs as uncertain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused English search query"},
            },
            "required": ["query"],
        },
    },
}


def _build_tools(*, maps_key: str, serper_key: str, has_coords: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if serper_key.strip():
        tools.append(_WEB_TOOL)
    if maps_key.strip() and has_coords:
        tools.append(_PLACES_TOOL)
    return tools


async def _dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    maps_key: str,
    serper_key: str,
    default_lat: float | None,
    default_lng: float | None,
) -> dict[str, Any]:
    if name == "search_nearby_medical_places":
        lat = arguments.get("latitude", default_lat)
        lng = arguments.get("longitude", default_lng)
        try:
            lat_f = float(lat) if lat is not None else None
            lng_f = float(lng) if lng is not None else None
        except (TypeError, ValueError):
            return {"error": "invalid_lat_lng"}
        if lat_f is None or lng_f is None:
            return {
                "error": (
                    "latitude and longitude required. Send map_latitude/map_longitude from the client "
                    "or ensure travel_location geocodes on the server."
                ),
            }
        if not maps_key.strip():
            return {"error": "GOOGLE_MAPS_SERVER_KEY not configured"}
        return await places_client.nearby_medical_places(
            latitude=lat_f,
            longitude=lng_f,
            api_key=maps_key,
            radius_meters=int(arguments.get("radius_meters") or 4500),
            place_type=str(arguments.get("place_type") or "hospital"),
            keyword=str(arguments.get("keyword") or ""),
            include_review_snippets=bool(arguments.get("include_review_snippets", True)),
        )
    if name == "web_search":
        q = str(arguments.get("query") or "").strip()
        if not q:
            return {"error": "empty_query"}
        if not serper_key.strip():
            return {"error": "SERPER_API_KEY not configured"}
        return await web_search.serper_search(q, serper_key)
    return {"error": f"unknown_tool:{name}"}


async def run_research_tool_loop(
    *,
    combined_user_text: str,
    travel_location: str | None,
    home_country: str | None,
    latitude: float | None,
    longitude: float | None,
    language: str,
    openai_key: str,
    maps_server_key: str,
    serper_key: str,
    model: str | None = None,
    max_rounds: int = 6,
    max_tool_calls: int = 10,
) -> tuple[str, int]:
    """
    Returns (research_digest_text, number_of_tool_messages_appended).
    Digest is plain text for downstream traveler JSON generation.
    """
    has_coords = latitude is not None and longitude is not None
    tools = _build_tools(maps_key=maps_server_key, serper_key=serper_key, has_coords=has_coords)
    if not tools:
        return "", 0

    model_name = (model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")).strip()
    sys = """You are a research sub-agent for travel healthcare orientation (educational, not clinical diagnosis).

You may call:
- search_nearby_medical_places when coordinates are available: use returned ratings, review snippets, and straight_line_distance_m (meters from origin; NOT driving time). Google price_level is mostly for dining—do not treat it as medical cost.
- web_search for **official government / public-health pages first**, then other reputable pages. Always treat costs from the web as uncertain and unverified.

Official local healthcare system (mandatory when travel_location is non-empty):
- Run **at least one** web_search whose query explicitly targets that jurisdiction's **government or public-health authority** guidance on how healthcare works for people in that area (access, emergency/urgent vs primary care, pharmacies, visitor or uninsured orientation where applicable). Prefer queries that surface city, state/province, or national ministry sites over blogs.
- In the digest, include a short subsection (2–4 bullets) summarizing **only what those official or high-trust pages/snippets support**—how a traveler can learn more on government sites, without inventing phone numbers or URLs not present in tool results.
- If official pages do not appear in results, say so and summarize the best-available neutral sources with clear uncertainty.

Typical out-of-pocket / self-pay cost signals (when travel_location is set):
- Run **at least one** web_search aimed at **ballpark** visit costs for the destination (e.g. “urgent care visit self pay <city>”, “ER typical cost uninsured <region>”, “walk-in clinic price range <country> tourist”) — prioritize neutral publishers, hospital price transparency pages, or government consumer health pages. Summarize only what snippets support; treat all figures as **uncertain** and jurisdiction-dependent.
- When the traveler case suggests **mild** symptoms suitable for pharmacy/self-care (and not an emergency narrative), consider one web_search for the destination’s **official consumer medicines / OTC** orientation (e.g. national regulator “medicines and you”, pharmacy licensing) so downstream answers can cite **real** https links when found — never invent URLs.

Rules:
- When travel_location is set: prefer **3–5** web queries (official health system + cost ballparks as above + other case-relevant topics) plus at most 1–2 nearby searches when coordinates exist. When it is missing, 1–3 web queries are enough.
- When you have enough to help a downstream assistant, STOP calling tools and output a single plain-text digest (English is fine) with bullet points: (1) official/local-system orientation, (2) **Cost signals (uncertain)** from web snippets if any, (3) key facilities if Places ran (name, distance_m if known, rating), short review takeaways, (4) other web findings with titles/links where useful. Max about 2500 characters.
- Do not invent tool results; only summarize tool outputs."""

    loc_set = bool((travel_location or "").strip())
    user_intro = {
        "traveler_case_excerpt": combined_user_text[:4000],
        "travel_location": travel_location,
        "traveler_home_country": home_country,
        "resolved_coordinates": {"latitude": latitude, "longitude": longitude} if has_coords else None,
        "output_language_for_final_app": language,
        "official_local_government_health_research": loc_set,
        "instructions": (
            "Call tools as needed, then produce the plain-text digest only (no JSON). "
            + (
                "Because travel_location is set, include government/public-health-oriented web_search "
                "queries for that place so the traveler can learn how the local system is described officially, "
                "plus at least one query aimed at typical uninsured/self-pay **ballpark** costs for urgent care vs ER "
                "vs clinic in that area (summarize only what results show)."
                if loc_set
                else "Use web_search for reputable orientation; there is no specific trip place—stay general."
            )
        ),
    }

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user_intro, ensure_ascii=False)},
    ]

    tool_calls_used = 0
    rounds = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        while rounds < max_rounds:
            rounds += 1
            payload: dict[str, Any] = {
                "model": model_name,
                "temperature": 0.2,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]
            messages.append(msg)

            tcalls = msg.get("tool_calls") or []
            if not tcalls:
                digest = (msg.get("content") or "").strip()
                return digest, tool_calls_used

            for tc in tcalls:
                if tool_calls_used >= max_tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps({"error": "tool_call_budget_exceeded"}),
                        }
                    )
                    continue
                tool_calls_used += 1
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await _dispatch(
                    name,
                    args,
                    maps_key=maps_server_key,
                    serper_key=serper_key,
                    default_lat=latitude,
                    default_lng=longitude,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False)[:12000],
                    }
                )

    return (
        "Research stopped at the step limit — use map pin + official local sources to verify facilities and costs.",
        tool_calls_used,
    )
