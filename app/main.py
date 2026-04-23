from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

QueryStrategy = Literal["single_turn", "single_turn_tools", "multi_turn", "unsolvable"]

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import llm, places_client, rag, research_agent
from app.triage import rule_triage


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


def _combined_text_for_signals(message: str, chat_history: list[ChatTurn]) -> str:
    """Triage + RAG over initial story plus every follow-up user line (order preserved)."""
    parts: list[str] = [message.strip()]
    for turn in chat_history:
        if turn.role == "user":
            parts.append(turn.content.strip())
    return "\n\n".join(parts)[:12000]


def _trim_prior_treatment_plans(raw: list[Any]) -> list[dict[str, str]]:
    """Keep ids/titles small for the LLM payload when refining a selected plan."""
    out: list[dict[str, str]] = []
    for p in raw[:8]:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "id": str(p.get("id", "")).strip()[:64],
                "title": str(p.get("title", "")).strip()[:220],
            }
        )
    return [x for x in out if x["id"]]


def _enriched_rag_query(combined: str, travel_location: str, home_country: str) -> str:
    """Bias keyword retrieval toward trip/home when the corpus has matching tokens."""
    parts = [combined.strip()]
    tl = (travel_location or "").strip()
    hc = (home_country or "").strip()
    if tl:
        parts.append(f"Traveler trip location context: {tl}")
    if hc:
        parts.append(f"Traveler home healthcare context: {hc}")
    return "\n\n".join(parts)[:12000]


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

app = FastAPI(title="Travel Care AI", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class AssistRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    language: str = Field(default="en", max_length=32)
    location: str = Field(
        default="",
        max_length=512,
        description="City, country, or free-text place the traveler is in (optional).",
    )
    home_country: str = Field(
        default="",
        max_length=256,
        description="Traveler's home country or region for contrasting healthcare systems (optional).",
    )
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=40,
        description="Optional follow-up turns after the initial message (user/assistant).",
    )
    query_strategy: QueryStrategy = Field(
        default="single_turn",
        description="Interaction pattern: single_turn | single_turn_tools | multi_turn | unsolvable",
    )
    map_latitude: float | None = Field(
        default=None,
        ge=-90.0,
        le=90.0,
        description="Optional map pin latitude for server-side Places nearby search.",
    )
    map_longitude: float | None = Field(
        default=None,
        ge=-180.0,
        le=180.0,
        description="Optional map pin longitude for server-side Places nearby search.",
    )
    research_tools: bool = Field(
        default=True,
        description="If true, run an OpenAI tool loop (Places + Serper) before the final traveler JSON when keys exist.",
    )
    selected_treatment_plan_id: str = Field(
        default="",
        max_length=64,
        description="When set to an id from a prior response treatment_plan_options, the LLM refines cost_estimate_table for that plan.",
    )
    prior_treatment_plan_options: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=8,
        description="Echo of last assist treatment_plan_options (id+title) so refinements keep context.",
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/public-config")
async def public_config() -> dict[str, str | bool]:
    key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    maps_server = os.getenv("GOOGLE_MAPS_SERVER_KEY", "").strip()
    serper = os.getenv("SERPER_API_KEY", "").strip()
    disable_research = os.getenv("DISABLE_RESEARCH_TOOLS", "").strip().lower() in ("1", "true", "yes")
    return {
        "googleMapsApiKey": key,
        "mapsEnabled": bool(key),
        "openAiConfigured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "placesServerConfigured": bool(maps_server),
        "webSearchConfigured": bool(serper),
        "researchToolsAllowedByEnv": not disable_research,
        "defaultTravelLocation": os.getenv("DEFAULT_TRAVEL_LOCATION", "").strip(),
        "defaultHomeCountry": os.getenv("DEFAULT_HOME_COUNTRY", "").strip(),
    }


@app.post("/api/assist")
async def assist(body: AssistRequest) -> dict:
    combined = _combined_text_for_signals(body.message, body.chat_history)
    rules = rule_triage(combined)
    strategy = body.query_strategy
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    language = (body.language or "").strip() or "en"
    travel_location = (body.location or "").strip() or (
        os.getenv("DEFAULT_TRAVEL_LOCATION", "") or ""
    ).strip()
    home_country = (body.home_country or "").strip() or (
        os.getenv("DEFAULT_HOME_COUNTRY", "") or ""
    ).strip()
    rag_query = _enriched_rag_query(combined, travel_location, home_country)

    retrieval_queries_used: list[str] | None = None
    if strategy == "single_turn_tools" and openai_key:
        try:
            extra = await llm.plan_retrieval_subqueries(user_text=combined)
        except Exception:  # noqa: BLE001
            extra = []
        queries = [rag_query, combined] + [q for q in extra if q and q.strip()]
        chunks = rag.retrieve_merged(queries, k_per_query=3, max_chunks=12)
        retrieval_queries_used = queries[:8]
    else:
        chunks = rag.retrieve(rag_query, k=4)

    citations = [{"id": c.id, "title": c.title, "excerpt": c.text[:400]} for c in chunks]

    maps_server_key = os.getenv("GOOGLE_MAPS_SERVER_KEY", "").strip()
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    lat: float | None = body.map_latitude
    lng: float | None = body.map_longitude
    geocoded_travel_location = False
    if maps_server_key and travel_location and (lat is None or lng is None):
        geo = await places_client.geocode_address(travel_location, maps_server_key)
        if geo:
            lat, lng = geo
            geocoded_travel_location = True

    research_digest = ""
    research_tool_calls = 0
    research_error: str | None = None
    disable_research = os.getenv("DISABLE_RESEARCH_TOOLS", "").strip().lower() in ("1", "true", "yes")
    if (
        openai_key
        and body.research_tools
        and not disable_research
        and (maps_server_key or serper_key)
    ):
        try:
            research_digest, research_tool_calls = await research_agent.run_research_tool_loop(
                combined_user_text=combined,
                travel_location=travel_location or None,
                home_country=home_country or None,
                latitude=lat,
                longitude=lng,
                language=language,
                openai_key=openai_key,
                maps_server_key=maps_server_key,
                serper_key=serper_key,
            )
        except Exception as e:  # noqa: BLE001
            research_error = str(e)

    base: dict = {
        "query_strategy": strategy,
        "travel_location": travel_location or None,
        "home_country": home_country or None,
        "locale": travel_location or "Unspecified",
        "care_level": rules.care_level,
        "emergency": rules.emergency,
        "matched_rules": rules.matched_rules,
        "rule_rationale": rules.rationale,
        "citations": citations,
        "llm": None,
        "map_coordinates_used": (
            {"latitude": lat, "longitude": lng} if lat is not None and lng is not None else None
        ),
        "geocoded_travel_location": geocoded_travel_location,
        "disclaimer": (
            "Educational decision support only — not a diagnosis. "
            "If you may be having a medical emergency, contact local emergency services or go to the nearest "
            "appropriate emergency department, using the official numbers for your jurisdiction."
        ),
    }
    if retrieval_queries_used is not None:
        base["retrieval_queries_used"] = retrieval_queries_used
    if research_digest.strip():
        base["research_from_tools_digest"] = research_digest.strip()
    if research_tool_calls:
        base["research_tool_calls_executed"] = research_tool_calls
    if research_error:
        base["research_from_tools_error"] = research_error

    if not openai_key:
        base["llm_skip_reason"] = "missing_openai_api_key"
    else:
        try:
            llm_out, llm_prompt_meta = await llm.augment_with_openai(
                user_message=body.message,
                language=language,
                travel_location=travel_location,
                home_country=home_country,
                care_level=rules.care_level,
                emergency=rules.emergency,
                rule_rationale=rules.rationale,
                citations=citations,
                chat_history=[t.model_dump() for t in body.chat_history],
                query_strategy=strategy,
                research_from_tools_digest=research_digest.strip() or None,
                selected_treatment_plan_id=(body.selected_treatment_plan_id or "").strip() or None,
                prior_treatment_plan_options=_trim_prior_treatment_plans(body.prior_treatment_plan_options),
            )
            base["llm"] = llm_out
            if llm_prompt_meta:
                base["llm_api_prompt"] = llm_prompt_meta
        except Exception as e:  # noqa: BLE001 — surface to client for class debugging
            base["llm_error"] = str(e)

    return base


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
