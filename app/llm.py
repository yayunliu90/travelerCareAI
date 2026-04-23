from __future__ import annotations

import json
import os
from typing import Any

import httpx

from app.triage import CareLevel


def _strategy_instruction_block(query_strategy: str) -> str:
    mode = (query_strategy or "single_turn").strip().lower()
    if mode == "single_turn":
        return """QUERY STRATEGY — SINGLE-TURN:
Produce one self-contained traveler-facing answer from the supplied server context. You may list a few questions_to_clarify for optional follow-up, but do not defer the whole guidance until the user replies unless a critical safety detail is truly missing.
Cost / plans: Without live web research in the payload, still output treatment_plan_options and cost_estimate_table using **broad educational ballparks** for the destination; never imply an exact quote."""
    if mode == "single_turn_tools":
        return """QUERY STRATEGY — SINGLE-TURN WITH INTERNAL TOOL USE (no extra user steps):
The server may have run additional retrieval passes without the user. Treat citations_from_retrieval as the merged result of those internal steps. Synthesize one coherent answer in a single user-visible turn; do not ask the user to approve intermediate tool steps.
Cost / plans: When research_from_tools_digest is present, **prefer** its web/Places signals for cost_estimate_table (still clearly uncertain); otherwise use wide educational ranges."""
    if mode == "multi_turn":
        return """QUERY STRATEGY — MULTI-TURN (tools + dialogue):
You may rely on server citations as already-fetched tools. Proactively use questions_to_clarify to request missing information (symptoms, timing, locale, insurance, red flags) when it would materially improve safe orientation. A short summary plus explicit follow-up questions is appropriate; the product supports chat for later rounds."""
    if mode == "unsolvable":
        return """QUERY STRATEGY — UNSOLVABLE / ABSTENTION:
If responsible guidance cannot be given from the provided citations and rules alone, or the user requests disallowed help (diagnosis, prescribing, emergency dispatch), set abstain to true and abstention_reason to a clear explanation for the traveler (in output_language). When abstaining, keep summary_for_traveler brief, set what_to_do_next to safe generic steps (e.g. contact local services / official sources), and avoid fabricating facts. When abstain is false, answer normally."""
    return """QUERY STRATEGY — DEFAULT (single-turn):
Produce one self-contained traveler-facing answer from the supplied server context."""


def _language_instruction_block(output_language: str) -> str:
    code = (output_language or "en").strip().lower()
    base = code[:2] if len(code) >= 2 else code
    labels: dict[str, str] = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "it": "Italian",
        "ar": "Arabic",
        "hi": "Hindi",
        "th": "Thai",
        "vi": "Vietnamese",
        "nl": "Dutch",
        "pl": "Polish",
        "tr": "Turkish",
        "ru": "Russian",
    }
    label = labels.get(base, f"the language best matching code {code!r}")
    return f"""TRAVELER-SELECTED OUTPUT LANGUAGE (mandatory):
The application sets output_language to {code!r} — write for a reader of {label}.

Hard rule: Every traveler-facing string you output in the JSON must be written entirely in that language:
summary_for_traveler, every string in what_to_do_next, every string in healthcare_system_contrast (each may be a multi-sentence paragraph), every string in questions_to_clarify, every string in sources_context_for_traveler, treatment_plan_options titles/descriptions/typical_setting, cost_uncertainty_note, cost_estimate_table plan_title and row item/notes, pharmacy_visit_tips, otc_medication_examples text fields, medication_reference_links titles, medication_reference_images captions, and disclaimer.

Do not use English (or any other language) for convenience when output_language is not English. Do not mix languages inside those fields unless the traveler's own message explicitly mixes them and mirroring is helpful (still default to output_language).

server_decision.rule_rationale and citations_from_retrieval may be in English from the server — translate their meaning into your traveler-facing strings in the selected language; do not paste untranslated English into summary_for_traveler, what_to_do_next, or other traveler-facing list/paragraph fields unless output_language is en."""


def build_system_prompt(
    travel_location: str,
    home_country: str,
    output_language: str,
    query_strategy: str,
    *,
    chat_follow_up: bool = False,
) -> str:
    """Location + optional home country for destination-vs-home healthcare framing."""
    loc = (travel_location or "").strip()
    home = (home_country or "").strip()
    loc_block = (
        loc
        if loc
        else "Current trip location: not specified — do not assume a particular country, city, or local emergency number."
    )
    if home:
        home_block = f"""AUTHORITATIVE TRAVELER HOME (already supplied by the client application — treat as a fixed fact):
{home}

Hard rule: Do NOT ask the traveler which country or region they are from, where they normally live, what passport they hold, or what healthcare system they grew up with — in summary_for_traveler, what_to_do_next, or questions_to_clarify. That information is already given above. questions_to_clarify must only cover other gaps (symptoms, timing, medications, specific insurance product, red-flag symptoms, etc.), not discovering their home jurisdiction."""
    else:
        home_block = """Traveler home country/region (usual healthcare system): NOT supplied in the client payload (traveler_home_country is null).

You should still give useful destination-oriented guidance. You may ask at most ONE concise question in questions_to_clarify to learn their home country or region if a home-vs-host healthcare comparison would materially help — do not repeat the same question if chat_history already contains their answer."""
    follow_block = (
        "\n\nCHAT FOLLOW-UP (this request includes chat_history with a new traveler message):\n"
        "You are answering after prior guidance may already exist in chat_history. The traveler's LATEST user "
        "message is the primary focus — answer it directly. Refresh summary_for_traveler, what_to_do_next, "
        "healthcare_system_contrast, treatment_plan_options, cost_estimate_table, cost_uncertainty_note, "
        "pharmacy_visit_tips, otc_medication_examples, medication_reference_links, medication_reference_images, "
        "questions_to_clarify, and sources_context_for_traveler so they reflect any "
        "NEW facts or questions from that latest message. Do not paste a generic first-visit brief unchanged when "
        "they asked something specific; avoid repeating prior assistant wording unless they explicitly asked for a "
        "recap or the medical facts truly did not change."
        if chat_follow_up
        else ""
    )
    return f"""You are a decision-support assistant for international travelers.
You are NOT a doctor and do NOT provide a medical diagnosis.
Follow the server's rule-based care_level and emergency flag strictly — do not contradict them.

Current trip location (from client):
{loc_block}

{home_block}

Cross-system healthcare education (high level, not legal or insurance advice):

healthcare_system_contrast (required JSON array of strings — each string is one **short, engaging paragraph**, not a one-line host-only fact):
- When BOTH trip location and traveler_home_country are known: **every paragraph must explicitly describe BOTH systems** for angles that fit the traveler’s story (first contact, ER vs clinic vs pharmacy, referrals, payment expectations, etc.). Open with what they might be used to **at home / where they usually live** (name their home using **traveler_home_country** from the JSON naturally), then pivot with a clear contrast (“whereas”, “by contrast”, “on this trip”, “here in …”) to how things **often work where they are now** (use **travel_location** from the JSON). Example **shape** (adapt fully into output_language; use real names from the payload, not placeholders): “For a concern like yours, in your country / at home people often …; **whereas** in <say the trip place> travelers usually …” — **do not** paste this English template verbatim when output_language is not English.
- **Do not** write paragraphs that only educate about the new country. If you describe the destination norm, the same paragraph (or an adjacent one) must anchor the **home-country** norm for that same theme so the reader feels the juxtaposition.
- When traveler home is **not** known (null): write flowing paragraphs about the destination for their kind of issue, and include at least one paragraph that explicitly asks them to picture **“where you usually live”** without naming a country, then contrasts with the trip place — still avoid host-only bullet fragments.
- Stay cautious: laws and plans differ; avoid stereotypes about groups of people; do not invent rules for the traveler’s private insurance.
- Never invent specific local phone numbers unless widely standard; prefer "look up the official emergency number where you are".

Treatment paths and indicative costs (treatment_plan_options + cost_estimate_table — educational only, not a price quote):
- Always output **treatment_plan_options**: 2–4 distinct, reasonable paths the traveler might choose among (e.g. pharmacy/self-care, telehealth or scheduled primary care, walk-in urgent care, emergency department) aligned with **server_decision** (if emergency is true, foreground emergency-appropriate pathways and do not suggest “wait and see” as equivalent to ED). Each option needs stable ASCII **id**, **title**, **description**, **typical_setting** (short).
- Always output **cost_estimate_table**: one object per treatment_plan_options **id**, with **plan_id**, **plan_title** (can mirror title), **currency** (ISO code matching the destination when possible, e.g. USD, EUR), **rows** listing plausible fee components (visit, common labs, typical Rx if relevant) each with **low_amount** / **high_amount** as numbers in that currency when you can infer ballparks, else null with explanatory **notes**. Optional **total_low** / **total_high** only as rough sums. **Never** present numbers as binding quotes, insurance-negotiated rates, or facility-specific bills.
- Always output **cost_uncertainty_note**: vivid reminder that real costs vary by facility, provider, insurance, and urgency; traveler must confirm before care.
- When **research_from_tools_digest** is present, align row notes and ranges with what the digest actually supports; flag when the digest lacked cost data.
- When the client sends **traveler_selected_treatment_plan_id**, output a **detailed** cost_estimate_table: either a single object for that plan_id with **≥4** granular rows, or keep all plans but expand only the selected plan’s rows — and tie **summary_for_traveler** to that choice. Use **prior_treatment_plan_options** in the JSON (when provided) to recover titles for that id.

Pharmacy, urgent-care context, and OTC medication naming (educational only — not prescribing):
- When **server_decision.emergency** is false and a **pharmacy** or **self-care / mild symptom** pathway is reasonable, you **may** elaborate in dedicated JSON fields (see schema): **pharmacy_visit_tips**, **otc_medication_examples**, **medication_reference_links**, **medication_reference_images**.
- **OTC / ingredients**: you may name **common active ingredients or OTC drug classes** travelers might discuss with a pharmacist (e.g. paracetamol/acetaminophen, ibuprofen, oral rehydration salts, simple antihistamines, lozenges). Use **non-directive** language (“many travelers ask a pharmacist about…”, “products may be sold under different brand names…”). **Never** give personal dosing, duration, or “take this” instructions; **never** suggest prescription-only or controlled drugs for self-treatment; **never** suggest antibiotics or opioids as OTC.
- **Brand names**: optional **non-binding** examples only to illustrate local packaging diversity — not endorsements. Prefer **ingredient-first** wording.
- **Links**: **medication_reference_links** may include **https** URLs only if they appear **verbatim** in **research_from_tools_digest** or **citations_from_retrieval**, OR if they are clearly generic official consumer portals for the destination (national medicines regulator, NHS/MHRA-style medicines A–Z, FDA “Drugs@FDA” / OTC basics, WHO essential medicines public pages — use **real** well-known domains, **never** fabricated paths). If unsure, omit the URL and describe what to search for instead.
- **Pictures**: **medication_reference_images** may list **https** image or product-info URLs **only** if copied **verbatim** from research_from_tools_digest. **Do not invent** image URLs, stock-photo hosts, or retailer deep links you did not see in tools.
- When the plan is **urgent care** or **emergency**, keep OTC/pharmacy talk secondary and do not imply OTC replaces needed clinical or emergency care.

Citations from retrieval (important):
The payload includes citations_from_retrieval — short excerpts from a small fixed travel-health text corpus matched by simple keyword overlap, not a jurisdiction-specific or map-backed knowledge base. The product may show those excerpts verbatim to the traveler. You MUST still populate sources_context_for_traveler (see schema): 1–4 concise bullets in output_language explaining how the retrieved themes apply — or do not apply — to their stated trip location, symptoms, home context, and any research_from_tools_digest. If excerpts are generic, say that clearly. Paraphrase themes; do not invent corpus ids, URLs, or local facts not supported by citations or the digest.

If trip location is missing or vague, keep logistics jurisdiction-agnostic and tell the traveler to verify local emergency numbers and care options for where they actually are.

{_strategy_instruction_block(query_strategy)}

{_language_instruction_block(output_language)}
{follow_block}

Output ONLY valid JSON matching the requested schema."""


def _normalize_amount(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize_treatment_plan_options(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for i, row in enumerate(raw[:6]):
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or f"plan_{i + 1}").strip()[:64]
        if not pid:
            continue
        out.append(
            {
                "id": pid,
                "title": str(row.get("title") or pid)[:400],
                "description": str(row.get("description") or "")[:2000],
                "typical_setting": str(row.get("typical_setting") or "")[:300],
            }
        )
    return out


def _normalize_cost_estimate_table(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for row in raw[:8]:
        if not isinstance(row, dict):
            continue
        rows_in = row.get("rows")
        norm_rows: list[dict[str, Any]] = []
        if isinstance(rows_in, list):
            for r in rows_in[:24]:
                if not isinstance(r, dict):
                    continue
                norm_rows.append(
                    {
                        "item": str(r.get("item") or "")[:400],
                        "low_amount": _normalize_amount(r.get("low_amount")),
                        "high_amount": _normalize_amount(r.get("high_amount")),
                        "notes": str(r.get("notes") or "")[:600],
                    }
                )
        out.append(
            {
                "plan_id": str(row.get("plan_id") or "")[:64],
                "plan_title": str(row.get("plan_title") or "")[:400],
                "currency": str(row.get("currency") or "")[:12],
                "rows": norm_rows,
                "total_low": _normalize_amount(row.get("total_low")),
                "total_high": _normalize_amount(row.get("total_high")),
            }
        )
    return out


def _truncate_prompt_for_response(s: str, max_chars: int = 120_000) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n… [truncated by server for response size]"


def _normalize_string_list(raw: Any, *, max_items: int, max_len: int) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for x in raw[:max_items]:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s[:max_len])
    return out


def _normalize_otc_examples(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for row in raw[:5]:
        if not isinstance(row, dict):
            continue
        ing = str(row.get("ingredient_or_class") or "").strip()[:220]
        if not ing:
            continue
        out.append(
            {
                "ingredient_or_class": ing,
                "why_mentioned": str(row.get("why_mentioned") or "")[:900],
                "ask_pharmacist_note": str(row.get("ask_pharmacist_note") or "")[:600],
                "brand_examples_nonbinding": str(row.get("brand_examples_nonbinding") or "")[:400],
            }
        )
    return out


def _normalize_https_link_pairs(raw: Any, *, max_items: int) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for row in raw[:max_items]:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url.startswith("https://"):
            continue
        out.append({"title": str(row.get("title") or "")[:280], "url": url[:2000]})
    return out


def _normalize_image_url_pairs(raw: Any) -> list[dict[str, str]]:
    """HTTPS image or product page URLs with captions."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for row in raw[:5]:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url") or "").strip()
        if not url.startswith("https://"):
            continue
        out.append({"caption": str(row.get("caption") or "")[:400], "url": url[:2000]})
    return out


def _normalize_traveler_llm_json(obj: dict[str, Any]) -> dict[str, Any]:
    obj.setdefault("abstain", False)
    ar = obj.get("abstention_reason")
    obj["abstention_reason"] = ar if isinstance(ar, str) else ("" if ar is None else str(ar))
    if obj["abstain"] is not True:
        obj["abstain"] = False
    sc = obj.get("sources_context_for_traveler")
    if isinstance(sc, list):
        obj["sources_context_for_traveler"] = [str(x) for x in sc if x is not None][:12]
    else:
        obj["sources_context_for_traveler"] = []
    obj["treatment_plan_options"] = _normalize_treatment_plan_options(obj.get("treatment_plan_options"))
    obj["cost_estimate_table"] = _normalize_cost_estimate_table(obj.get("cost_estimate_table"))
    cun = obj.get("cost_uncertainty_note")
    obj["cost_uncertainty_note"] = cun if isinstance(cun, str) else ""
    obj["pharmacy_visit_tips"] = _normalize_string_list(obj.get("pharmacy_visit_tips"), max_items=6, max_len=500)
    obj["otc_medication_examples"] = _normalize_otc_examples(obj.get("otc_medication_examples"))
    obj["medication_reference_links"] = _normalize_https_link_pairs(obj.get("medication_reference_links"), max_items=8)
    obj["medication_reference_images"] = _normalize_image_url_pairs(obj.get("medication_reference_images"))
    return obj


def _chat_follow_up_context(history: list[dict[str, str]]) -> tuple[str | None, str | None]:
    """Latest user message and the assistant reply immediately before it (if any)."""
    if not history:
        return None, None
    last_user_idx: int | None = None
    for i in range(len(history) - 1, -1, -1):
        role = (history[i].get("role") or "").strip()
        content = (history[i].get("content") or "").strip()
        if role == "user" and content:
            last_user_idx = i
            break
    if last_user_idx is None:
        return None, None
    last_user = str(history[last_user_idx]["content"]).strip()
    prior_asst: str | None = None
    for j in range(last_user_idx - 1, -1, -1):
        role = (history[j].get("role") or "").strip()
        content = (history[j].get("content") or "").strip()
        if role == "assistant" and content:
            prior_asst = content[:1200]
            break
    return last_user, prior_asst


async def augment_with_openai(
    *,
    user_message: str,
    language: str,
    travel_location: str,
    home_country: str,
    care_level: CareLevel,
    emergency: bool,
    rule_rationale: list[str],
    citations: list[dict[str, str]],
    chat_history: list[dict[str, str]] | None = None,
    query_strategy: str = "single_turn",
    research_from_tools_digest: str | None = None,
    selected_treatment_plan_id: str | None = None,
    prior_treatment_plan_options: list[dict[str, str]] | None = None,
) -> tuple[dict[str, Any] | Any | None, dict[str, Any] | None]:
    """Returns (traveler_llm_json_or_none, llm_api_prompt_meta_or_none)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None, None

    history = chat_history or []
    chat_follow_up = bool(history)
    latest_follow_up, prior_assistant_excerpt = (
        _chat_follow_up_context(history) if chat_follow_up else (None, None)
    )
    home_set = bool((home_country or "").strip())
    strat = (query_strategy or "single_turn").strip().lower()
    home_resolved = (home_country or "").strip() or None
    loc_resolved = (travel_location or "").strip() or None
    digest = (research_from_tools_digest or "").strip() or None
    sel_plan = (selected_treatment_plan_id or "").strip() or None
    prior_plans = prior_treatment_plan_options or []

    task = (
        "Produce traveler-facing JSON guidance. Always include healthcare_system_contrast as a non-empty "
        "array of 2–5 strings; **each string must be one vivid, engaging short paragraph** (several sentences), "
        "not a terse single-line bullet. When traveler_home_country is set, each paragraph must **pair home norms "
        "with destination norms** for the traveler’s kind of issue (see system prompt). "
        f"Write ALL traveler-facing strings strictly in the language given by output_language ({language!r}) — "
        "same requirement as the system prompt. "
        f"The active query_strategy is {strat!r} (see system prompt)."
    )
    task += (
        " TREATMENT OPTIONS AND COST TABLE: Output treatment_plan_options (2–4 objects: id, title, description, "
        "typical_setting) and cost_estimate_table (one block per plan id with plan_id, plan_title, currency, rows "
        "array of {item, low_amount, high_amount, notes}, optional total_low/total_high). Use plausible destination "
        "currency. Numbers are **indicative** only. Include cost_uncertainty_note (non-empty). "
        "If research_from_tools_digest is empty and query_strategy is single_turn, use **wide** heuristic ranges. "
        "If digest is non-empty (especially with single_turn_tools or research enabled), narrow rows only when "
        "snippets support it and cite uncertainty in notes."
    )
    task += (
        " PHARMACY / OTC EDUCATION: When pharmacy or self-care is a reasonable part of guidance and the server "
        "emergency flag is false, populate pharmacy_visit_tips (0–5 short strings), otc_medication_examples (0–4 objects "
        "with ingredient_or_class, why_mentioned, ask_pharmacist_note, optional brand_examples_nonbinding), "
        "medication_reference_links (0–6 objects with title + https url — grounded per system prompt), "
        "medication_reference_images (0–4 objects with caption + https url — **only** URLs verbatim from "
        "research_from_tools_digest). Use empty lists when not applicable. Never give personal dosing or prescribe."
    )
    if strat == "unsolvable":
        task += (
            " Always output boolean abstain and string abstention_reason (use empty string when abstain is false). "
            "Set abstain true only when the UNSOLVABLE strategy requires declining or severely limiting the answer."
        )
    else:
        task += " Set abstain to false and abstention_reason to an empty string unless a rare safety/policy block requires abstention."
    if home_set:
        task += (
            " CRITICAL: client_context.traveler_home_resolved is non-empty — the traveler's origin/home "
            "jurisdiction is ALREADY provided. Do not ask them to state their country/region of origin or "
            "'where they are from' anywhere in your JSON output."
        )
    else:
        task += (
            " client_context.traveler_home_resolved is empty — you may ask at most one concise question "
            "about home country/region in questions_to_clarify if needed for comparison."
        )
    if chat_follow_up:
        task += (
            " This is a CHAT FOLLOW-UP: chat_history is non-empty. Field latest_follow_up_from_traveler (when present) "
            "is the traveler's newest message — answer it directly first. Refresh summary_for_traveler, what_to_do_next, "
            "healthcare_system_contrast, treatment_plan_options, cost_estimate_table, cost_uncertainty_note, "
            "pharmacy_visit_tips, otc_medication_examples, medication_reference_links, medication_reference_images, "
            "questions_to_clarify, and sources_context_for_traveler to incorporate any NEW "
            "symptoms, timing, or questions from that message. If prior_assistant_message_excerpt is present, treat it as "
            "what they already saw: do not repeat the same blocks verbatim; add delta guidance, corrections, or narrower "
            "next steps. Still follow server_decision strictly."
        )
        if home_set:
            task += (
                " If chat_history contains the user stating their home country, treat it as redundant with "
                "traveler_home_country — still do not ask again."
            )
    if digest:
        task += (
            " The field research_from_tools_digest contains live Google Places (ratings, distances, review snippets) "
            "and/or web search snippets, often including **official local government or public-health pages** when "
            "the research sub-agent found them: weave factual points into what_to_do_next and healthcare_system_contrast "
            "when helpful—especially how the destination system differs from the traveler's home when both are known. "
            "Treat distances as straight-line meters, not drive time. Treat any cost hints as uncertain; "
            "say the traveler should confirm with the facility or official sources. "
            "Use digest content to inform **cost_estimate_table** row notes and ranges when the digest mentions prices "
            "or typical settings; never invent exact facility fees not hinted in the digest. "
            "Use the digest for **medication_reference_links** / **medication_reference_images** only when URLs "
            "appear there verbatim (see system prompt)."
        )
    elif strat == "single_turn":
        task += (
            " research_from_tools_digest is absent: rely on cautious general knowledge for cost_estimate_table "
            "with **broad** bands and strong caveats in cost_uncertainty_note."
        )
    task += (
        " Include sources_context_for_traveler as a list of 1–4 strings (use an empty list only when "
        "citations_from_retrieval is empty): each line should connect retrieval themes to this traveler's trip text, "
        "location, home context, and digest when present; acknowledge when snippets are generic."
    )

    user_blob: dict[str, Any] = {
        "task": task,
        "output_language": language,
        "output_language_note": (
            "All natural-language values in your JSON reply must match output_language; "
            "see system prompt TRAVELER-SELECTED OUTPUT LANGUAGE."
        ),
        "travel_location": loc_resolved,
        "traveler_home_country": home_resolved,
        "client_context": {
            "travel_location_resolved": loc_resolved,
            "traveler_home_resolved": home_resolved,
            "traveler_home_is_known": bool(home_resolved),
        },
        "user_symptoms_or_story": user_message,
        "chat_history": history,
    }
    if chat_follow_up and latest_follow_up:
        user_blob["latest_follow_up_from_traveler"] = latest_follow_up
    if chat_follow_up and prior_assistant_excerpt:
        user_blob["prior_assistant_message_excerpt"] = prior_assistant_excerpt
    if prior_plans:
        user_blob["prior_treatment_plan_options"] = prior_plans[:8]
    if sel_plan:
        user_blob["traveler_selected_treatment_plan_id"] = sel_plan
        task += (
            f" traveler_selected_treatment_plan_id is {sel_plan!r}: produce a **detailed** cost_estimate_table — "
            "prefer a single expanded block for that plan_id with at least four granular rows (visit, common add-ons) "
            "and align summary_for_traveler with that path. Keep treatment_plan_options consistent with "
            "prior_treatment_plan_options when that array is provided; otherwise reconstruct a single matching option."
        )

    system_content = build_system_prompt(
        travel_location,
        home_country,
        language,
        strat,
        chat_follow_up=chat_follow_up,
    )
    user_message_payload: dict[str, Any] = {
        **user_blob,
        "server_decision": {
            "care_level": care_level,
            "emergency": emergency,
            "rule_rationale": rule_rationale,
        },
        "citations_from_retrieval": citations,
        "query_strategy": strat,
        "research_from_tools_digest": digest,
        "schema": {
            "abstain": "boolean",
            "abstention_reason": "string (empty when abstain is false)",
            "summary_for_traveler": "string",
            "what_to_do_next": ["string"],
            "healthcare_system_contrast": [
                "string (each item: one engaging short paragraph; when home is known, pair at-home vs on-trip norms — not host-only bullets)"
            ],
            "questions_to_clarify": ["string"],
            "sources_context_for_traveler": ["string"],
            "treatment_plan_options": [
                {
                    "id": "string (stable slug)",
                    "title": "string",
                    "description": "string",
                    "typical_setting": "string",
                }
            ],
            "cost_estimate_table": [
                {
                    "plan_id": "string",
                    "plan_title": "string",
                    "currency": "string (ISO)",
                    "rows": [
                        {
                            "item": "string",
                            "low_amount": "number|null",
                            "high_amount": "number|null",
                            "notes": "string",
                        }
                    ],
                    "total_low": "number|null",
                    "total_high": "number|null",
                }
            ],
            "cost_uncertainty_note": "string",
            "pharmacy_visit_tips": ["string"],
            "otc_medication_examples": [
                {
                    "ingredient_or_class": "string",
                    "why_mentioned": "string",
                    "ask_pharmacist_note": "string",
                    "brand_examples_nonbinding": "string (optional)",
                }
            ],
            "medication_reference_links": [{"title": "string", "url": "string (https only, grounded)"}],
            "medication_reference_images": [{"caption": "string", "url": "string (https only, from tools only)"}],
            "disclaimer": "string",
        },
    }
    user_content_str = json.dumps(user_message_payload, ensure_ascii=False)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    temperature = 0.38 if chat_follow_up else 0.2
    payload = {
        "model": model_name,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content_str},
        ],
    }

    prompt_meta: dict[str, Any] = {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": model_name,
        "temperature": temperature,
        "system": _truncate_prompt_for_response(system_content),
        "user": _truncate_prompt_for_response(user_content_str),
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return _normalize_traveler_llm_json(parsed), prompt_meta
        return parsed, prompt_meta


async def plan_retrieval_subqueries(*, user_text: str) -> list[str]:
    """LLM proposes extra corpus search strings (internal 'tool' step; corpus keywords are English)."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return []

    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0.15,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You help plan keyword searches for a tiny English travel-health text corpus (educational, not clinical). "
                    "Output valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "traveler_case_text": user_text[:6000],
                        "task": (
                            "Propose 2-4 short English search phrases (keywords, not full sentences) to retrieve "
                            "relevant corpus lines about access to care, insurance, emergencies, pharmacies, or heat illness."
                        ),
                        "schema": {"retrieval_queries": ["string"], "notes": "string"},
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    raw = parsed.get("retrieval_queries") or []
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                out.append(item.strip()[:200])
    return out[:5]
