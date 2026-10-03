"""
NeuroSense AI - Curated Resource Selector

Purpose
-------
Selects optional, verified YouTube resources AFTER the recommendation engine has
already decided what the patient needs. This file never decides medical advice.
It only matches a curated video to RAG-selected topics such as gait, balance,
strength, speech or voice.

Expected location:
    recommendations_app/rag/resource_selector.py

Expected resource file:
    recommendations_app/rag/resources/youtube_resources.json
"""

from __future__ import annotations

import json
import html
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = APP_DIR / "rag"
DOCUMENTS_DIR = RAG_DIR / "documents"
RESOURCES_DIR = RAG_DIR / "resources"
YOUTUBE_FILE = RESOURCES_DIR / "youtube_resources.json"

SUPPORTED_SECTIONS = (
    "physical_therapy_exercise",
    "speech_voice_therapy",
)

_resources_cache: Optional[List[Dict[str, Any]]] = None
_video_tags_by_card: Optional[Dict[str, Set[str]]] = None


def _normalize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _as_set(values: Any) -> Set[str]:
    if values is None:
        return set()
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    return {_normalize(v) for v in values if _normalize(v)}


def _load_resources() -> List[Dict[str, Any]]:
    global _resources_cache

    if _resources_cache is not None:
        return _resources_cache

    if not YOUTUBE_FILE.exists():
        logger.warning("YouTube resource file not found: %s", YOUTUBE_FILE)
        _resources_cache = []
        return _resources_cache

    try:
        with YOUTUBE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _resources_cache = [
            resource for resource in payload.get("resources", [])
            if isinstance(resource, dict)
            and resource.get("verified") is True
            and resource.get("url")
        ]
    except Exception as exc:
        logger.exception("Could not load YouTube resources: %s", exc)
        _resources_cache = []

    return _resources_cache


def reload_resources() -> None:
    """Clear cached resource metadata after editing the JSON file."""
    global _resources_cache, _video_tags_by_card
    _resources_cache = None
    _video_tags_by_card = None


def _load_video_tags_by_card() -> Dict[str, Set[str]]:
    """
    Read video_resource_tags from the recommendation knowledge cards.

    This keeps the resource decision linked to the recommendation cards that
    RAG already selected, rather than asking the LLM to decide which video fits.
    """
    global _video_tags_by_card

    if _video_tags_by_card is not None:
        return _video_tags_by_card

    mapping: Dict[str, Set[str]] = {}

    if DOCUMENTS_DIR.exists():
        for file_path in DOCUMENTS_DIR.glob("*.json"):
            if file_path.name == "recommendation_sources.json":
                continue

            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue

            for card in payload.get("knowledge_cards", []):
                if not isinstance(card, dict) or not card.get("id"):
                    continue
                mapping[str(card["id"])] = _as_set(card.get("video_resource_tags"))

    _video_tags_by_card = mapping
    return mapping


def _extract_safety_state(intelligence: Dict[str, Any]) -> Dict[str, bool]:
    """
    Detect only explicit safety clues already present in NeuroSense intelligence.
    We do not infer high fall risk merely from a high Parkinson's probability.
    """
    text_parts: List[str] = []

    questionnaire = intelligence.get("questionnaire") or {}
    q_analysis = questionnaire.get("analysis") or {}
    text_parts.extend(str(v) for v in q_analysis.get("safety_flags", []) or [])

    for insight in intelligence.get("insights") or []:
        if isinstance(insight, dict):
            text_parts.extend([
                str(insight.get("type", "")),
                str(insight.get("en", "")),
            ])

    text = " ".join(text_parts).lower()

    return {
        "fall_risk": any(term in text for term in (
            "fall", "near-fall", "near fall", "repeated falls", "balance safety"
        )),
        "swallowing_concern": any(term in text for term in (
            "swallow", "choking", "coughing with food", "coughing with drink"
        )),
    }


def _query_tags_for_section(
    section: str,
    retrieved_cards: List[Dict[str, Any]],
) -> Set[str]:
    tags: Set[str] = set()
    card_tag_map = _load_video_tags_by_card()

    for card in retrieved_cards:
        card_id = str(card.get("id", ""))
        tags.update(card_tag_map.get(card_id, set()))
        tags.update(_as_set(card.get("topics")))

    # Normalize a few recommendation-side names to the video library vocabulary.
    aliases = {
        "general_activity": "general_exercise",
        "general_mobility": "mobility",
        "voice_maintenance": "voice",
        "speech_dominant": "speech",
        "functional_mobility": "mobility",
        "fall_prevention": "balance",
        "slp_guided": "speech",
    }
    tags.update(aliases[tag] for tag in list(tags) if tag in aliases)

    # Section itself is useful as a weak tag.
    tags.add(_normalize(section))
    return tags


def _resource_score(
    resource: Dict[str, Any],
    query_tags: Set[str],
    safety: Dict[str, bool],
) -> Optional[float]:
    topic_tags = _as_set(resource.get("topic_tags"))
    use_when = _as_set(resource.get("use_when"))
    safety_tags = _as_set(resource.get("safety_tags"))
    avoid_when = _as_set(resource.get("avoid_when"))

    # Hard safety exclusions.
    if safety.get("fall_risk"):
        if (
            "unassessed_high_fall_risk" in avoid_when
            or "standing" in safety_tags
            or "balance_challenge" in safety_tags
        ):
            return None

    if safety.get("swallowing_concern"):
        if (
            "new_swallowing_difficulty_without_assessment" in avoid_when
            and resource.get("section") == "speech_voice_therapy"
        ):
            return None

    topic_overlap = len(query_tags.intersection(topic_tags))
    use_overlap = len(query_tags.intersection(use_when))

    score = (topic_overlap * 3.0) + (use_overlap * 2.0)

    # Prefer safer seated/lower-intensity choices when an explicit fall concern exists.
    if safety.get("fall_risk"):
        if "seated" in safety_tags:
            score += 5.0
        if "reduced_balance_demand" in safety_tags:
            score += 3.0

    # Small preference for direct topic matching over broad playlists.
    if "playlist" in safety_tags:
        score -= 0.25

    return score


def select_resource_for_section(
    section: str,
    intelligence: Dict[str, Any],
    retrieved_cards: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return at most one verified resource for a supported report section."""
    if section not in SUPPORTED_SECTIONS:
        return None

    resources = [
        resource for resource in _load_resources()
        if resource.get("section") == section
    ]
    if not resources:
        return None

    query_tags = _query_tags_for_section(section, retrieved_cards)
    safety = _extract_safety_state(intelligence)

    ranked = []
    for resource in resources:
        score = _resource_score(resource, query_tags, safety)
        if score is None:
            continue
        ranked.append((score, resource))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1].get("id", "")), reverse=True)
    best_score, best = ranked[0]

    # Do not attach an unrelated video just to fill space.
    if best_score <= 0:
        return None

    selected = dict(best)
    selected["selection_score"] = round(best_score, 2)
    selected["matched_tags"] = sorted(
        query_tags.intersection(_as_set(best.get("topic_tags")))
    )
    return selected


def select_report_resources(
    intelligence: Dict[str, Any],
    retrieved_report: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Select optional resources for the two report sections that can use them."""
    return {
        section: select_resource_for_section(
            section=section,
            intelligence=intelligence,
            retrieved_cards=retrieved_report.get(section) or [],
        )
        for section in SUPPORTED_SECTIONS
    }


def format_resources_for_prompt(
    resources: Dict[str, Optional[Dict[str, Any]]],
) -> str:
    """
    Give Gemini visibility of already-selected resources.
    Gemini is explicitly not allowed to change the URLs.
    """
    lines: List[str] = []

    for section in SUPPORTED_SECTIONS:
        resource = resources.get(section)
        if not resource:
            continue

        lines.extend([
            f"Section: {section}",
            f"Resource title: {resource.get('title', '')}",
            f"Provider: {resource.get('provider', '')}",
            f"Exact URL: {resource.get('url', '')}",
            f"Safety note: {resource.get('safety_note', '')}",
            "",
        ])

    return "\n".join(lines).strip() or "No external resource selected."


def _resource_html(resource: Dict[str, Any], lang: str = "en") -> str:
    title = resource.get("title", "Guided resource")
    provider = resource.get("provider", "")
    url = resource.get("url", "")

    if lang == "ur":
        label = "رہنمائی ویڈیو"
        provider_label = "ذریعہ"
        practice_label = "آسان مشقیں"
        safety_note = (
            resource.get("safety_note_ur")
            or resource.get("safety_note", "")
        )
        practice_points = (
            resource.get("practice_points_ur")
            or resource.get("practice_points")
            or []
        )
        list_dir = "rtl"
    else:
        label = "Guided Resource"
        provider_label = "Provider"
        practice_label = "Quick practice"
        safety_note = resource.get("safety_note", "")
        practice_points = resource.get("practice_points") or []
        list_dir = "ltr"

    practice_html = ""
    if practice_points:
        items = "".join(
            f"<li>{html.escape(str(point))}</li>"
            for point in practice_points[:3]
            if str(point).strip()
        )

        if items:
            practice_html = (
                '<div class="resource-practice">'
                f'<span class="resource-practice-label">{practice_label}</span>'
                f'<ul class="resource-practice-list" dir="{list_dir}">'
                f'{items}'
                '</ul>'
                '</div>'
            )

    return (
        '<div class="guided-resource">'
        f'<strong class="resource-label">{label}:</strong>'
        f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>'
        f'<span class="resource-provider">({provider_label}: {provider})</span>'
        + practice_html
        + (
            f'<small class="resource-safety-note">{safety_note}</small>'
            if safety_note else ""
        )
        + '</div>'
    )


def inject_resources_into_report(
    report_markdown: str,
    resources: Dict[str, Optional[Dict[str, Any]]],
    lang: str = "en",
) -> str:
    """
    Inject exact curated URLs into the generated report programmatically.

    This is deliberately done AFTER LLM generation, so the LLM cannot alter,
    hallucinate, or substitute a YouTube link.
    """
    if not report_markdown:
        return report_markdown

    output = report_markdown

    insertions = (
        ("physical_therapy_exercise", 2, 3),
        ("speech_voice_therapy", 3, 4),
    )

    for section, current_num, next_num in insertions:
        resource = resources.get(section)
        if not resource:
            continue

        block = "\n\n" + _resource_html(resource, lang=lang) + "\n"
        pattern = rf"(?m)(^##\s+{next_num}\.\s+)"

        match = re.search(pattern, output)
        if match:
            output = output[:match.start()] + block + "\n" + output[match.start():]
        else:
            # Safe fallback if an LLM slightly changes the final heading.
            output = output.rstrip() + block + "\n"

    return output
