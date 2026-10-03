"""
NeuroSense AI - Recommendation RAG Validator

Purpose
-------
Validates the recommendation knowledge base before integration/testing.

Checks:
- all six required recommendation JSON files exist
- each file uses the expected section name
- card IDs are unique
- required base cards exist
- source_ids referenced by cards exist in recommendation_sources.json
- YouTube resource IDs are unique
- YouTube URLs are present and HTTPS
- only supported report sections are used

Expected location:
    recommendations_app/rag/validator.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

RAG_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = RAG_DIR / "documents"
RESOURCES_DIR = RAG_DIR / "resources"

SOURCE_REGISTRY_FILE = DOCUMENTS_DIR / "recommendation_sources.json"
YOUTUBE_FILE = RESOURCES_DIR / "youtube_resources.json"

REQUIRED_FILES = {
    "medical_consultation": "medical_consultation.json",
    "physical_therapy_exercise": "physical_therapy_exercise.json",
    "speech_voice_therapy": "speech_voice_therapy.json",
    "lifestyle_wellness": "lifestyle_wellness.json",
    "symptom_monitoring": "symptom_monitoring.json",
    "follow_up_schedule": "follow_up_schedule.json",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_knowledge_base() -> Dict[str, Any]:
    """
    Validate NeuroSense recommendation RAG files.

    Returns:
        {
            "ok": bool,
            "errors": [...],
            "warnings": [...],
            "stats": {...}
        }
    """
    errors: List[str] = []
    warnings: List[str] = []

    stats = {
        "sections_found": 0,
        "knowledge_cards": 0,
        "sources": 0,
        "youtube_resources": 0,
    }

    # ------------------------------------------------------------
    # SOURCE REGISTRY
    # ------------------------------------------------------------
    source_ids: Set[str] = set()

    if not SOURCE_REGISTRY_FILE.exists():
        errors.append(
            f"Missing source registry: {SOURCE_REGISTRY_FILE}"
        )
    else:
        try:
            registry = _read_json(SOURCE_REGISTRY_FILE)
            sources = registry.get("sources", [])

            if not isinstance(sources, list):
                errors.append(
                    "recommendation_sources.json: 'sources' must be a list."
                )
                sources = []

            for source in sources:
                if not isinstance(source, dict):
                    warnings.append(
                        "recommendation_sources.json contains a non-object source entry."
                    )
                    continue

                source_id = str(source.get("id", "")).strip()
                if not source_id:
                    errors.append(
                        "A source in recommendation_sources.json has no id."
                    )
                    continue

                if source_id in source_ids:
                    errors.append(f"Duplicate source id: {source_id}")

                source_ids.add(source_id)

                url = str(source.get("url", "")).strip()
                if not url:
                    warnings.append(f"Source {source_id} has no URL.")
                elif not url.startswith("https://"):
                    warnings.append(
                        f"Source {source_id} URL is not HTTPS: {url}"
                    )

            stats["sources"] = len(source_ids)

        except Exception as exc:
            errors.append(
                f"Could not parse recommendation_sources.json: {exc}"
            )

    # ------------------------------------------------------------
    # SIX RECOMMENDATION SECTIONS
    # ------------------------------------------------------------
    all_card_ids: Set[str] = set()

    for expected_section, filename in REQUIRED_FILES.items():
        path = DOCUMENTS_DIR / filename

        if not path.exists():
            errors.append(f"Missing recommendation file: {filename}")
            continue

        try:
            document = _read_json(path)
        except Exception as exc:
            errors.append(f"{filename}: invalid JSON: {exc}")
            continue

        actual_section = document.get("section")
        if actual_section != expected_section:
            errors.append(
                f"{filename}: section must be '{expected_section}', "
                f"found '{actual_section}'."
            )

        cards = document.get("knowledge_cards")
        if not isinstance(cards, list) or not cards:
            errors.append(
                f"{filename}: 'knowledge_cards' must be a non-empty list."
            )
            continue

        stats["sections_found"] += 1
        stats["knowledge_cards"] += len(cards)

        policy = document.get("retrieval_policy") or {}
        required_base_card = policy.get("required_base_card")

        file_card_ids: Set[str] = set()

        for index, card in enumerate(cards, start=1):
            if not isinstance(card, dict):
                errors.append(
                    f"{filename}: card #{index} is not a JSON object."
                )
                continue

            card_id = str(card.get("id", "")).strip()
            if not card_id:
                errors.append(
                    f"{filename}: card #{index} has no id."
                )
                continue

            if card_id in file_card_ids:
                errors.append(
                    f"{filename}: duplicate card id {card_id}."
                )

            if card_id in all_card_ids:
                errors.append(
                    f"Card id {card_id} is duplicated across files."
                )

            file_card_ids.add(card_id)
            all_card_ids.add(card_id)

            recommendation = str(card.get("recommendation", "")).strip()
            if not recommendation:
                errors.append(
                    f"{filename}: card {card_id} has no recommendation text."
                )

            source_refs = card.get("source_ids", [])
            if source_refs is None:
                source_refs = []

            if not isinstance(source_refs, list):
                errors.append(
                    f"{filename}: card {card_id} source_ids must be a list."
                )
                source_refs = []

            for source_id in source_refs:
                if source_ids and source_id not in source_ids:
                    errors.append(
                        f"{filename}: card {card_id} references "
                        f"missing source id {source_id}."
                    )

        if required_base_card:
            if required_base_card not in file_card_ids:
                errors.append(
                    f"{filename}: required base card "
                    f"{required_base_card} does not exist."
                )
        else:
            warnings.append(
                f"{filename}: no required_base_card is configured."
            )

    # ------------------------------------------------------------
    # YOUTUBE / RESOURCE LIBRARY
    # ------------------------------------------------------------
    if not YOUTUBE_FILE.exists():
        warnings.append(
            f"YouTube resource file is missing: {YOUTUBE_FILE}"
        )
    else:
        try:
            resource_doc = _read_json(YOUTUBE_FILE)
            resources = resource_doc.get("resources", [])

            if not isinstance(resources, list):
                errors.append(
                    "youtube_resources.json: 'resources' must be a list."
                )
                resources = []

            resource_ids: Set[str] = set()

            for index, resource in enumerate(resources, start=1):
                if not isinstance(resource, dict):
                    errors.append(
                        f"youtube_resources.json: resource #{index} "
                        f"is not an object."
                    )
                    continue

                resource_id = str(resource.get("id", "")).strip()
                if not resource_id:
                    errors.append(
                        f"youtube_resources.json: resource #{index} "
                        f"has no id."
                    )
                    continue

                if resource_id in resource_ids:
                    errors.append(
                        f"Duplicate YouTube resource id: {resource_id}"
                    )
                resource_ids.add(resource_id)

                section = resource.get("section")
                if section not in {
                    "physical_therapy_exercise",
                    "speech_voice_therapy",
                }:
                    errors.append(
                        f"{resource_id}: unsupported resource section "
                        f"'{section}'."
                    )

                url = str(resource.get("url", "")).strip()
                if not url:
                    errors.append(
                        f"{resource_id}: missing URL."
                    )
                elif not url.startswith("https://"):
                    errors.append(
                        f"{resource_id}: URL must use HTTPS."
                    )

                if resource.get("verified") is not True:
                    warnings.append(
                        f"{resource_id}: verified is not true, "
                        f"so the selector will ignore it."
                    )

            stats["youtube_resources"] = len(resource_ids)

        except Exception as exc:
            errors.append(
                f"Could not parse youtube_resources.json: {exc}"
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def print_validation_report() -> bool:
    """Print a readable validation report and return True when valid."""
    result = validate_knowledge_base()

    print("\nNeuroSense Recommendation RAG Validation")
    print("=" * 42)

    stats = result["stats"]
    print(f"Sections:          {stats['sections_found']}/6")
    print(f"Knowledge cards:   {stats['knowledge_cards']}")
    print(f"Reference sources: {stats['sources']}")
    print(f"Video resources:   {stats['youtube_resources']}")

    if result["errors"]:
        print("\nERRORS:")
        for item in result["errors"]:
            print(f"  - {item}")

    if result["warnings"]:
        print("\nWARNINGS:")
        for item in result["warnings"]:
            print(f"  - {item}")

    if result["ok"]:
        print("\nSTATUS: VALID")
    else:
        print("\nSTATUS: INVALID")

    return bool(result["ok"])


if __name__ == "__main__":
    raise SystemExit(0 if print_validation_report() else 1)
