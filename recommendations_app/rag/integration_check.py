"""
NeuroSense AI - Recommendation RAG Integration Check

Purpose
-------
Run this after placing all RAG files in the recommendation app.

It checks:
1. Required Python packages/imports.
2. All six knowledge JSON files and source references.
3. Embedding/index construction.
4. Personalized retrieval for a synthetic profile where all sections are relevant.
5. Optional verified YouTube/resource selection.

Expected location:
    recommendations_app/rag/integration_check.py

Run from the Django project root:
    python -m recommendations_app.rag.integration_check

For a quick JSON-only check without downloading/loading the embedding model:
    python -m recommendations_app.rag.integration_check --skip-embeddings
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Dict, List


def _check_import(module_name: str, display_name: str, required: bool = True):
    try:
        importlib.import_module(module_name)
        return True, f"[OK] {display_name}"
    except Exception as exc:
        label = "ERROR" if required else "WARN"
        return False, f"[{label}] {display_name}: {exc}"


def _sample_intelligence() -> Dict[str, Any]:
    """
    Synthetic integration profile only.

    It does not represent a real patient. All three modules are deliberately
    elevated and a constipation clue is included so every fixed report section
    has a legitimate reason to retrieve content during the integration check.
    """
    return {
        "primary_pattern": "all_pd",
        "primary_pattern_label": "Multi-domain Pattern",
        "primary_pattern_description": (
            "Synthetic test profile with gait, speech and questionnaire signals."
        ),
        "detected_patterns": ["all_pd"],
        "risk_summary": {
            "risk_level": "High",
            "final_pd_probability": 0.72,
            "fusion_confidence": 0.86,
        },
        "fusion": {
            "analysis": {
                "module_probabilities": {
                    "gait": 0.78,
                    "speech": 0.70,
                    "questionnaire": 0.65,
                },
            },
            "dominant_module": "gait",
            "dominant_share_pct": 47.0,
            "discordance_level": "low",
            "certainty_level": "high",
        },
        "module_predictions": {
            "gait": "PD",
            "speech": "PD",
            "questionnaire": "PD",
        },
        "speech": {
            "consistency": "stable",
            "threshold_proximity": "above_parkinson",
        },
        "questionnaire": {
            "motor_severity": "moderate",
            "support_level": "Some support recommended",
            "reliability": "high",
            "analysis": {
                "safety_flags": ["constipation"],
            },
        },
        "longitudinal": {
            "overall_trend": "stable",
            "delta_direction": "stable",
            "previous_pd_probability": 0.70,
            "previous_screening_date": "synthetic-test",
            "delta_percentage": 2.0,
        },
        "insights": [
            {
                "type": "multi_domain",
                "en": "Synthetic multi-domain test profile.",
            }
        ],
    }


def _print_heading(text: str):
    print("\n" + text)
    print("=" * len(text))


def run(skip_embeddings: bool = False) -> bool:
    success = True

    _print_heading("1. Python dependency check")

    dependency_checks = [
        ("numpy", "NumPy", True),
        ("sentence_transformers", "Sentence Transformers", not skip_embeddings),
        ("google.genai", "Google GenAI", False),
        ("faiss", "FAISS CPU", False),
    ]

    for module_name, display_name, required in dependency_checks:
        ok, message = _check_import(module_name, display_name, required=required)
        print(message)
        if required and not ok:
            success = False

    _print_heading("2. Knowledge-base validation")

    try:
        from .validator import validate_knowledge_base

        validation = validate_knowledge_base()
        stats = validation.get("stats", {})

        print(f"Sections:          {stats.get('sections_found', 0)}/6")
        print(f"Knowledge cards:   {stats.get('knowledge_cards', 0)}")
        print(f"Reference sources: {stats.get('sources', 0)}")
        print(f"Video resources:   {stats.get('youtube_resources', 0)}")

        for warning in validation.get("warnings", []):
            print(f"[WARN] {warning}")

        for error in validation.get("errors", []):
            print(f"[ERROR] {error}")

        if validation.get("ok"):
            print("[OK] Knowledge-base structure is valid.")
        else:
            success = False

    except Exception as exc:
        print(f"[ERROR] Validator could not run: {exc}")
        success = False

    if skip_embeddings:
        _print_heading("3. Embedding/retrieval check")
        print("[SKIPPED] --skip-embeddings was supplied.")
        return success

    _print_heading("3. RAG index and six-section retrieval")

    try:
        from .rag_service import (
            REPORT_SECTIONS,
            build_index,
            retrieve_report_recommendations,
        )

        build_index(force=True)
        print("[OK] Recommendation embedding index built.")

        intelligence = _sample_intelligence()
        retrieved = retrieve_report_recommendations(intelligence)

        missing_sections: List[str] = []

        for section in REPORT_SECTIONS:
            cards = retrieved.get(section) or []
            ids = [str(card.get("id", "?")) for card in cards]

            if not cards:
                missing_sections.append(section)
                print(f"[ERROR] {section}: no recommendation retrieved")
            else:
                print(
                    f"[OK] {section}: {len(cards)} card(s) -> "
                    + ", ".join(ids)
                )

        if missing_sections:
            success = False

    except Exception as exc:
        print(f"[ERROR] RAG index/retrieval failed: {exc}")
        print(
            "Tip: install packages from rag_requirements.txt and make sure the "
            "embedding model can be downloaded on first use."
        )
        return False

    _print_heading("4. Curated resource selection")

    try:
        from .resource_selector import select_report_resources

        resources = select_report_resources(
            intelligence=_sample_intelligence(),
            retrieved_report=retrieved,
        )

        selected_count = 0
        for section, resource in resources.items():
            if resource:
                selected_count += 1
                print(
                    f"[OK] {section}: {resource.get('id')} - "
                    f"{resource.get('title')}"
                )
            else:
                print(f"[INFO] {section}: no safe matching video selected")

        if selected_count == 0:
            print(
                "[INFO] Zero videos can still be valid because the selector is "
                "designed to return nothing rather than attach an unsafe or "
                "irrelevant resource."
            )

    except Exception as exc:
        print(f"[ERROR] Resource selection failed: {exc}")
        success = False

    _print_heading("FINAL STATUS")
    if success:
        print(
            "PASS - The recommendation RAG foundation is ready for Django "
            "request/view integration."
        )
    else:
        print(
            "FAIL - Fix the errors above before changing views.py or testing "
            "the report page."
        )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Check the NeuroSense recommendation RAG integration."
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Validate files/imports only; do not build the embedding index.",
    )
    args = parser.parse_args()

    ok = run(skip_embeddings=args.skip_embeddings)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
