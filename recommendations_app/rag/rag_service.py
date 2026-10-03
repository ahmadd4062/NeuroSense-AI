"""
NeuroSense AI - Recommendation RAG Service

Purpose
-------
This service retrieves recommendation content from NeuroSense's curated JSON
knowledge base. The application/analyzers decide the patient's structured
profile (risk, pattern, dominant module, longitudinal trend, etc.). RAG then
selects the best eligible knowledge cards for each of the six fixed report
sections.

The LLM is NOT the recommendation decision-maker. A later service may use the
retrieved cards only to make the wording natural, concise, bilingual, and
patient-friendly while preserving the existing report structure.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import faiss
except ImportError:  # Safe fallback: semantic search still works with NumPy dot products.
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


# ============================================================================
# PATHS / CONFIG
# ============================================================================

# rag_service.py is expected at: recommendations_app/rag/rag_service.py
APP_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = APP_DIR / "rag"
DOCUMENTS_DIR = RAG_DIR / "documents"
RESOURCES_DIR = RAG_DIR / "resources"
SOURCE_REGISTRY_FILE = DOCUMENTS_DIR / "recommendation_sources.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

REPORT_SECTIONS: Tuple[str, ...] = (
    "physical_therapy_exercise",
    "speech_voice_therapy",
    "lifestyle_wellness",
    "symptom_monitoring",
    "medical_consultation",
    "follow_up_schedule",
)

SECTION_LABELS = {
    "physical_therapy_exercise": "1. Physical Therapy & Exercise",
    "speech_voice_therapy": "2. Speech & Voice Therapy",
    "lifestyle_wellness": "3. Lifestyle & Wellness",
    "symptom_monitoring": "4. Symptom Monitoring",
    "medical_consultation": "5. Changes to Watch Closely",
    "follow_up_schedule": "6. Follow-up Schedule",
}

# Metadata should dominate ranking because it comes from NeuroSense's own
# analyzers. Semantic similarity is used to rank already-eligible cards.
METADATA_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.30
KEYWORD_WEIGHT = 0.15

STOP_WORDS = {
    "a", "an", "the", "is", "are", "am", "i", "my", "me", "of", "to",
    "in", "on", "and", "or", "can", "could", "why", "what", "how", "with",
    "have", "has", "been", "do", "does", "for", "from", "this", "that",
    "your", "you", "their", "they", "it", "as", "at", "be", "by",
}


# ============================================================================
# LAZY GLOBAL STATE
# ============================================================================

_embedding_model: Any = None
_cards: List[Dict[str, Any]] = []
_card_index: Any = None
_embedding_matrix: Optional[np.ndarray] = None
_source_registry: Dict[str, Dict[str, Any]] = {}
_loaded = False


# ============================================================================
# BASIC HELPERS
# ============================================================================

def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _normalize_set(values: Iterable[Any]) -> Set[str]:
    return {_normalize_token(v) for v in values if _normalize_token(v)}


def _get_embedding_model():
    global _embedding_model
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is required for recommendation RAG embeddings. "
            "Install it in the same environment used by the existing chatbot RAG."
        )
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _keyword_score(query: str, text: str) -> float:
    query_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", query.lower())
    text_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]*\b", text.lower())

    query_words = [word for word in query_words if word not in STOP_WORDS]
    if not query_words:
        return 0.0

    text_counter = Counter(text_words)
    unique_query = set(query_words)
    matches = sum(1 for word in unique_query if text_counter[word] > 0)
    return matches / max(len(unique_query), 1)


# ============================================================================
# KNOWLEDGE-BASE LOADING
# ============================================================================

def _load_source_registry() -> Dict[str, Dict[str, Any]]:
    if not SOURCE_REGISTRY_FILE.exists():
        logger.warning("Recommendation source registry not found: %s", SOURCE_REGISTRY_FILE)
        return {}

    try:
        with SOURCE_REGISTRY_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        registry = {}
        for source in payload.get("sources", []):
            source_id = source.get("id")
            if source_id:
                registry[source_id] = source
        return registry
    except Exception as exc:
        logger.exception("Could not load recommendation source registry: %s", exc)
        return {}


def _card_embedding_text(card: Dict[str, Any]) -> str:
    parts = [
        card.get("section", ""),
        card.get("title", ""),
        " ".join(_as_list(card.get("topics"))),
        card.get("recommendation", ""),
        card.get("personalization_hint", ""),
        card.get("clinical_follow_up", ""),
        card.get("neurosense_repeat_plan", ""),
        card.get("safety_note", ""),
    ]
    return " ".join(str(part) for part in parts if part).strip()


def _load_knowledge_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []

    if not DOCUMENTS_DIR.exists():
        logger.warning("Recommendation RAG documents directory not found: %s", DOCUMENTS_DIR)
        return cards

    for file_path in sorted(DOCUMENTS_DIR.glob("*.json")):
        # Registry contains references, not recommendation cards.
        if file_path.name == SOURCE_REGISTRY_FILE.name:
            continue

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except Exception as exc:
            logger.warning("Skipping invalid RAG JSON %s: %s", file_path.name, exc)
            continue

        section = document.get("section")
        knowledge_cards = document.get("knowledge_cards")

        # youtube_resources.json is deliberately not indexed here even if a user
        # accidentally puts it in documents/. It belongs to the resource layer.
        if not section or not isinstance(knowledge_cards, list):
            continue

        if section not in REPORT_SECTIONS:
            logger.warning("Skipping unknown recommendation section '%s' in %s", section, file_path.name)
            continue

        required_base_card = document.get("retrieval_policy", {}).get("required_base_card")
        max_cards = document.get("retrieval_policy", {}).get("max_cards_per_report")

        for raw_card in knowledge_cards:
            if not isinstance(raw_card, dict):
                continue

            card_id = raw_card.get("id")
            recommendation = raw_card.get("recommendation")
            if not card_id or not recommendation:
                logger.warning("Skipping incomplete card in %s", file_path.name)
                continue

            card = dict(raw_card)
            card["section"] = section
            card["display_heading"] = document.get(
                "display_heading", SECTION_LABELS.get(section, section)
            )
            card["source_file"] = file_path.name
            card["required_base_card"] = required_base_card
            card["section_max_cards"] = max_cards
            card["embedding_text"] = _card_embedding_text(card)
            cards.append(card)

    return cards


def build_index(force: bool = False) -> None:
    """Load the JSON knowledge base and build the in-memory FAISS index."""
    global _cards, _card_index, _embedding_matrix, _source_registry, _loaded

    if _loaded and not force:
        return

    _source_registry = _load_source_registry()
    _cards = _load_knowledge_cards()
    _card_index = None
    _embedding_matrix = None

    if not _cards:
        _loaded = True
        logger.warning("No recommendation knowledge cards were loaded.")
        return

    model = _get_embedding_model()
    texts = [card["embedding_text"] for card in _cards]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    _embedding_matrix = embeddings

    if faiss is not None:
        _card_index = faiss.IndexFlatIP(embeddings.shape[1])
        _card_index.add(embeddings)
    else:
        _card_index = None
        logger.warning(
            "FAISS is not installed; recommendation RAG is using the NumPy "
            "cosine-similarity fallback. Retrieval behavior remains semantic."
        )

    _loaded = True
    logger.info("Recommendation RAG index built with %d knowledge cards.", len(_cards))


def reload_index() -> None:
    """Explicitly rebuild the index after JSON files are edited."""
    build_index(force=True)


# ============================================================================
# PROFILE EXTRACTION FROM EXISTING NEUROSENSE INTELLIGENCE
# ============================================================================

def _normalize_risk_level(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "high": "high",
    }
    return mapping.get(token, token or "low")


def _extract_questionnaire_safety_flags(intelligence: Dict[str, Any]) -> Set[str]:
    flags: Set[str] = set()
    questionnaire = intelligence.get("questionnaire") or {}
    analysis = questionnaire.get("analysis") or {}

    for raw_flag in _as_list(analysis.get("safety_flags")):
        normalized = _normalize_token(raw_flag)
        if normalized:
            flags.add(normalized)

        text = str(raw_flag).lower()
        keyword_map = {
            "fall": "fall",
            "near fall": "near_fall",
            "balance": "balance_problem",
            "swallow": "swallowing_difficulty",
            "chok": "choking",
            "dizz": "dizziness",
            "lightheaded": "lightheadedness",
            "constipat": "constipation",
            "bowel": "bowel_change",
            "sleep": "sleep_problem",
            "fatigue": "fatigue",
            "tired": "fatigue",
            "weight": "weight_change",
            "appetite": "appetite_change",
            "anxiety": "anxiety",
            "mood": "low_mood",
            "daily": "daily_function_decline",
        }
        for keyword, normalized_flag in keyword_map.items():
            if keyword in text:
                flags.add(normalized_flag)

    return flags


def _extract_profile(
    intelligence: Dict[str, Any],
    symptom_flags: Optional[Sequence[str]] = None,
    safety_triggers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    risk = intelligence.get("risk_summary") or {}
    fusion = intelligence.get("fusion") or {}
    longitudinal = intelligence.get("longitudinal") or {}
    questionnaire = intelligence.get("questionnaire") or {}

    primary_pattern = _normalize_token(intelligence.get("primary_pattern"))
    detected_patterns = _normalize_set(intelligence.get("detected_patterns") or [])
    if primary_pattern:
        detected_patterns.add(primary_pattern)

    risk_level = _normalize_risk_level(risk.get("risk_level", "Low"))
    dominant_module = _normalize_token(fusion.get("dominant_module"))

    trend = _normalize_token(longitudinal.get("overall_trend")) if longitudinal else "no_history"
    trend_aliases = {trend}
    if trend in {"", "no_history", "none"}:
        trend_aliases.update({"no_history", "not_available"})

    # Include the immediate previous-screening direction as another useful tag.
    delta_direction = _normalize_token(longitudinal.get("delta_direction")) if longitudinal else ""
    if delta_direction:
        trend_aliases.add(delta_direction)

    reliability = _normalize_token(questionnaire.get("reliability")) if questionnaire else ""

    normalized_symptom_flags = _normalize_set(symptom_flags or [])
    normalized_symptom_flags.update(_extract_questionnaire_safety_flags(intelligence))

    normalized_safety_triggers = _normalize_set(safety_triggers or [])

    insight_types = _normalize_set(
        insight.get("type")
        for insight in (intelligence.get("insights") or [])
        if isinstance(insight, dict) and insight.get("type")
    )

    fusion_analysis = fusion.get("analysis") or {}
    module_probabilities = fusion_analysis.get("module_probabilities") or {}
    module_predictions = intelligence.get("module_predictions") or {}
    speech = intelligence.get("speech") or {}

    def probability_band(value: Any) -> str:
        """Module-specific band used for recommendation focus.

        This is intentionally different from the final Low/Medium/High risk
        label because a module can deserve attention even when the weighted
        overall result is low.
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "unknown"
        if value < 0.30:
            return "low"
        if value < 0.50:
            return "watch"
        return "attention"

    gait_probability = module_probabilities.get("gait")
    speech_probability = module_probabilities.get("speech")
    questionnaire_probability = module_probabilities.get("questionnaire")

    gait_band = probability_band(gait_probability)
    speech_band = probability_band(speech_probability)
    questionnaire_band = probability_band(questionnaire_probability)

    # Recommendation patterns must respect the actual module probabilities,
    # even if a saved prediction label is conservative. This is what allows a
    # 50% gait/speech signal to drive targeted advice while the weighted
    # overall result can still be Low.
    gait_attention = gait_band == "attention"
    speech_attention = speech_band == "attention"
    questionnaire_attention = questionnaire_band == "attention"

    if gait_attention and speech_attention:
        detected_patterns.add("all_pd" if questionnaire_attention else "mixed_pd")
    elif gait_attention:
        detected_patterns.add("gait_questionnaire_pd" if questionnaire_attention else "gait_only_pd")
    elif speech_attention:
        detected_patterns.add("speech_questionnaire_pd" if questionnaire_attention else "speech_only_pd")
    elif questionnaire_attention:
        detected_patterns.add("questionnaire_only")

    speech_threshold_proximity = _normalize_token(speech.get("threshold_proximity"))
    speech_consistency = _normalize_token(speech.get("consistency"))
    discordance_level = _normalize_token(fusion.get("discordance_level"))
    certainty_level = _normalize_token(fusion.get("certainty_level"))
    motor_severity = _normalize_token(questionnaire.get("motor_severity")) if questionnaire else ""

    # Existing analyzers already expose some useful structured clues.
    if "reliability_low" in insight_types:
        detected_patterns.add("low_reliability")
    if "overall_worsening" in insight_types or "delta_worsening" in insight_types:
        detected_patterns.add("worsening_trend")

    return {
        "risk_level": risk_level,
        "primary_pattern": primary_pattern,
        "patterns": detected_patterns,
        "dominant_module": dominant_module,
        "longitudinal_trends": trend_aliases,
        "reliability": reliability,
        "symptom_flags": normalized_symptom_flags,
        "safety_triggers": normalized_safety_triggers,
        "insight_types": insight_types,
        "final_pd_probability": risk.get("final_pd_probability"),
        "final_probability_band": probability_band(risk.get("final_pd_probability")),
        "fusion_confidence": risk.get("fusion_confidence"),
        "module_probabilities": {
            "gait": gait_probability,
            "speech": speech_probability,
            "questionnaire": questionnaire_probability,
        },
        "module_probability_bands": {
            "gait": gait_band,
            "speech": speech_band,
            "questionnaire": questionnaire_band,
        },
        "module_predictions": {
            key: _normalize_token(value)
            for key, value in module_predictions.items()
        },
        "speech_threshold_proximity": speech_threshold_proximity,
        "speech_consistency": speech_consistency,
        "discordance_level": discordance_level,
        "certainty_level": certainty_level,
        "motor_severity": motor_severity,
        "objective_attention_count": int(gait_attention) + int(speech_attention),
        "questionnaire_only_elevated": (
            questionnaire_attention and not gait_attention and not speech_attention
        ),
    }


def _section_mode(section: str, profile: Dict[str, Any]) -> str:
    """Return active/watch/not_needed for fixed recommendation sections.

    Gait and speech are objective recommendation drivers. Questionnaire is
    supporting evidence and cannot by itself trigger gait/speech treatment or
    videos. Lifestyle advice is only activated when the profile contains a
    concrete symptom/safety flag; otherwise the section is intentionally left
    without generic filler.
    """
    bands = profile.get("module_probability_bands") or {}
    gait_band = bands.get("gait", "low")
    speech_band = bands.get("speech", "low")

    if section == "physical_therapy_exercise":
        return "not_needed" if gait_band == "low" else ("watch" if gait_band == "watch" else "active")

    if section == "speech_voice_therapy":
        return "not_needed" if speech_band == "low" else ("watch" if speech_band == "watch" else "active")

    if section == "lifestyle_wellness":
        # Lifestyle content is shown only when there is a concrete reason.
        # A speech module >=50% is one reason because the curated knowledge base
        # contains voice-care hydration guidance. Questionnaire-only lifestyle
        # advice still requires an actually reported symptom flag.
        lifestyle_flags = {
            "constipation", "bowel_change", "sleep_problem", "daytime_sleepiness",
            "fatigue", "weight_change", "appetite_change", "hydration_concern",
            "dizziness", "lightheadedness", "anxiety", "low_mood",
        }
        specific_flag = bool(
            lifestyle_flags.intersection(profile.get("symptom_flags") or set())
        )
        speech_support = speech_band == "attention"
        return "active" if (specific_flag or speech_support) else "not_needed"

    return "active"


# ============================================================================
# METADATA ELIGIBILITY + SCORING
# ============================================================================

def _constraint_values(card: Dict[str, Any], key: str) -> Set[str]:
    applies_when = card.get("applies_when") or {}
    return _normalize_set(_as_list(applies_when.get(key)))


def _matches_any_constraint(allowed: Set[str], actual: Set[str]) -> bool:
    if not allowed or "any" in allowed:
        return True
    return bool(allowed.intersection(actual))


def _is_card_eligible(card: Dict[str, Any], profile: Dict[str, Any]) -> bool:
    # Risk level
    allowed_risks = _constraint_values(card, "risk_levels")
    if allowed_risks and "any" not in allowed_risks:
        if profile["risk_level"] not in allowed_risks:
            return False

    # Pattern
    allowed_patterns = _constraint_values(card, "patterns")
    if not _matches_any_constraint(allowed_patterns, profile["patterns"]):
        return False

    # Dominant module: if a card explicitly requires one, the profile must have it.
    allowed_modules = _constraint_values(card, "dominant_modules")
    if allowed_modules and "any" not in allowed_modules:
        if not profile["dominant_module"] or profile["dominant_module"] not in allowed_modules:
            return False

    # Longitudinal trend
    allowed_trends = _constraint_values(card, "longitudinal_trends")
    if not _matches_any_constraint(allowed_trends, profile["longitudinal_trends"]):
        return False

    # Symptom-specific cards are only eligible when the symptom is actually present.
    required_symptoms = _constraint_values(card, "symptom_flags")
    if required_symptoms and not required_symptoms.intersection(profile["symptom_flags"]):
        return False

    # Urgent cards are only eligible when an explicit safety trigger is present.
    required_triggers = _constraint_values(card, "safety_triggers")
    if required_triggers and not required_triggers.intersection(profile["safety_triggers"]):
        return False

    # These constraints already exist in the JSON knowledge base. The previous
    # retriever ignored them, which made speech/discordance cards eligible for
    # profiles they did not belong to.
    allowed_speech_proximity = _constraint_values(card, "speech_threshold_proximity")
    if allowed_speech_proximity and "any" not in allowed_speech_proximity:
        if profile["speech_threshold_proximity"] not in allowed_speech_proximity:
            return False

    allowed_speech_consistency = _constraint_values(card, "speech_consistency")
    if allowed_speech_consistency and "any" not in allowed_speech_consistency:
        if profile["speech_consistency"] not in allowed_speech_consistency:
            return False

    allowed_discordance = _constraint_values(card, "discordance_levels")
    if allowed_discordance and "any" not in allowed_discordance:
        if profile["discordance_level"] not in allowed_discordance:
            return False

    allowed_certainty = _constraint_values(card, "certainty_levels")
    if allowed_certainty and "any" not in allowed_certainty:
        if profile["certainty_level"] not in allowed_certainty:
            return False

    return True


def _metadata_score(card: Dict[str, Any], profile: Dict[str, Any]) -> float:
    """
    Return a 0..1 personalization score for an already-eligible card.

    Specific matches MUST outrank generic ``any`` cards. The old implementation
    gave a generic card the same metadata score as an exact pattern/module card,
    which is a major reason different screenings retrieved the same advice.
    """
    score = 0.0
    weight = 0.0

    def add(component_weight: float, quality: float) -> None:
        nonlocal score, weight
        weight += component_weight
        score += component_weight * max(0.0, min(float(quality), 1.0))

    risks = _constraint_values(card, "risk_levels")
    if not risks or "any" in risks:
        add(0.16, 0.45)
    else:
        add(0.16, 1.0 if profile["risk_level"] in risks else 0.0)

    patterns = _constraint_values(card, "patterns")
    if not patterns or "any" in patterns:
        add(0.28, 0.30)
    elif profile["primary_pattern"] and profile["primary_pattern"] in patterns:
        add(0.28, 1.0)
    elif patterns.intersection(profile["patterns"]):
        add(0.28, 0.80)
    else:
        add(0.28, 0.0)

    modules = _constraint_values(card, "dominant_modules")
    if not modules or "any" in modules:
        add(0.15, 0.30)
    else:
        add(0.15, 1.0 if profile["dominant_module"] in modules else 0.0)

    trends = _constraint_values(card, "longitudinal_trends")
    if not trends or "any" in trends:
        add(0.10, 0.30)
    else:
        add(0.10, 1.0 if trends.intersection(profile["longitudinal_trends"]) else 0.0)

    symptoms = _constraint_values(card, "symptom_flags")
    if not symptoms:
        add(0.08, 0.25)
    else:
        add(0.08, 1.0 if symptoms.intersection(profile["symptom_flags"]) else 0.0)

    triggers = _constraint_values(card, "safety_triggers")
    if not triggers:
        add(0.08, 0.20)
    else:
        add(0.08, 1.0 if triggers.intersection(profile["safety_triggers"]) else 0.0)

    speech_proximity = _constraint_values(card, "speech_threshold_proximity")
    if not speech_proximity:
        add(0.06, 0.25)
    else:
        add(0.06, 1.0 if profile["speech_threshold_proximity"] in speech_proximity else 0.0)

    speech_consistency = _constraint_values(card, "speech_consistency")
    if not speech_consistency:
        add(0.03, 0.25)
    else:
        add(0.03, 1.0 if profile["speech_consistency"] in speech_consistency else 0.0)

    discordance = _constraint_values(card, "discordance_levels")
    if not discordance:
        add(0.03, 0.25)
    else:
        add(0.03, 1.0 if profile["discordance_level"] in discordance else 0.0)

    certainty = _constraint_values(card, "certainty_levels")
    if not certainty:
        add(0.03, 0.25)
    else:
        add(0.03, 1.0 if profile["certainty_level"] in certainty else 0.0)

    base = score / weight if weight else 0.0

    priority = card.get("priority", 5)
    try:
        priority = int(priority)
    except (TypeError, ValueError):
        priority = 5
    priority_boost = max(0.0, (6 - min(max(priority, 1), 5)) / 100.0)

    return min(1.0, base + priority_boost)


# ============================================================================
# QUERY BUILDING / RETRIEVAL
# ============================================================================

def _build_profile_query(section: str, intelligence: Dict[str, Any], profile: Dict[str, Any]) -> str:
    module_bits = []
    for module in ("gait", "speech", "questionnaire"):
        prediction = profile["module_predictions"].get(module, "unknown")
        band = profile["module_probability_bands"].get(module, "unknown")
        probability = profile["module_probabilities"].get(module)
        try:
            probability_text = f"{float(probability):.3f}"
        except (TypeError, ValueError):
            probability_text = "unknown"
        module_bits.append(
            f"{module} prediction {prediction} probability {probability_text} band {band}"
        )

    parts = [
        SECTION_LABELS.get(section, section),
        f"risk {profile['risk_level']}",
        f"overall probability band {profile['final_probability_band']}",
        f"primary pattern {profile['primary_pattern']}" if profile["primary_pattern"] else "",
        f"dominant module {profile['dominant_module']}" if profile["dominant_module"] else "",
        "patterns " + " ".join(sorted(profile["patterns"])),
        "trend " + " ".join(sorted(profile["longitudinal_trends"])),
        "speech proximity " + profile["speech_threshold_proximity"],
        "speech consistency " + profile["speech_consistency"],
        "discordance " + profile["discordance_level"],
        "certainty " + profile["certainty_level"],
        "motor severity " + profile["motor_severity"],
        " ".join(module_bits),
        "symptoms " + " ".join(sorted(profile["symptom_flags"])),
    ]

    for insight in intelligence.get("insights") or []:
        if not isinstance(insight, dict):
            continue
        text = insight.get("en") or ""
        if text:
            parts.append(text)

    return " ".join(part for part in parts if part).strip()


def _semantic_scores_for_query(query: str) -> np.ndarray:
    build_index()
    if _embedding_matrix is None or not _cards:
        return np.array([], dtype="float32")

    model = _get_embedding_model()
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = np.asarray(query_embedding, dtype="float32")

    # Search all cards because the KB is intentionally small and curated.
    if faiss is not None and _card_index is not None:
        scores, indices = _card_index.search(query_embedding, len(_cards))
        semantic = np.zeros(len(_cards), dtype="float32")
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                semantic[index] = float(score)
        return semantic

    # NumPy fallback when faiss-cpu is unavailable. Embeddings are normalized,
    # so dot product is cosine similarity.
    return np.asarray(_embedding_matrix @ query_embedding[0], dtype="float32")


def _attach_sources(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = []
    for source_id in _as_list(card.get("source_ids")):
        source = _source_registry.get(str(source_id))
        if source:
            sources.append({
                "id": source.get("id"),
                "organization": source.get("organization"),
                "title": source.get("title"),
                "url": source.get("url"),
                "source_type": source.get("source_type"),
            })
    return sources


def retrieve_section_recommendations(
    section: str,
    intelligence: Dict[str, Any],
    top_k: Optional[int] = None,
    symptom_flags: Optional[Sequence[str]] = None,
    safety_triggers: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve personalized cards for one fixed report section.

    Selection order:
    1. Hard metadata eligibility from NeuroSense analyzers.
    2. Metadata personalization score.
    3. Semantic similarity using FAISS + MiniLM.
    4. Keyword overlap as a small tie-breaker.
    5. Required base card is preserved when defined by the section JSON.
    """
    if section not in REPORT_SECTIONS:
        raise ValueError(f"Unknown recommendation section: {section}")

    build_index()
    if not _cards:
        return []

    profile = _extract_profile(intelligence, symptom_flags, safety_triggers)
    section_mode = _section_mode(section, profile)

    # Keep the section in the final report, but do not retrieve unrelated cards
    # just to fill space. services.py will show a short neutral sentence for
    # sections marked not_needed.
    if section_mode in {"not_needed", "watch"}:
        logger.info(
            "RAG section=%s skipped as %s | module_bands=%s",
            section,
            section_mode,
            profile.get("module_probability_bands"),
        )
        return []

    query = _build_profile_query(section, intelligence, profile)
    semantic_scores = _semantic_scores_for_query(query)

    section_cards = [
        (idx, card)
        for idx, card in enumerate(_cards)
        if card.get("section") == section
    ]
    if not section_cards:
        return []

    # Use section policy unless caller overrides it.
    default_max = next(
        (
            card.get("section_max_cards")
            for _, card in section_cards
            if card.get("section_max_cards")
        ),
        3,
    )
    try:
        default_max = int(default_max)
    except (TypeError, ValueError):
        default_max = 3
    limit = max(1, int(top_k if top_k is not None else default_max))

    required_base_id = next(
        (card.get("required_base_card") for _, card in section_cards if card.get("required_base_card")),
        None,
    )

    ranked: List[Dict[str, Any]] = []
    base_result: Optional[Dict[str, Any]] = None

    for global_idx, card in section_cards:
        is_base = bool(required_base_id and card.get("id") == required_base_id)

        # Required base cards are section-level safety/context cards and are always retained.
        if not is_base and not _is_card_eligible(card, profile):
            continue

        semantic_score = float(semantic_scores[global_idx]) if len(semantic_scores) else 0.0
        keyword_score = _keyword_score(query, card.get("embedding_text", ""))
        metadata_score = 1.0 if is_base else _metadata_score(card, profile)

        final_score = (
            METADATA_WEIGHT * metadata_score
            + SEMANTIC_WEIGHT * semantic_score
            + KEYWORD_WEIGHT * keyword_score
        )

        result = {
            "id": card.get("id"),
            "section": section,
            "display_heading": card.get("display_heading", SECTION_LABELS.get(section, section)),
            "title": card.get("title"),
            "topics": card.get("topics", []),
            "recommendation": card.get("recommendation"),
            "personalization_hint": card.get("personalization_hint"),
            "clinical_follow_up": card.get("clinical_follow_up"),
            "neurosense_repeat_plan": card.get("neurosense_repeat_plan"),
            "safety_level": card.get("safety_level"),
            "priority": card.get("priority", 5),
            "action_type": card.get("action_type"),
            "source_ids": card.get("source_ids", []),
            "sources": _attach_sources(card),
            "retrieval": {
                "metadata_score": round(metadata_score, 4),
                "semantic_score": round(semantic_score, 4),
                "keyword_score": round(keyword_score, 4),
                "final_score": round(final_score, 4),
                "is_required_base_card": is_base,
            },
        }

        if is_base:
            base_result = result
        else:
            ranked.append(result)

    ranked.sort(
        key=lambda item: (
            item["retrieval"]["final_score"],
            -int(item.get("priority") or 5),
        ),
        reverse=True,
    )

    output: List[Dict[str, Any]] = []
    if base_result:
        output.append(base_result)

    # For active gait/speech sections, reserve one practical self-care card when
    # the knowledge base provides one. This prevents professional-referral cards
    # from occupying every slot and keeps the care plan useful before escalation.
    if section in {"physical_therapy_exercise", "speech_voice_therapy"}:
        self_care = next(
            (item for item in ranked if _normalize_token(item.get("action_type")) == "self_care"),
            None,
        )
        if self_care and len(output) < limit:
            output.append(self_care)
            ranked = [item for item in ranked if item.get("id") != self_care.get("id")]

    remaining = max(0, limit - len(output))
    output.extend(ranked[:remaining])

    logger.info(
        "RAG section=%s | risk=%s | pattern=%s | dominant=%s | module_bands=%s | selected=%s",
        section,
        profile.get("risk_level"),
        profile.get("primary_pattern"),
        profile.get("dominant_module"),
        profile.get("module_probability_bands"),
        [item.get("id") for item in output],
    )
    return output


def retrieve_report_recommendations(
    intelligence: Dict[str, Any],
    top_k_per_section: Optional[int] = None,
    symptom_flags: Optional[Sequence[str]] = None,
    safety_triggers: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve RAG context for all six existing recommendation sections."""
    return {
        section: retrieve_section_recommendations(
            section=section,
            intelligence=intelligence,
            top_k=top_k_per_section,
            symptom_flags=symptom_flags,
            safety_triggers=safety_triggers,
        )
        for section in REPORT_SECTIONS
    }


# ============================================================================
# PROMPT/OUTPUT FORMATTING FOR THE LATER LLM INTEGRATION
# ============================================================================

def format_retrieved_context(
    retrieved: Dict[str, List[Dict[str, Any]]],
    include_sources: bool = True,
) -> str:
    """
    Convert retrieved cards into a strict, section-preserving context block.

    This is intentionally suitable for a later Gemini/Groq prompt where the LLM
    is instructed to paraphrase ONLY this retrieved content.
    """
    blocks: List[str] = []

    for section in REPORT_SECTIONS:
        cards = retrieved.get(section) or []
        heading = SECTION_LABELS.get(section, section)
        lines = [f"## {heading}"]

        if not cards:
            lines.append("- No eligible curated recommendation was retrieved for this section.")
            blocks.append("\n".join(lines))
            continue

        for card in cards:
            lines.append(f"- [{card['id']}] {card.get('recommendation', '')}")

            clinical = card.get("clinical_follow_up")
            if clinical:
                lines.append(f"  Clinical follow-up: {clinical}")

            repeat_plan = card.get("neurosense_repeat_plan")
            if repeat_plan:
                lines.append(f"  NeuroSense monitoring: {repeat_plan}")

            if include_sources:
                for source in card.get("sources") or []:
                    lines.append(
                        "  Source: "
                        f"{source.get('organization', '')} — {source.get('title', '')} — {source.get('url', '')}"
                    )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# ============================================================================
# VALIDATION / DEBUGGING
# ============================================================================

def validate_knowledge_base() -> Dict[str, Any]:
    """Return non-destructive validation information for development/debugging."""
    build_index()

    duplicate_ids: List[str] = []
    unknown_source_ids: List[Dict[str, str]] = []
    section_counts = {section: 0 for section in REPORT_SECTIONS}
    seen: Set[str] = set()

    for card in _cards:
        card_id = str(card.get("id"))
        if card_id in seen:
            duplicate_ids.append(card_id)
        seen.add(card_id)

        section = card.get("section")
        if section in section_counts:
            section_counts[section] += 1

        for source_id in _as_list(card.get("source_ids")):
            if str(source_id) not in _source_registry:
                unknown_source_ids.append({
                    "card_id": card_id,
                    "source_id": str(source_id),
                })

    missing_sections = [section for section, count in section_counts.items() if count == 0]

    return {
        "valid": not duplicate_ids and not unknown_source_ids and not missing_sections,
        "total_cards": len(_cards),
        "total_sources": len(_source_registry),
        "section_counts": section_counts,
        "duplicate_card_ids": sorted(set(duplicate_ids)),
        "unknown_source_ids": unknown_source_ids,
        "missing_sections": missing_sections,
    }
