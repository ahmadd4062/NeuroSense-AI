"""
NeuroSense AI - Recommendation Services
Hybrid architecture: NeuroSense analyzers + curated RAG + OpenRouter language layer.

Methodology
-----------
1. Existing NeuroSense analyzers determine the structured patient profile.
2. The RAG service selects eligible recommendation cards from the curated KB.
3. The resource selector optionally attaches verified video resources.
4. OpenRouter composes warm, patient-facing prose using the profile and the
   retrieved cards as the ONLY source of medical content. The LLM must
   REWRITE card text in second-person warm voice — never quote it verbatim.
5. A validator rejects responses that copy cards verbatim, contain technical
   jargon, or leave sentences unfinished. On failure, the failed sections are
   re-requested with a targeted retry prompt; if that also fails, those
   sections are replaced with deterministic Python text instead of discarding
   the whole response.
6. After LLM output, sections that RAG intentionally skipped (no cards
   retrieved) are forced to their deterministic Python neutral text.

Why OpenRouter instead of Gemini / Groq:
Both Gemini and Groq block Azure East Asia datacenter (Singapore region).
OpenRouter works globally.

Model note
----------
nvidia/nemotron-3-super-120b-a12b:free is a REASONING model. It burns tokens
on internal reasoning before emitting the final JSON. max_tokens must be large
enough to cover reasoning + full JSON. If the JSON is truncated, the helper
_repair_truncated_json attempts to close unbalanced braces/quotes so a partial
response can still be salvaged instead of falling back entirely.
"""

import logging
import json
import re
from typing import Any, Dict, List, Optional

from django.conf import settings

from .analyzers.synthesizer import synthesize_intelligence
from .rag.rag_service import (
    REPORT_SECTIONS,
    SECTION_LABELS,
    format_retrieved_context,
    retrieve_report_recommendations,
)
from .rag.resource_selector import (
    inject_resources_into_report,
    select_report_resources,
)

logger = logging.getLogger(__name__)


# ================================================================
# OPENROUTER CLIENT
# ================================================================
def get_openrouter_client():
    try:
        api_key = getattr(settings, "OPENROUTER_API_KEY", None) or None
        if not api_key:
            import os
            api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in settings or environment")
        from openai import OpenAI
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://neurosense-ai.azurewebsites.net",
                "X-Title": "NeuroSense AI",
            },
        )
    except Exception as exc:
        logger.error("Failed to initialize OpenRouter client: %s", exc)
        return None


RECOMMENDATION_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# Reasoning models need a large output budget: reasoning tokens + JSON tokens.
RECOMMENDATION_MAX_TOKENS = 28000

# Verbatim threshold. Length-aware: short card pools need a higher threshold
# because there is less room to paraphrase while preserving meaning.
VERBATIM_THRESHOLD_LONG = 0.90     # card pool > 60 tokens
VERBATIM_THRESHOLD_SHORT = 0.95    # card pool <= 60 tokens


# ================================================================
# BANNED / TECHNICAL WORD LISTS
# ================================================================
_BANNED_WORDS = re.compile(
    r"\b(confidence|threshold|discordance|certainty|fusion|RAG|algorithm|"
    r"domain|domains|longitudinal|score|model|module|modality|"
    r"percentile|probability value|cutoff|baseline finding)\b",
    re.IGNORECASE,
)

# Content sections owned by the LLM. Safety + summary are Python-owned.
LLM_CONTENT_SECTIONS = ("physical", "speech", "lifestyle", "monitoring", "follow_up")


# ================================================================
# JSON EXTRACTION + REPAIR (reasoning-model friendly)
# ================================================================
def _extract_json_from_response(response) -> str:
    msg = response.choices[0].message

    content = getattr(msg, "content", None)
    if content and content.strip():
        return content.strip()

    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
    )
    if not reasoning:
        return ""

    depth = 0
    json_start = None
    for i, ch in enumerate(reasoning):
        if ch == "{":
            if depth == 0:
                json_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and json_start is not None:
                candidate = reasoning[json_start : i + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    json_start = None

    return ""


def _strip_markdown_fences(raw: str) -> str:
    if not raw:
        return raw
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def _repair_truncated_json(raw: str) -> str:
    if not raw:
        return raw

    unescaped_quotes = len(re.findall(r'(?<!\\)"', raw))
    if unescaped_quotes % 2 == 1:
        raw += '"'

    opens = raw.count("{") - raw.count("}")
    while opens > 0:
        raw += "}"
        opens -= 1

    opens = raw.count("[") - raw.count("]")
    while opens > 0:
        raw += "]"
        opens -= 1

    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return raw


def _parse_llm_json(response) -> dict:
    raw_json = _extract_json_from_response(response)
    raw_json = _strip_markdown_fences(raw_json)

    if not raw_json:
        finish = response.choices[0].finish_reason
        reasoning = (
            getattr(response.choices[0].message, "reasoning", None) or ""
        )[:400]
        logger.error(
            "Empty LLM content | finish=%s | reasoning=%r",
            finish,
            reasoning,
        )
        raise ValueError("Empty response from model")

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("Initial JSON parse failed: %s", exc)
        repaired = _repair_truncated_json(raw_json)
        try:
            payload = json.loads(repaired)
            logger.info("JSON repaired successfully after truncation.")
            return payload
        except json.JSONDecodeError as exc2:
            logger.error(
                "JSON parse failed even after repair: %s | raw_len=%d | tail=%r",
                exc2,
                len(raw_json),
                raw_json[-200:],
            )
            raise


# ================================================================
# POINT-LEVEL HELPERS
# ================================================================
def _trim_words(text, limit):
    words = str(text or "").strip().split()
    if len(words) <= limit:
        return " ".join(words)
    truncated = " ".join(words[:limit])
    last_boundary = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
    )
    if last_boundary > len(truncated) * 0.5:
        return truncated[: last_boundary + 1].strip()
    return truncated.rstrip(" ,;:.") + "…"


def _clean_patient_text(text, limit=50):
    text = str(text or "").strip()

    replacements = [
        (r"\bspeech-language pathologist\b", "speech therapist"),
        (r"\bclinical review\b", "doctor visit"),
        (r"\bclinical assessment\b", "doctor check-up"),
        (r"\bneurological evaluation\b", "check-up with a neurologist"),
        (r"\bmultidisciplinary assessment\b", "assessment with several specialists"),
        (r"\bmultidisciplinary care\b", "help from the right health professionals"),
        (r"\bmultidisciplinary team\b", "team of specialists"),
        (r"\bmultidisciplinary\b", "team-based"),
        (r"\bscreening domains?\b", "screening areas"),
        (r"\bmodel'?s?\s+threshold\b", "screening boundary"),
        (r"\bthreshold\b", "boundary"),
        (r"\bconfidence interval\b", "range"),
        (r"\bconfidence\b", "reliability"),
        (r"\bdiscordance\b", "difference between areas"),
        (r"\bcertainty\b", "reliability"),
        (r"\bfusion\b", "combined"),
        (r"\bRAG\b", ""),
        (r"\balgorithm\b", ""),
        (r"\blongitudinal\b", "over time"),
        (r"\btarget about\b", "drink about"),
        (r"\brestrict fluids\b", "limit fluids"),
        (r"\bfluid restriction\b", "limit on fluids"),
        (r"\bmaintain\b", "keep"),
        (r"\badequate\b", "enough"),
        (r"\bsimultaneous changes\b", "many changes at once"),
        (r"\bwithin safe limits\b", "only as much as feels safe"),
        (r"\bcommunication needs\b", "speaking needs"),
        (r"\bare persistent\b", "keep happening"),
        (r"\bis persistent\b", "keeps happening"),
        (r"\bpersistent\b", "long-lasting"),
        (r"\bmonitor\b", "watch"),
        (r"\boverall score\b", "overall picture"),
        (r"\bscreening score\b", "screening result"),
        (r"\bcombined score\b", "combined picture"),
        (r"\bcombined number\b", "overall picture"),
        (r"\bmodule\b", "area"),
        (r"\bmodules\b", "areas"),
        (r"\bgait-dominant\b", "walking-focused"),
        (r"\bspeech-dominant\b", "voice-focused"),
        (r"\bdominant module\b", "main area"),
        # Grammar tidy-ups caused by replacements above.
        (r"\bonly a overall picture\b", "only the overall picture"),
        (r"\ba overall\b", "an overall"),
        (r"\ba area\b", "an area"),
        (r"\ba assessment\b", "an assessment"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\b0\.\d+\b", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*%", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-")

    return _trim_words(text, limit)


def _normalize_points(value, min_points=1, max_points=3, limit=50):
    if isinstance(value, str):
        raw_points = [
            item.strip()
            for item in re.split(r"[\n•]+", value)
            if item.strip()
        ]
        if len(raw_points) == 1:
            raw_points = [
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+", value)
                if item.strip()
            ]
    elif isinstance(value, list):
        raw_points = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw_points = []

    cleaned = []
    for item in raw_points:
        item = _clean_patient_text(item, limit=limit)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= max_points:
            break

    return cleaned


# ================================================================
# VERBATIM-OVERLAP DETECTION (length-aware)
# ================================================================
def _normalize_tokens(text):
    return re.findall(r"\b[a-z]{3,}\b", str(text or "").lower())


def _verbatim_overlap_ratio(point, card_text):
    point_tokens = set(_normalize_tokens(point))
    if not point_tokens:
        return 0.0
    card_tokens = set(_normalize_tokens(card_text))
    if not card_tokens:
        return 0.0
    return len(point_tokens.intersection(card_tokens)) / len(point_tokens)


def _build_card_text_pool(retrieved):
    mapping = {
        "physical": "physical_therapy_exercise",
        "speech": "speech_voice_therapy",
        "lifestyle": "lifestyle_wellness",
        "monitoring": "symptom_monitoring",
        "medical": "medical_consultation",
        "follow_up": "follow_up_schedule",
    }
    pool = {}
    for output_key, rag_section in mapping.items():
        cards = retrieved.get(rag_section) or []
        pool[output_key] = " ".join(
            str(card.get("recommendation") or "") for card in cards
        )
    return pool


def _verbatim_threshold_for(pool_text):
    """Short card pools get a looser threshold; long pools keep 0.90."""
    token_count = len(_normalize_tokens(pool_text))
    if token_count <= 60:
        return VERBATIM_THRESHOLD_SHORT
    return VERBATIM_THRESHOLD_LONG


def _is_verbatim_copy(point, section_pool):
    if not section_pool:
        return False
    threshold = _verbatim_threshold_for(section_pool)
    return _verbatim_overlap_ratio(point, section_pool) >= threshold


# ================================================================
# PATIENT PROFILE HINT (feeds the LLM)
# ================================================================
def _patient_profile_hint(intelligence, lang="en"):
    fusion = intelligence.get("fusion") or {}
    fusion_analysis = fusion.get("analysis") or {}
    longitudinal = intelligence.get("longitudinal") or {}
    risk = intelligence.get("risk_summary") or {}
    speech = intelligence.get("speech") or {}
    questionnaire = intelligence.get("questionnaire") or {}

    dominant = str(fusion.get("dominant_module") or "").lower()
    trend = str(longitudinal.get("overall_trend") or "").lower()
    risk_level = str(risk.get("risk_level") or "Unknown")
    module_probs = fusion_analysis.get("module_probabilities") or {}
    module_predictions = intelligence.get("module_predictions") or {}

    def fmt_prob(name):
        value = module_probs.get(name)
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return "unknown"

    q_analysis = questionnaire.get("analysis") or {}
    symptom_flags = q_analysis.get("safety_flags") or []

    dominant_label = {
        "gait": "walking and balance",
        "speech": "voice and speech",
        "questionnaire": "day-to-day symptoms",
    }.get(dominant, "day-to-day symptoms")

    trend_label = {
        "worsening": "getting more noticeable over time",
        "significantly_worsening": "clearly getting more noticeable over time",
        "improving": "looking a bit better than before",
        "significantly_improving": "clearly better than before",
        "stable": "about the same as last time",
    }.get(trend, "no prior comparison available")

    return (
        f"Risk level: {risk_level}. "
        f"Main area to focus on: {dominant_label}. "
        f"Gait result: {module_predictions.get('gait', 'Unknown')} ({fmt_prob('gait')}); "
        f"Speech result: {module_predictions.get('speech', 'Unknown')} ({fmt_prob('speech')}); "
        f"Questionnaire result: {module_predictions.get('questionnaire', 'Unknown')} ({fmt_prob('questionnaire')}). "
        f"Speech proximity to screening boundary: {speech.get('threshold_proximity', 'unknown')}; "
        f"Speech recording consistency: {speech.get('consistency', 'unknown')}. "
        f"Reported symptom flags: {', '.join(map(str, symptom_flags)) if symptom_flags else 'none'}. "
        f"Trend compared to previous screenings: {trend_label}."
    )


# ================================================================
# DETERMINISTIC SUMMARY + MEANING (Python-owned)
# ================================================================
def _module_state(value):
    try:
        value = float(value or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value < 0.30:
        return "low"
    if value < 0.50:
        return "watch"
    return "attention"


def _recommendation_policy(intelligence):
    fusion = intelligence.get("fusion") or {}
    analysis = fusion.get("analysis") or {}
    probs = analysis.get("module_probabilities") or {}
    risk = intelligence.get("risk_summary") or {}
    longitudinal = intelligence.get("longitudinal") or {}
    questionnaire = intelligence.get("questionnaire") or {}
    q_analysis = questionnaire.get("analysis") or {}

    gait_state = _module_state(probs.get("gait"))
    speech_state = _module_state(probs.get("speech"))
    questionnaire_state = _module_state(probs.get("questionnaire"))

    raw_flags = [str(x).lower() for x in (q_analysis.get("safety_flags") or [])]
    lifestyle_keywords = (
        "constipat", "bowel", "sleep", "fatigue", "tired", "weight",
        "appetite", "hydration", "dizz", "lightheaded", "anxiety", "mood",
    )
    lifestyle_specific = any(
        keyword in flag
        for flag in raw_flags
        for keyword in lifestyle_keywords
    )

    speech_support_lifestyle = speech_state == "attention"

    trend = str(longitudinal.get("overall_trend") or "").lower()
    worsening = "wors" in trend
    risk_level = str(risk.get("risk_level") or "Low").lower()

    return {
        "gait": gait_state,
        "speech": speech_state,
        "questionnaire": questionnaire_state,
        "questionnaire_only_elevated": (
            questionnaire_state == "attention"
            and gait_state != "attention"
            and speech_state != "attention"
        ),
        "physical": "active" if gait_state == "attention" else gait_state,
        "speech_section": "active" if speech_state == "attention" else speech_state,
        "lifestyle": "active" if (lifestyle_specific or speech_support_lifestyle) else "not_needed",
        "lifestyle_reason": (
            "reported_symptom" if lifestyle_specific
            else "speech_support" if speech_support_lifestyle
            else "none"
        ),
        "medical": "active",
        "monitoring": "active",
        "follow_up": "active",
        "worsening": worsening,
        "risk_level": risk_level,
        "questionnaire_flags": raw_flags,
    }


def _module_trend(intelligence, module: str) -> str:
    longitudinal = intelligence.get("longitudinal") or {}
    if not longitudinal:
        return ""
    analysis = longitudinal.get("analysis") or {}
    value = (
        analysis.get(f"{module}_trend")
        or longitudinal.get(f"{module}_trend")
        or ""
    )
    value = str(value).lower()
    if "worsening" in value:
        return "worsening"
    if "improving" in value:
        return "improving"
    if "stable" in value:
        return "stable"
    return ""


def _has_previous_screening(intelligence) -> bool:
    longitudinal = intelligence.get("longitudinal") or {}
    return longitudinal.get("previous_pd_probability") is not None


def _risk_aware_opener(risk_level: str, lang: str) -> str:
    risk = (risk_level or "Low").lower()
    if lang == "ur":
        return {
            "low": "اس اسکریننگ کے نتائج زیادہ تر مستحکم ہیں۔ ",
            "medium": "اس اسکریننگ میں کچھ ایسی تبدیلیاں ہیں جن پر نظر رکھنا اچھا رہے گا۔ ",
            "high": "اس اسکریننگ میں کچھ تبدیلیاں ہیں جن پر جلد توجہ دینا بہتر رہے گا۔ ",
        }.get(risk, "")
    return {
        "low": "This screening looks mostly steady. ",
        "medium": "This screening shows a few changes that are worth keeping an eye on. ",
        "high": "This screening shows some changes that are worth paying attention to soon. ",
    }.get(risk, "")


def _describe_module_against_trend(
    module_name_en: str,
    module_name_ur: str,
    module_state: str,
    trend: str,
    has_previous: bool,
    lang: str,
) -> str:
    plural_en = " and " in module_name_en

    if lang == "ur":
        if module_state == "attention":
            if has_previous and trend == "worsening":
                return f"{module_name_ur} میں تبدیلیاں پہلے سے زیادہ نمایاں ہیں، جن پر توجہ دینا اچھا رہے گا"
            if has_previous and trend == "improving":
                return f"{module_name_ur} پہلے سے بہتر ہیں، لیکن ان پر نظر رکھنا ابھی بھی اچھا ہے"
            if has_previous and trend == "stable":
                return f"{module_name_ur} تقریباً پہلے جیسے ہیں، لیکن ان پر توجہ دینا ابھی بھی مفید ہے"
            return f"{module_name_ur} پر اس وقت توجہ دینا اچھا رہے گا"
        if module_state == "watch":
            if has_previous and trend == "worsening":
                return f"{module_name_ur} میں ہلکی تبدیلیاں ہیں جو پہلے سے زیادہ نمایاں ہو رہی ہیں"
            if has_previous and trend == "improving":
                return f"{module_name_ur} میں ہلکی تبدیلیاں تھیں جو اب بہتر ہو رہی ہیں"
            if has_previous and trend == "stable":
                return f"{module_name_ur} تقریباً پہلے جیسے ہیں"
            return f"{module_name_ur} میں ہلکی تبدیلیاں ہیں جن پر نظر رکھنا اچھا ہے"
        if has_previous and trend == "improving":
            return f"{module_name_ur} پہلے سے بہتر ہیں"
        if has_previous and trend == "stable":
            return f"{module_name_ur} تقریباً پہلے جیسے ہیں"
        return f"{module_name_ur} ٹھیک لگ رہے ہیں"

    verb_has = "have" if plural_en else "has"
    verb_looks = "look" if plural_en else "looks"

    if module_state == "attention":
        if has_previous and trend == "worsening":
            return f"your {module_name_en} {verb_has} become a bit more noticeable than before, so it's worth attention now"
        if has_previous and trend == "improving":
            return f"your {module_name_en} {verb_looks} better than before, but it's still worth keeping an eye on"
        if has_previous and trend == "stable":
            return f"your {module_name_en} {verb_has} been steady compared to last time, though it still needs some attention"
        return f"your {module_name_en} needs some attention right now"
    if module_state == "watch":
        if has_previous and trend == "worsening":
            return f"your {module_name_en} is showing small changes that are becoming more noticeable"
        if has_previous and trend == "improving":
            return f"your {module_name_en} is showing small changes that are looking better than before"
        if has_previous and trend == "stable":
            return f"your {module_name_en} {verb_looks} about the same as last time"
        return f"there are some mild changes in your {module_name_en} that are worth keeping an eye on"
    if has_previous and trend == "improving":
        return f"your {module_name_en} {verb_looks} a bit better than last time"
    if has_previous and trend == "stable":
        return f"your {module_name_en} {verb_looks} about the same as last time"
    return f"your {module_name_en} {verb_looks} steady"


def _urdu_risk_label(value):
    value = str(value or "").strip().lower()
    return {"low": "کم", "medium": "درمیانی", "high": "زیادہ"}.get(value, str(value or ""))


def _summary_and_meaning(intelligence, lang="en"):
    policy = _recommendation_policy(intelligence)
    risk = str((intelligence.get("risk_summary") or {}).get("risk_level") or "Low")

    gait = policy["gait"]
    speech = policy["speech"]
    questionnaire = policy["questionnaire"]

    has_prev = _has_previous_screening(intelligence)
    gait_trend = _module_trend(intelligence, "gait")
    speech_trend = _module_trend(intelligence, "speech")

    opener = _risk_aware_opener(risk, lang)

    if lang == "ur":
        gait_phrase = _describe_module_against_trend("gait", "چال", gait, gait_trend, has_prev, "ur")
        speech_phrase = _describe_module_against_trend("speech", "آواز یا بولنے", speech, speech_trend, has_prev, "ur")

        if gait == "attention" and speech == "attention":
            body = f"{gait_phrase}، اور {speech_phrase}۔ آئیے ان دونوں حصوں پر مل کر توجہ دیں۔"
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "چلنے، توازن، آواز اور بولنے میں ہونے والی تبدیلیوں کو الگ الگ نوٹ کریں۔",
            ]
        elif gait == "attention":
            body = f"{gait_phrase}۔ {speech_phrase}۔"
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "چلنے، مڑنے اور توازن میں ہونے والی تبدیلیوں پر خاص نظر رکھیں۔",
            ]
        elif speech == "attention":
            body = f"{speech_phrase}۔ {gait_phrase}۔"
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "آواز کی طاقت، وضاحت، وقفوں اور سمجھ آنے میں ہونے والی تبدیلیوں کو نوٹ کریں۔",
            ]
        elif policy["questionnaire_only_elevated"]:
            body = (
                "چال اور آواز میں کوئی خاص تبدیلی نہیں آئی، لیکن سوالنامے میں کچھ علامات بتائی گئی ہیں۔ "
                "صرف وہ علامات نوٹ کریں جو آپ واقعی روزمرہ محسوس کر رہے ہیں۔"
            )
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "سوالنامہ صرف معاون معلومات ہے؛ اس اسکریننگ میں چال اور آواز نے وہی تبدیلیاں ظاہر نہیں کیں۔",
            ]
        else:
            body = f"{gait_phrase}، اور {speech_phrase}۔"
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "صرف نئی یا مسلسل ہونے والی روزمرہ تبدیلیوں پر نظر رکھیں۔",
            ]

        summary = opener + body
    else:
        gait_phrase = _describe_module_against_trend("walking and balance", "", gait, gait_trend, has_prev, "en")
        speech_phrase = _describe_module_against_trend("voice and speech", "", speech, speech_trend, has_prev, "en")

        if gait == "attention" and speech == "attention":
            body = (
                f"Right now, {gait_phrase}, and {speech_phrase}. "
                "Let's focus on both of these areas together over the next few weeks."
            )
            meaning = [
                f"Your combined screening level is {risk}.",
                "Track walking, balance, voice, and speech changes separately, rather than relying only on the overall result.",
            ]
        elif gait == "attention":
            body = f"Right now, {gait_phrase}. {speech_phrase.capitalize()}."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Focus on changes in walking, turning, and balance instead of treating every screening area the same way.",
            ]
        elif speech == "attention":
            body = f"Right now, {speech_phrase}. {gait_phrase.capitalize()}."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Focus on voice volume, clarity, and pauses rather than treating every screening area the same way.",
            ]
        elif policy["questionnaire_only_elevated"]:
            body = (
                "Your walking and speech look steady this time, but your questionnaire reported a few symptoms. "
                "Let's focus on the symptoms you actually notice day to day."
            )
            meaning = [
                f"Your combined screening level is {risk}.",
                "The questionnaire adds supporting information. In this screening, walking and speech did not show matching changes.",
            ]
        else:
            body = f"{gait_phrase}, and {speech_phrase}."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Keep normal awareness of any new or long-lasting changes in your day-to-day routine.",
            ]

        summary = opener + body

    if questionnaire == "attention" and not policy["questionnaire_only_elevated"]:
        if lang == "ur":
            meaning[-1] += " سوالنامے میں بتائی گئی علامات کو بھی معاون معلومات کے طور پر نوٹ کرتے رہیں۔"
        else:
            meaning[-1] += " Also keep an eye on the symptoms you reported in the questionnaire, as supporting information."

    return summary, meaning


# ================================================================
# PYTHON-OWNED SAFETY SECTION
# ================================================================
def _normalized_questionnaire_flags(intelligence):
    questionnaire = intelligence.get("questionnaire") or {}
    analysis = questionnaire.get("analysis") or {}
    flags = []
    for raw in analysis.get("safety_flags") or []:
        text = str(raw or "").strip().lower()
        if text:
            flags.append(text)
    return flags


def _watch_closely_points(intelligence, lang="en"):
    policy = _recommendation_policy(intelligence)
    gait = policy["gait"]
    speech = policy["speech"]
    flags = _normalized_questionnaire_flags(intelligence)

    swallowing_reported = any("swallow" in f or "chok" in f for f in flags)
    balance_reported = any("fall" in f or "balance" in f for f in flags)

    if lang == "ur":
        module_points = []
        if gait == "attention":
            module_points.append(
                "چلنے پر نظر رکھیں۔ چھوٹے یا گھسٹتے قدم، مڑنے میں دشواری، "
                "یا توازن کے لیے زیادہ سہارے کی ضرورت اہم تبدیلیاں ہیں۔"
            )
        elif gait == "watch":
            module_points.append(
                "اپنے قدموں کی لمبائی، گھسٹنے، مڑنے اور توازن پر نظر رکھیں، "
                "اور دیکھیں کہ کیا یہ تبدیلیاں زیادہ ہو رہی ہیں۔"
            )

        if speech == "attention":
            module_points.append(
                "بولنے پر بھی نظر رکھیں۔ آواز کا نرم ہونا، الفاظ کا کم واضح ہونا، "
                "زیادہ ہچکچاہٹ، لمبے وقفے، یا دوسروں کا سمجھ نہ پانا اہم تبدیلیاں ہیں۔"
            )
        elif speech == "watch":
            module_points.append(
                "آواز کے نرم ہونے، الفاظ کے کم واضح ہونے، ہچکچاہٹ یا لمبے وقفوں پر نظر رکھیں، "
                "اور دیکھیں کہ کیا یہ تبدیلیاں مسلسل رہتی ہیں۔"
            )

        support_point = ""
        if swallowing_reported:
            support_point = (
                "آپ نے نگلنے یا دم گھٹنے جیسی مشکل بتائی ہے۔ اگر کھانے یا پینے کے ساتھ "
                "یہ مسئلہ بار بار ہو تو ڈاکٹر سے مشورہ کریں، صرف اسکریننگ پر انحصار نہ کریں۔"
            )
        elif balance_reported and gait != "attention":
            support_point = (
                "آپ نے توازن یا گرتے گرتے بچنے کی مشکل بتائی ہے۔ یہ کب اور کتنی بار ہوتا ہے، نوٹ کریں، "
                "خاص طور پر اگر یہ بڑھ رہا ہو۔"
            )

        if gait == "attention" and speech == "attention":
            escalation = (
                "اگر چلنا یا بولنا اچانک واضح طور پر بگڑ جائے، "
                "یا بغیر سہارے چلنا محفوظ نہ رہے تو فوراً طبی مدد لیں۔"
            )
        elif gait == "attention":
            escalation = (
                "اگر چلنا اچانک بہت مشکل ہو جائے، توازن کی وجہ سے بار بار گرتے گرتے بچیں، "
                "یا بغیر سہارے محفوظ چلنا ممکن نہ رہے تو فوراً طبی مدد لیں۔"
            )
        elif speech == "attention":
            escalation = (
                "اگر بولنا اچانک معمول سے بہت کم واضح ہو جائے یا دوسروں کے لیے سمجھنا بہت مشکل ہو جائے تو فوراً طبی مدد لیں۔"
            )
        elif gait == "watch" or speech == "watch":
            escalation = (
                "اگر یہ تبدیلیاں واضح طور پر بڑھیں یا روزمرہ کام متاثر کرنے لگیں تو "
                "اگلی اسکریننگ کا انتظار کرنے کے بجائے ڈاکٹر سے مشورہ کریں۔"
            )
        elif policy.get("questionnaire_only_elevated"):
            escalation = (
                "صرف وہی علامات نوٹ کریں جو آپ نے سوالنامے میں بتائیں۔ "
                "چال یا آواز سے متعلق کوئی نئی تنبیہی علامت خود سے فرض نہ کریں۔"
            )
        else:
            escalation = (
                "اس اسکریننگ میں چال یا آواز سے متعلق کوئی خاص تنبیہی علامت سامنے نہیں آئی۔ "
                "پھر بھی کوئی نئی یا مسلسل پریشان کن علامت ہو تو نظر انداز نہ کریں۔"
            )

        points = list(module_points)
        if support_point and len(points) < 2:
            points.append(support_point)
        points.append(escalation)
        return points[:3]

    module_points = []
    if gait == "attention":
        module_points.append(
            "Keep an eye on how you walk. Shorter or shuffling steps, feet feeling stuck, "
            "harder turning, or needing more support are worth noting."
        )
    elif gait == "watch":
        module_points.append(
            "Watch your step length, shuffling, turning, and balance, and note whether "
            "any of these changes are becoming more frequent."
        )

    if speech == "attention":
        module_points.append(
            "Also watch your speech. A softer voice, less clear words, more hesitation, "
            "longer pauses, or difficulty being understood are worth noting."
        )
    elif speech == "watch":
        module_points.append(
            "Watch for a softer voice, unclear words, more hesitation, or longer pauses, "
            "and note whether these changes are long-lasting."
        )

    support_point = ""
    if swallowing_reported:
        support_point = (
            "You reported some difficulty with swallowing or choking. If food or drink "
            "repeatedly causes this, please see a doctor rather than relying only on this screening."
        )
    elif balance_reported and gait != "attention":
        support_point = (
            "You reported some trouble with balance or near-falls. Note when this happens "
            "and whether it is becoming more frequent."
        )

    if gait == "attention" and speech == "attention":
        escalation = (
            "If your walking or speech suddenly becomes clearly worse, or walking is no longer "
            "safe without support, please seek medical help promptly."
        )
    elif gait == "attention":
        escalation = (
            "If walking suddenly becomes much harder, balance causes repeated near-falls, "
            "or you cannot walk safely without support, please seek medical help promptly."
        )
    elif speech == "attention":
        escalation = (
            "If speech suddenly becomes much harder to understand than usual, or changes sharply "
            "over a short time, please seek medical help promptly."
        )
    elif gait == "watch" or speech == "watch":
        escalation = (
            "If these changes clearly worsen or begin to affect daily activities, "
            "please see a doctor rather than waiting for the next screening."
        )
    elif policy.get("questionnaire_only_elevated"):
        escalation = (
            "Watch only the symptoms you actually reported in the questionnaire. "
            "Do not assume a gait or speech warning that was not detected."
        )
    else:
        escalation = (
            "No specific walking or speech warning was detected in this screening. "
            "Still, please pay attention to any new or persistent symptom that worries you."
        )

    points = list(module_points)
    if support_point and len(points) < 2:
        points.append(support_point)
    points.append(escalation)

    deduped = []
    for point in points:
        if point not in deduped:
            deduped.append(point)
    return deduped[:3]


def _neutral_section_points(section_key, policy, lang="en"):
    mode = {
        "physical": policy["physical"],
        "speech": policy["speech_section"],
        "lifestyle": policy["lifestyle"],
    }.get(section_key, "active")

    if lang == "ur":
        if section_key == "physical" and mode == "low":
            return ["آپ کی چال کے نتائج ابھی ٹھیک ہیں، اس لیے کسی خاص ورزش یا فزیکل تھراپی کی ضرورت نہیں ہے۔"]
        if section_key == "physical" and mode == "watch":
            return ["آپ کی چال میں ہلکی تبدیلیاں ہیں جن پر نظر رکھنا اچھا ہے۔ ابھی کسی خاص ورزش یا ویڈیو کی ضرورت نہیں۔"]
        if section_key == "speech" and mode == "low":
            return ["آپ کی آواز کے نتائج ابھی ٹھیک ہیں، اس لیے کسی خاص آواز کی مشق یا ویڈیو کی ضرورت نہیں ہے۔"]
        if section_key == "speech" and mode == "watch":
            return ["آپ کی آواز میں ہلکی تبدیلیاں ہیں جن پر نظر رکھنا اچھا ہے۔ ابھی کسی خاص مشق یا ویڈیو کی ضرورت نہیں۔"]
        if section_key == "lifestyle" and mode == "not_needed":
            return ["اس اسکریننگ سے آپ کی روزمرہ طرزِ زندگی میں کسی اضافی تبدیلی کی ضرورت سامنے نہیں آئی۔"]
        return []

    if section_key == "physical" and mode == "low":
        return ["Your walking looks steady this time, so no specific exercise or physiotherapy plan is needed right now."]
    if section_key == "physical" and mode == "watch":
        return ["There are some mild changes in your walking that are worth watching. No specific exercise video is needed yet."]
    if section_key == "speech" and mode == "low":
        return ["Your voice looks steady this time, so no specific voice exercise plan or video is needed right now."]
    if section_key == "speech" and mode == "watch":
        return ["There are some mild changes in your voice that are worth watching. No specific exercise or video is needed yet."]
    if section_key == "lifestyle" and mode == "not_needed":
        return ["This screening does not add a specific lifestyle change. There's no need to make diet or supplement changes just because this section is here."]
    return []


def _points_from_retrieved(section, retrieved, max_points=3):
    cards = retrieved.get(section) or []
    personalized = [
        card for card in cards
        if not (card.get("retrieval") or {}).get("is_required_base_card")
    ]
    chosen = personalized or cards
    points = []
    for card in chosen:
        text = _clean_patient_text(
            card.get("patient_action")
            or card.get("recommendation")
            or card.get("personalization_hint")
            or "",
            limit=50,
        )
        if text and text not in points:
            points.append(text)
        if len(points) >= max_points:
            break
    return points


# ================================================================
# RAG-EMPTY SECTION ENFORCEMENT
# ================================================================
def _enforce_empty_rag_sections(payload, intelligence, retrieved, lang="en"):
    if not isinstance(payload, dict):
        return payload

    policy = _recommendation_policy(intelligence)
    section_map = _section_map()

    for output_key, rag_section in section_map.items():
        cards = retrieved.get(rag_section) or []
        if cards:
            continue

        neutral = _neutral_section_points(output_key, policy, lang=lang)
        if neutral:
            payload[output_key] = neutral
            logger.info(
                "RAG-empty section '%s' enforced with Python neutral text (no cards retrieved).",
                output_key,
            )

    return payload


# ================================================================
# PROMPT BUILDER
# ================================================================
def build_doctor_prompt(
    intelligence,
    retrieved=None,
    selected_resources=None,
    lang="en",
):
    retrieved = retrieved or {}
    rag_context = format_retrieved_context(retrieved, include_sources=False)
    profile_hint = _patient_profile_hint(intelligence, lang=lang)

    if lang == "ur":
        return f"""آپ NeuroSense کے لیے warm، patient-friendly care guidance لکھ رہے ہیں۔
لکھتے وقت ایسے محسوس کریں جیسے آپ ایک تجربہ کار نیورولوجسٹ ہیں جو مریض کے ساتھ بیٹھے بات کر رہے ہیں۔

مریض کا پس منظر:
{profile_hint}

CURATED RAG CONTEXT — صرف اسی سے medical content لیا جا سکتا ہے:
{rag_context}

⚠️ سب سے اہم اصول — REWRITE, DO NOT COPY:
RAG context کے جملے clinician کے لیے لکھے گئے ہیں۔ آپ کو انہیں مریض سے مخاطب کرتے ہوئے "آپ" اور "ہم" کے ساتھ دوبارہ لکھنا ہے۔ RAG کے الفاظ حرف بہ حرف کاپی نہ کریں۔ ہر جملہ نئے الفاظ میں ہو، صرف مطلب وہی رہے۔

مثال:
RAG: "Follow up should pay attention to walking, turning, balance and falls."
❌ غلط (کاپی): "فالو اپ میں چلنے، مڑنے، توازن اور گرنے پر توجہ دیں۔"
✅ صحیح (rewrite): "اگلی ملاقات میں ہم آپ کے چلنے، مڑنے اور توازن کے بارے میں بات کریں گے، اور یہ بھی دیکھیں گے کہ کہیں آپ کو گرنے کا خدشہ تو نہیں۔"

دیگر اہم اصول:
1. ہر point 20 سے 45 الفاظ کے درمیان ہو، 2-3 مکمل چھوٹے جملے۔ ادھورا جملہ نہ چھوڑیں۔
2. Summary اور Meaning خود بخود بھر جائیں گے۔
3. ہر section دکھائیں۔ اگر کسی section میں RAG context خالی ہو، تو صرف لکھیں: "اس اسکریننگ سے اس حصے میں کسی خاص تبدیلی کی ضرورت سامنے نہیں آئی۔"
4. Physical اور Speech میں user کے خود کرنے والے actions پہلے آئیں، professional referral آخر میں۔
5. Lifestyle صرف RAG میں reason ہو تو specific advice دے۔
6. Monitoring میں کن changes کو note کرنا ہے، نرم لہجے میں لکھیں۔
7. "Changes to Watch Closely" کے لیے غیر متعلقہ emergency نہ بنائیں۔
8. Follow-up میں next screening کی practical guidance دیں — RAG کے الفاظ کاپی نہ کریں۔
9. confidence, threshold, score, model, fusion, module, domain جیسے الفاظ نہ لکھیں۔
10. percentages/decimals نہ لکھیں۔
11. YouTube link نہ بنائیں۔
12. خالص آسان اردو لکھیں۔
13. "When X..." سے شروع نہ کریں — سیدھا مریض سے مخاطب ہوں۔
14. صحیح JSON واپس کریں، کوئی markdown fence نہ لگائیں۔

JSON:
{{
  "summary": "short",
  "meaning": ["point 1", "point 2"],
  "physical": ["point 1", "point 2", "point 3"],
  "speech": ["point 1", "point 2", "point 3"],
  "lifestyle": ["point 1", "point 2", "point 3"],
  "monitoring": ["point 1", "point 2", "point 3"],
  "medical": ["point 1"],
  "follow_up": ["point 1", "point 2", "point 3"]
}}
"""

    return f"""You write warm, personalized patient guidance for NeuroSense.
Write as if you are a caring neurologist sitting with the patient and talking through their results.

PATIENT PROFILE (personalize tone, priority, and emphasis):
{profile_hint}

CURATED RAG CONTEXT — THE ONLY SOURCE OF MEDICAL CONTENT:
{rag_context}

⚠️ MOST IMPORTANT RULE — REWRITE, DO NOT COPY:
The RAG context sentences are written FOR A CLINICIAN. You must REWRITE them in second-person voice directed at the patient, using "you" and "we". Do NOT copy the RAG wording verbatim. Every sentence must use different words — only the meaning stays the same.

Example:
RAG text: "For gait-dominant findings, follow-up should pay particular attention to walking, turning, freezing, balance and falls."
❌ WRONG (copied): "For gait-dominant findings, follow-up should pay particular attention to walking, turning, freezing, balance and falls."
✅ RIGHT (rewritten): "Since walking is the area we're focusing on, let's talk about your walking, turning, balance, and any falls at your next check-up."

STRICT RULES:
1. Each point must be 20 to 45 words, made of 2-3 short COMPLETE sentences. Never leave a sentence unfinished.
2. Summary and Meaning will be auto-filled. Keep them short in JSON.
3. Keep all six sections. If a section has NO RAG content, write only: "This screening does not add a specific change for this area."
4. Active Physical/Speech sections: give 2-3 warm actions. Something the user can do first; professional assessment goes last.
5. Low/watch Physical/Speech sections: one warm neutral line is enough.
6. Lifestyle: only use food or fluid guidance that appears in the RAG context.
7. Symptom Monitoring: name WHAT to watch, in warm plain words.
8. Changes to Watch Closely: only the affected area and reported symptoms.
9. Never say the user does not need medical care.
10. Follow-up: warm, practical next-screening guidance addressed to "you". Rewrite, do NOT quote.
11. Do not repeat the same recommendation across sections.
12. NEVER expose these words: confidence, threshold, score, model, fusion, discordance, certainty, RAG, algorithm, domain, domains, longitudinal, module, modality, percentile.
13. No percentages or decimals.
14. No YouTube/resource URLs.
15. Everyday English. Short words, short sentences. Use "you" and "we".
16. Prefer "Notice if..." over "Note whether...". Prefer "Keep an eye on..." over "Monitor...".
17. PERSONALIZE: Adapt each section to the dominant area and trend.
18. NEVER start a point with "When X...", "For X-dominant findings...", "In cases of...".
19. Return ONLY valid JSON. Do NOT wrap in markdown fences.

Return ONLY valid JSON:
{{
  "summary": "1-2 warm short sentences (auto-replaced)",
  "meaning": ["warm point 1", "warm point 2"],
  "physical": ["warm action 1", "warm action 2", "warm action 3"],
  "speech": ["warm action 1", "warm action 2", "warm action 3"],
  "lifestyle": ["warm action 1", "warm action 2", "warm action 3"],
  "monitoring": ["warm action 1", "warm action 2", "warm action 3"],
  "medical": ["warm watch point"],
  "follow_up": ["warm action 1", "warm action 2", "warm action 3"]
}}
"""


# ================================================================
# VALIDATOR
# ================================================================
def _validate_llm_payload(payload, retrieved=None, lang="en"):
    """Return (ok, reason, bad_sections). bad_sections is a list of output keys."""
    if not isinstance(payload, dict):
        return False, "payload not a dict", []

    card_pool = _build_card_text_pool(retrieved or {})
    bad_sections = []

    for key in ("physical", "speech", "lifestyle", "monitoring", "medical", "follow_up"):
        value = payload.get(key)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not value:
            return False, f"section '{key}' is missing or empty", [key]

        section_ok = True
        for point in value:
            text = str(point or "").strip()
            if len(text.split()) < 6:
                return False, f"point too short in '{key}'", [key]

            if _BANNED_WORDS.search(text):
                reason = f"banned word in '{key}': {_BANNED_WORDS.search(text).group(0)}"
                return False, reason, [key]

            if _is_verbatim_copy(text, card_pool.get(key, "")):
                reason = f"point in '{key}' is copied verbatim from RAG"
                return False, reason, [key]

    meaning = payload.get("meaning")
    if isinstance(meaning, str):
        meaning = [meaning]
    if not isinstance(meaning, list) or not meaning:
        return False, "meaning missing", []

    return True, "", bad_sections


# ================================================================
# TARGETED RETRY (only fix the failed sections)
# ================================================================
def _targeted_retry(
    client,
    payload,
    bad_sections,
    intelligence,
    retrieved,
    lang="en",
):
    """Ask the LLM to rewrite only the failed sections with strict instructions."""
    if not bad_sections:
        return payload

    card_pool = _build_card_text_pool(retrieved or {})

    # Build a short prompt that shows each failed section + its source card text
    section_lines = []
    for key in bad_sections:
        source = card_pool.get(key, "")[:600]
        section_lines.append(
            f"Section '{key}' — source card text:\n{source}\n"
            f"You MUST rewrite this in second-person warm doctor voice. "
            f"Use different words. Do NOT copy the source wording."
        )

    if lang == "ur":
        retry_instruction = (
            "آپ کے پچھلے جواب میں کچھ سیکشن RAG متن سے کاپی کیے گئے تھے۔ "
            "براہ کرم صرف درج ذیل سیکشن دوبارہ لکھیں، ہر جملہ نئے الفاظ میں، "
            "مریض سے سیدھا مخاطب ہو کر۔ صرف valid JSON واپس کریں جس میں صرف یہ سیکشن ہوں۔"
        )
    else:
        retry_instruction = (
            "Some sections in your previous response were copied verbatim from the "
            "RAG source. Rewrite ONLY the sections listed below in second-person "
            "warm doctor voice, using different words, keeping the same meaning. "
            "Return ONLY valid JSON containing just those sections."
        )

    retry_prompt = (
        f"{retry_instruction}\n\n"
        + "\n\n".join(section_lines)
        + "\n\nReturn JSON like: {"
        + ", ".join(f'"{k}": ["rewritten point 1", "rewritten point 2"]' for k in bad_sections)
        + "}"
    )

    try:
        response = client.chat.completions.create(
            model=RECOMMENDATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rewrite patient-facing recommendation sections in "
                        "second-person warm doctor voice, using different words than "
                        "the source. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.55,
            top_p=0.90,
            max_tokens=4000,
        )
        retry_payload = _parse_llm_json(response)
    except Exception as exc:
        logger.warning("Targeted retry failed for sections %s: %s", bad_sections, exc)
        return payload

    # Merge rewritten sections back into the payload; validate each one.
    for key in bad_sections:
        rewritten = retry_payload.get(key)
        if isinstance(rewritten, str):
            rewritten = [rewritten]
        if not isinstance(rewritten, list) or not rewritten:
            continue

        section_ok = True
        for point in rewritten:
            text = str(point or "").strip()
            if len(text.split()) < 6:
                section_ok = False
                break
            if _BANNED_WORDS.search(text):
                section_ok = False
                break
            if _is_verbatim_copy(text, card_pool.get(key, "")):
                section_ok = False
                break

        if section_ok:
            payload[key] = rewritten
            logger.info("Targeted retry accepted for section '%s'.", key)
        else:
            logger.info(
                "Targeted retry still failed for section '%s'. Will use Python fallback.",
                key,
            )

    return payload


# ================================================================
# URDU CLEANUP + REPORT FORMATTING
# ================================================================
def _clean_urdu_output_text(text):
    text = str(text or "")
    replacements = [
        (r"\bMedium\b", "درمیانی"),
        (r"\bLow\b", "کم"),
        (r"\bHigh\b", "زیادہ"),
        (r"\bmedical advice\b", "طبی مشورہ"),
        (r"\bgait\b", "چال"),
        (r"\bspeech\b", "آواز یا بولنے"),
        (r"\bbalance\b", "توازن"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _format_action_report(payload, lang="en"):
    if lang == "ur":
        headings = [
            ("خلاصہ", "summary"),
            ("آپ کے نتائج کا مطلب", "meaning"),
            ("1. فزیکل تھراپی اور ورزش", "physical"),
            ("2. اسپیچ اور آواز کی تھراپی", "speech"),
            ("3. طرز زندگی اور صحت", "lifestyle"),
            ("4. علامات کی نگرانی", "monitoring"),
            ("5. قریب سے دیکھنے والی تبدیلیاں", "medical"),
            ("6. فالو اپ شیڈول", "follow_up"),
        ]
    else:
        headings = [
            ("Your Summary", "summary"),
            ("What Your Results Mean", "meaning"),
            ("1. Physical Therapy & Exercise", "physical"),
            ("2. Speech & Voice Therapy", "speech"),
            ("3. Lifestyle & Wellness", "lifestyle"),
            ("4. Symptom Monitoring", "monitoring"),
            ("5. Changes to Watch Closely", "medical"),
            ("6. Follow-up Schedule", "follow_up"),
        ]

    lines = []
    summary = _clean_patient_text(payload.get("summary", ""), limit=70)
    lines.extend([f"## {headings[0][0]}", summary, ""])

    for heading, key in headings[1:]:
        points = _normalize_points(payload.get(key), min_points=1, max_points=3, limit=50)
        if not points:
            raise ValueError(f"Recommendation section '{key}' returned no useful content.")
        lines.append(f"## {heading}")
        for point in points:
            lines.append(f"- {point}")
        lines.append("")

    return "\n".join(lines).strip()


# ================================================================
# PYTHON-FIRST FALLBACK REPORT
# ================================================================
def _section_map():
    return {
        "physical": "physical_therapy_exercise",
        "speech": "speech_voice_therapy",
        "lifestyle": "lifestyle_wellness",
        "monitoring": "symptom_monitoring",
        "follow_up": "follow_up_schedule",
    }


def get_fallback_recommendations(
    intelligence,
    lang="en",
    retrieved=None,
    selected_resources=None,
):
    retrieved = retrieved or retrieve_report_recommendations(intelligence)

    summary, meaning = _summary_and_meaning(intelligence, lang=lang)
    policy = _recommendation_policy(intelligence)

    payload = {
        "summary": summary,
        "meaning": meaning,
        "medical": _watch_closely_points(intelligence, lang=lang),
    }

    for key, rag_section in _section_map().items():
        neutral = _neutral_section_points(key, policy, lang=lang)
        if neutral:
            payload[key] = neutral
            continue

        points = _points_from_retrieved(rag_section, retrieved, max_points=3)
        if not points:
            points = (
                ["No additional targeted recommendation is needed for this area from the current screening."]
                if lang != "ur"
                else ["موجودہ اسکریننگ کے مطابق اس حصے کے لیے مزید مخصوص سفارش درکار نہیں۔"]
            )
        payload[key] = points[:3]

    report = _format_action_report(payload, lang=lang)
    if lang == "ur":
        report = _clean_urdu_output_text(report)

    try:
        if selected_resources is None:
            selected_resources = select_report_resources(intelligence, retrieved)
        report = inject_resources_into_report(report, selected_resources or {}, lang=lang)
    except Exception as exc:
        logger.warning("Could not attach resource in fallback: %s", exc)
    return report


# ================================================================
# MAIN ENTRY POINT
# ================================================================
def generate_ai_recommendations(
    user,
    fusion_result,
    questionnaire_result=None,
    questionnaire_result_id=None,
    screening_id=None,
    lang="en",
):
    try:
        intelligence = synthesize_intelligence(
            user=user,
            fusion_result=fusion_result,
            questionnaire_result=questionnaire_result,
            questionnaire_result_id=questionnaire_result_id,
            screening_id=screening_id,
            lang=lang,
        )
        logger.info(
            "Intelligence synthesized | screening=%s | risk=%s | pattern=%s | dominant=%s | modules=%s | insights=%d",
            screening_id,
            (intelligence.get("risk_summary") or {}).get("risk_level"),
            intelligence.get("primary_pattern"),
            (intelligence.get("fusion") or {}).get("dominant_module"),
            intelligence.get("module_predictions"),
            len(intelligence.get("insights", [])),
        )
    except Exception as exc:
        logger.exception("Intelligence synthesis failed: %s", exc)
        raise Exception(f"Failed to synthesize intelligence: {exc}")

    try:
        retrieved = retrieve_report_recommendations(intelligence)
        counts = {section: len(retrieved.get(section) or []) for section in REPORT_SECTIONS}
        logger.info("RAG recommendations retrieved: %s", counts)
    except Exception as exc:
        logger.exception("Recommendation RAG retrieval failed: %s", exc)
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved={section: [] for section in REPORT_SECTIONS},
            selected_resources={},
        )

    try:
        selected_resources = select_report_resources(
            intelligence=intelligence,
            retrieved_report=retrieved,
        )
    except Exception as exc:
        logger.warning("Resource selection failed: %s", exc)
        selected_resources = {}

    prompt = build_doctor_prompt(
        intelligence=intelligence,
        retrieved=retrieved,
        selected_resources=selected_resources,
        lang=lang,
    )

    client = get_openrouter_client()
    if not client:
        logger.warning("OpenRouter client not available. Using fallback.")
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved=retrieved,
            selected_resources=selected_resources,
        )

    attempt_prompts = [
        prompt,
        prompt + "\n\n⚠️ Retry. Output ONLY valid JSON. No markdown fences. Rewrite every point in your own words — do NOT quote the RAG context.",
    ]

    payload = None
    for attempt_index, attempt_prompt in enumerate(attempt_prompts):
        try:
            response = client.chat.completions.create(
                model=RECOMMENDATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a warm, careful neurologist writing patient-facing "
                            "recommendations in second-person voice. You always return "
                            "valid JSON only, never explanations or markdown."
                        ),
                    },
                    {"role": "user", "content": attempt_prompt},
                ],
                temperature=0.40 if attempt_index == 0 else 0.30,
                top_p=0.85,
                max_tokens=RECOMMENDATION_MAX_TOKENS,
            )

            candidate = _parse_llm_json(response)

            required = {
                "summary", "meaning", "medical", "physical",
                "speech", "lifestyle", "monitoring", "follow_up",
            }
            if not required.issubset(candidate):
                raise ValueError("Recommendation JSON is missing required keys.")

            ok, reason, bad_sections = _validate_llm_payload(
                candidate, retrieved=retrieved, lang=lang,
            )
            if ok:
                payload = candidate
                break

            logger.warning(
                "LLM validation failed (attempt %d): %s | bad sections: %s",
                attempt_index + 1, reason, bad_sections,
            )

            # Try targeted retry for just the failed sections.
            patched = _targeted_retry(
                client=client,
                payload=candidate,
                bad_sections=bad_sections,
                intelligence=intelligence,
                retrieved=retrieved,
                lang=lang,
            )
            ok2, reason2, bad2 = _validate_llm_payload(
                patched, retrieved=retrieved, lang=lang,
            )
            if ok2:
                payload = patched
                logger.info(
                    "Targeted retry succeeded. All sections valid.",
                )
                break

            logger.warning(
                "Targeted retry still failing for: %s | %s",
                bad2, reason2,
            )
            # Keep the patched payload; per-section Python fallback happens next.
            payload = patched
            break

        except Exception as exc:
            logger.exception("OpenRouter attempt %d error: %s", attempt_index + 1, exc)
            if attempt_index == 0:
                continue

    if payload is None:
        logger.warning("All LLM attempts failed. Using full Python fallback.")
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved=retrieved,
            selected_resources=selected_resources,
        )

    # Python-owned sections.
    python_summary, python_meaning = _summary_and_meaning(intelligence, lang=lang)
    payload["summary"] = python_summary
    payload["meaning"] = python_meaning
    payload["medical"] = _watch_closely_points(intelligence, lang=lang)

    # Force Python neutral text for sections where RAG returned no cards.
    payload = _enforce_empty_rag_sections(
        payload,
        intelligence=intelligence,
        retrieved=retrieved,
        lang=lang,
    )

    # Final per-section safety net: any section that still fails local validation
    # (banned words, verbatim, too short) gets replaced with Python fallback.
    card_pool = _build_card_text_pool(retrieved)
    policy = _recommendation_policy(intelligence)
    for key in LLM_CONTENT_SECTIONS:
        points = payload.get(key)
        if isinstance(points, str):
            points = [points]
        if not isinstance(points, list) or not points:
            points = []

        section_ok = True
        for point in points:
            text = str(point or "").strip()
            if len(text.split()) < 6:
                section_ok = False
                break
            if _BANNED_WORDS.search(text):
                section_ok = False
                break
            if _is_verbatim_copy(text, card_pool.get(key, "")):
                section_ok = False
                break

        if not section_ok or not points:
            fallback = _neutral_section_points(key, policy, lang=lang)
            if not fallback:
                rag_section = _section_map().get(key)
                fallback = _points_from_retrieved(rag_section, retrieved, max_points=3) if rag_section else []
            if not fallback:
                fallback = (
                    ["No additional targeted recommendation is needed for this area from the current screening."]
                    if lang != "ur"
                    else ["موجودہ اسکریننگ کے مطابق اس حصے کے لیے مزید مخصوص سفارش درکار نہیں۔"]
                )
            payload[key] = fallback
            logger.info(
                "Section '%s' replaced with Python fallback after final validation.",
                key,
            )

    try:
        result = _format_action_report(payload, lang=lang)
    except Exception as exc:
        logger.exception("Report formatting failed: %s", exc)
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved=retrieved,
            selected_resources=selected_resources,
        )

    if lang == "ur":
        result = _clean_urdu_output_text(result)

    result = inject_resources_into_report(result, selected_resources, lang=lang)
    return result