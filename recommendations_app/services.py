"""
NeuroSense AI - Recommendation Services
Hybrid architecture: NeuroSense analyzers + curated RAG + Gemini language layer.

Important methodology
---------------------
1. Existing NeuroSense analyzers determine the structured patient profile.
2. The RAG service selects eligible recommendation cards from NeuroSense's
   curated JSON knowledge base.
3. The curated resource selector optionally attaches verified video resources.
4. Gemini is used only to organize, personalize and translate the already
   selected content. It is NOT allowed to invent medical recommendations,
   exercises, timelines or URLs.
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
    format_resources_for_prompt,
    inject_resources_into_report,
    select_report_resources,
)

logger = logging.getLogger(__name__)


# ================================================================
# GEMINI CLIENT
# ================================================================
def get_gemini_client():
    """Initialize and return the google-genai client."""
    try:
        api_key = getattr(settings, "GEMINI_API_KEY", None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in settings")

        from google import genai
        return genai.Client(api_key=api_key)

    except Exception as exc:
        logger.error("Failed to initialize Gemini client: %s", exc)
        return None


# ================================================================
# FORMATTERS
# ================================================================
def _format_insights_for_prompt(insights, lang):
    if not insights:
        return "No specific insights detected."

    lines = []
    for insight in insights:
        if not isinstance(insight, dict):
            continue
        text = insight.get(lang, insight.get("en", ""))
        source = str(insight.get("source", "system")).upper()
        if text:
            lines.append(f"- [{source}] {text}")

    return "\n".join(lines) if lines else "No specific insights detected."


def _format_longitudinal_for_prompt(longitudinal, risk, lang):
    """Build the previous-screening comparison block for the prompt."""
    if not longitudinal or longitudinal.get("previous_pd_probability") is None:
        return (
            "یہ پہلی اسکریننگ ہے؛ پچھلا موازنہ دستیاب نہیں۔"
            if lang == "ur"
            else "This is the first screening. No previous comparison is available."
        )

    overall_trend = str(longitudinal.get("overall_trend", "")).lower()
    delta_dir = str(longitudinal.get("delta_direction", "")).lower()

    gait_trend = str(longitudinal.get("gait_trend", "") or "").lower()
    speech_trend = str(longitudinal.get("speech_trend", "") or "").lower()
    questionnaire_trend = str(longitudinal.get("questionnaire_trend", "") or "").lower()
    motor_trend = str(longitudinal.get("motor_score_trend", "") or "").lower()

    def direction_word(value: str, lang_code: str) -> str:
        value = (value or "").lower()
        if "significantly_worsening" in value:
            return "زیادہ نمایاں" if lang_code == "ur" else "much more noticeable"
        if "worsening" in value:
            return "نمایاں" if lang_code == "ur" else "a bit more noticeable"
        if "significantly_improving" in value:
            return "کافی بہتر" if lang_code == "ur" else "clearly better"
        if "improving" in value:
            return "بہتر" if lang_code == "ur" else "a bit better"
        if "stable" in value:
            return "قریب قریب ایک جیسا" if lang_code == "ur" else "about the same"
        return ""

    if lang == "ur":
        lines = ["پچھلی اسکریننگ سے موازنہ:"]
        ow = direction_word(overall_trend or delta_dir, "ur")
        if ow:
            lines.append(f"- مجموعی طور پر: {ow}۔")
        gw = direction_word(gait_trend, "ur")
        if gw:
            lines.append(f"- چال: {gw}۔")
        sw = direction_word(speech_trend, "ur")
        if sw:
            lines.append(f"- آواز: {sw}۔")
        qw = direction_word(questionnaire_trend, "ur")
        if qw:
            lines.append(f"- سوالنامہ: {qw}۔")
        mw = direction_word(motor_trend, "ur")
        if mw:
            lines.append(f"- حرکت کی علامات: {mw}۔")
        return "\n".join(lines)

    lines = ["Comparison with previous screening:"]
    ow = direction_word(overall_trend or delta_dir, "en")
    if ow:
        lines.append(f"- Overall: {ow}.")
    gw = direction_word(gait_trend, "en")
    if gw:
        lines.append(f"- Gait: {gw}.")
    sw = direction_word(speech_trend, "en")
    if sw:
        lines.append(f"- Speech: {sw}.")
    qw = direction_word(questionnaire_trend, "en")
    if qw:
        lines.append(f"- Questionnaire: {qw}.")
    mw = direction_word(motor_trend, "en")
    if mw:
        lines.append(f"- Motor symptoms: {mw}.")
    return "\n".join(lines)


def _module_trend(intelligence, module: str) -> str:
    """Return simplified per-module trend: worsening | improving | stable | '' (unknown)."""
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


def _build_locked_rag_context(
    retrieved: Dict[str, List[Dict[str, Any]]]
) -> str:
    """Convert selected RAG cards to a locked content block for Gemini."""
    return format_retrieved_context(retrieved, include_sources=False)


# ================================================================
# PATIENT-FACING TEXT HELPERS
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

    if lang == "ur":
        area = {
            "speech": "آواز یا بولنے میں تبدیلی زیادہ توجہ چاہتی ہے۔",
            "gait": "چلنے یا توازن میں تبدیلی زیادہ توجہ چاہتی ہے۔",
            "questionnaire": "روزمرہ علامات پر زیادہ توجہ دیں۔",
        }.get(dominant, "اپنی روزمرہ علامات پر نظر رکھیں۔")
        change = (
            "کچھ تبدیلیاں پچھلی اسکریننگ سے زیادہ نمایاں ہیں۔" if "wors" in trend
            else "کچھ بہتری نظر آتی ہے۔" if "improv" in trend
            else "نتائج میں بڑی تبدیلی نظر نہیں آتی۔"
        )
    else:
        area = {
            "speech": "Voice or speech needs more attention.",
            "gait": "Walking or balance needs more attention.",
            "questionnaire": "Day-to-day symptoms need more attention.",
        }.get(dominant, "Keep an eye on day-to-day symptoms.")
        change = (
            "Some changes are more noticeable than before." if "wors" in trend
            else "Some findings look better than before." if "improv" in trend
            else "There is no major change compared with the previous screening."
        )

    return (
        f"Screening level: {risk_level}. Dominant area: {dominant}. "
        f"Gait={module_predictions.get('gait', 'Unknown')} ({fmt_prob('gait')}); "
        f"Speech={module_predictions.get('speech', 'Unknown')} ({fmt_prob('speech')}); "
        f"Questionnaire={module_predictions.get('questionnaire', 'Unknown')} ({fmt_prob('questionnaire')}). "
        f"Speech proximity={speech.get('threshold_proximity', 'unknown')}; "
        f"Speech consistency={speech.get('consistency', 'unknown')}; "
        f"Motor severity={questionnaire.get('motor_severity', 'unknown')}. "
        f"Main message: {area} Change: {change}"
    )


def _module_state(value):
    """Recommendation state for an individual module, independent of overall risk."""
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
    """Build the patient-facing content policy for this exact screening."""
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


def _policy_for_prompt(policy):
    return (
        f"physical={policy['physical']}; speech={policy['speech_section']}; "
        f"lifestyle={policy['lifestyle']}({policy.get('lifestyle_reason')}); "
        f"monitoring={policy['monitoring']}; watch_closely={policy['medical']}; "
        f"follow_up={policy['follow_up']}; gait_state={policy['gait']}; "
        f"speech_state={policy['speech']}; questionnaire_state={policy['questionnaire']}; "
        f"questionnaire_only_elevated={policy['questionnaire_only_elevated']}"
    )


def _urdu_risk_label(value):
    value = str(value or "").strip().lower()
    return {
        "low": "کم",
        "medium": "درمیانی",
        "high": "زیادہ",
    }.get(value, str(value or ""))


# ================================================================
# SUMMARY + MEANING — SINGLE-WRITER, RISK-AWARE, COMPARISON-AWARE
# ================================================================
def _describe_module_against_trend(
    module_name_en: str,
    module_name_ur: str,
    module_state: str,
    trend: str,
    has_previous: bool,
    lang: str,
) -> str:
    """Return a warm sentence fragment describing one module's current state + trend.

    Handles plural module names like "walking and balance" with correct verb agreement.
    """
    # Detect plural module names so verb agreement is correct.
    plural_en = " and " in module_name_en
    plural_ur = "اور" in module_name_ur

    if lang == "ur":
        # Urdu verb forms are the same for singular and plural in most of these
        # phrasings, so plural_ur is informational only.
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
        # low
        if has_previous and trend == "improving":
            return f"{module_name_ur} پہلے سے بہتر ہیں"
        if has_previous and trend == "stable":
            return f"{module_name_ur} تقریباً پہلے جیسے ہیں"
        return f"{module_name_ur} ٹھیک لگ رہے ہیں"

    # English — verb agreement based on plural detection
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
    # low
    if has_previous and trend == "improving":
        return f"your {module_name_en} {verb_looks} a bit better than last time"
    if has_previous and trend == "stable":
        return f"your {module_name_en} {verb_looks} about the same as last time"
    return f"your {module_name_en} {verb_looks} steady"


def _risk_aware_opener(risk_level: str, lang: str) -> str:
    """Return a warm opener sentence matching the risk level."""
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


def _summary_and_meaning(intelligence, lang="en"):
    """Build the patient-facing Summary and What Your Results Mean.

    Single writer. Risk-aware + comparison-aware. No prepending. No contradictions.
    """
    policy = _recommendation_policy(intelligence)
    risk = str((intelligence.get("risk_summary") or {}).get("risk_level") or "Low")

    gait = policy["gait"]
    speech = policy["speech"]
    questionnaire = policy["questionnaire"]

    has_prev = _has_previous_screening(intelligence)
    gait_trend = _module_trend(intelligence, "gait")
    speech_trend = _module_trend(intelligence, "speech")

    opener = _risk_aware_opener(risk, lang)

    # ============================================================
    # URDU
    # ============================================================
    if lang == "ur":
        gait_phrase = _describe_module_against_trend(
            "gait", "چال", gait, gait_trend, has_prev, "ur"
        )
        speech_phrase = _describe_module_against_trend(
            "speech", "آواز یا بولنے", speech, speech_trend, has_prev, "ur"
        )

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
        elif gait == "watch" or speech == "watch":
            if gait == "watch" and speech == "watch":
                body = f"{gait_phrase}، اور {speech_phrase}۔"
            elif gait == "watch":
                body = f"{gait_phrase}۔ {speech_phrase}۔"
            else:
                body = f"{speech_phrase}۔ {gait_phrase}۔"
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "اگلی اسکریننگ میں انہی حصوں کا موازنہ کریں۔",
            ]
        else:
            body = f"{gait_phrase}، اور {speech_phrase}۔"
            meaning = [
                f"آپ کی مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "صرف نئی یا مسلسل ہونے والی روزمرہ تبدیلیوں پر نظر رکھیں۔",
            ]

        summary = opener + body
    # ============================================================
    # ENGLISH
    # ============================================================
    else:
        gait_phrase = _describe_module_against_trend(
            "walking and balance", "", gait, gait_trend, has_prev, "en"
        )
        speech_phrase = _describe_module_against_trend(
            "voice and speech", "", speech, speech_trend, has_prev, "en"
        )

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
        elif gait == "watch" or speech == "watch":
            if gait == "watch" and speech == "watch":
                body = f"{gait_phrase}, and {speech_phrase}."
            elif gait == "watch":
                body = f"{gait_phrase}. {speech_phrase.capitalize()}."
            else:
                body = f"{speech_phrase}. {gait_phrase.capitalize()}."
            meaning = [
                f"Your combined screening level is {risk}.",
                "No specific exercise video is needed at this stage.",
                "Let's compare the same areas again at your next screening, under similar conditions.",
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
# NEUTRAL FALLBACK TEXT — WARM DOCTOR TONE
# ================================================================
def _neutral_section_points(section_key, policy, lang="en"):
    """Fallback text for low/watch sections, written in warm doctor tone."""
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
            if policy.get("questionnaire_only_elevated"):
                return ["آپ کے سوالنامے سے کوئی خاص خوراک، پانی یا نیند کی تبدیلی سامنے نہیں آئی۔ صرف ان علامات پر توجہ دیں جو آپ واقعی محسوس کر رہے ہیں۔"]
            if policy.get("gait") == "attention":
                return ["اس وقت کھانے یا غذا میں کسی خاص تبدیلی کی ضرورت نہیں۔ اپنی توجہ چلنے اور توازن پر رکھیں۔"]
            return ["اس اسکریننگ سے آپ کی روزمرہ طرزِ زندگی میں کسی اضافی تبدیلی کی ضرورت سامنے نہیں آئی۔ غیر ضروری خوراک یا غذائی اضافے شامل نہ کریں۔"]
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
        if policy.get("questionnaire_only_elevated"):
            return ["Your questionnaire did not point to a specific food, fluid, or sleep change. Focus on the symptoms you actually notice."]
        if policy.get("gait") == "attention":
            return ["There isn't a specific food or diet change to focus on right now. Let's keep our attention on your walking and balance."]
        return ["This screening does not add a specific lifestyle change. There's no need to make diet or supplement changes just because this section is here."]
    return []


def _normalized_questionnaire_flags(intelligence):
    questionnaire = intelligence.get("questionnaire") or {}
    analysis = questionnaire.get("analysis") or {}
    flags = []
    for raw in analysis.get("safety_flags") or []:
        text = str(raw or "").strip().lower()
        if text:
            flags.append(text)
    return flags


# ================================================================
# WATCH CLOSELY (SECTION 5) — WARM DOCTOR TONE
# ================================================================
def _watch_closely_points(intelligence, lang="en"):
    """Deterministic, module-aware safety wording in warm doctor tone."""
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


def _professional_referral_point(text):
    return bool(re.search(
        r"\b(doctor|neurologist|clinician|specialist|physiotherapist|physical therapist|speech therapist|speech-language|professional assessment|professional evaluation)\b",
        str(text or ""),
        flags=re.IGNORECASE,
    ))


def _self_care_point_from_retrieved(section, retrieved):
    for card in retrieved.get(section) or []:
        if str(card.get("action_type") or "").lower() != "self_care":
            continue
        text = _clean_patient_text(
            card.get("patient_action") or card.get("recommendation") or card.get("personalization_hint") or "",
            limit=35,
        )
        if text:
            return text
    return ""


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
            card.get("patient_action") or card.get("recommendation") or card.get("personalization_hint") or "",
            limit=35,
        )
        if text and text not in points:
            points.append(text)
        if len(points) >= max_points:
            break
    return points


def _limit_repeated_general_referrals(payload):
    """Keep routine doctor/neurologist referral near the end of the care plan."""
    payload = dict(payload or {})
    routine_terms = re.compile(
        r"\b(doctor|neurologist|clinician|healthcare professional|medical review|medical assessment|check-up)\b",
        re.IGNORECASE,
    )
    urgent_terms = re.compile(
        r"\b(urgent|sudden|prompt medical advice|choking|unsafe without support)\b",
        re.IGNORECASE,
    )

    for key in ("physical", "speech", "lifestyle", "monitoring"):
        points = payload.get(key)
        if isinstance(points, str):
            points = [points]
        if not isinstance(points, list):
            continue
        kept = []
        for point in points:
            text = str(point or "").strip()
            if not text:
                continue
            if routine_terms.search(text) and not urgent_terms.search(text):
                continue
            kept.append(text)
        payload[key] = kept

    points = payload.get("follow_up")
    if isinstance(points, str):
        points = [points]
    if isinstance(points, list):
        kept = []
        routine_kept = False
        for point in points:
            text = str(point or "").strip()
            if not text:
                continue
            if routine_terms.search(text) and not urgent_terms.search(text):
                if routine_kept:
                    continue
                routine_kept = True
            kept.append(text)
        payload["follow_up"] = kept

    return payload


def _specific_lifestyle_points(policy, retrieved, lang="en"):
    """Return short, concrete lifestyle points when this section is active."""
    if policy.get("lifestyle") != "active":
        return []

    reason = policy.get("lifestyle_reason")

    if lang == "ur":
        return []

    if reason == "speech_support":
        return [
            "Sip water through the day, about 6 to 8 glasses, unless your doctor has told you to limit fluids.",
            "For meals, try beans, eggs or fish, brown rice or whole grains, and fruit or vegetables such as apples, tomatoes, or spinach.",
        ]

    points = _points_from_retrieved(
        "lifestyle_wellness",
        retrieved,
        max_points=3,
    )
    return points


def _apply_section_policy(payload, intelligence, retrieved, lang="en"):
    """Enforce the final NeuroSense care-plan behavior after the language layer."""
    payload = dict(payload or {})
    policy = _recommendation_policy(intelligence)
    summary, meaning = _summary_and_meaning(intelligence, lang=lang)
    payload["summary"] = summary
    payload["meaning"] = meaning

    section_map = {
        "physical": "physical_therapy_exercise",
        "speech": "speech_voice_therapy",
        "lifestyle": "lifestyle_wellness",
        "monitoring": "symptom_monitoring",
        "follow_up": "follow_up_schedule",
    }

    for key, rag_section in section_map.items():
        # Specific lifestyle advice takes priority over neutral fallback.
        if key == "lifestyle":
            specific_lifestyle = _specific_lifestyle_points(
                policy,
                retrieved,
                lang=lang,
            )
            if specific_lifestyle:
                payload[key] = specific_lifestyle[:3]
                continue

        neutral = _neutral_section_points(key, policy, lang=lang)
        if neutral:
            payload[key] = neutral
            continue

        points = _normalize_points(payload.get(key), min_points=1, max_points=3)
        if not points:
            points = _points_from_retrieved(rag_section, retrieved, max_points=3)

        if key in {"physical", "speech"} and points:
            if all(_professional_referral_point(point) for point in points):
                self_care = _self_care_point_from_retrieved(rag_section, retrieved)
                if self_care:
                    points.insert(0, self_care)

            points = sorted(points, key=lambda point: 1 if _professional_referral_point(point) else 0)

        if not points:
            points = [
                "No additional targeted recommendation is needed for this area from the current screening."
                if lang != "ur"
                else "موجودہ اسکریننگ کے مطابق اس حصے کے لیے مزید مخصوص سفارش درکار نہیں۔"
            ]
        payload[key] = points[:3]

    payload["medical"] = _watch_closely_points(intelligence, lang=lang)
    payload = _limit_repeated_general_referrals(payload)

    for key in ("physical", "speech", "lifestyle", "monitoring", "medical", "follow_up"):
        if payload.get(key):
            continue
        if key == "medical":
            payload[key] = _watch_closely_points(intelligence, lang=lang)
        else:
            neutral = _neutral_section_points(key, policy, lang=lang)
            if neutral:
                payload[key] = neutral
            else:
                rag_section = section_map.get(key)
                payload[key] = _points_from_retrieved(rag_section, retrieved, max_points=2) if rag_section else []
            if not payload[key]:
                payload[key] = [
                    "No additional targeted recommendation is needed for this area from the current screening."
                    if lang != "ur"
                    else "موجودہ اسکریننگ کے مطابق اس حصے کے لیے مزید مخصوص سفارش درکار نہیں۔"
                ]

    return payload


# ================================================================
# PROMPT BUILDER — WARM DOCTOR TONE + LONGITUDINAL COMPARISON
# ================================================================
def build_doctor_prompt(
    intelligence,
    retrieved=None,
    selected_resources=None,
    lang="en",
):
    """Build the strict wording prompt. RAG decides the content; Gemini only phrases it."""
    retrieved = retrieved or {}
    rag_context = format_retrieved_context(retrieved, include_sources=False)
    profile_hint = _patient_profile_hint(intelligence, lang=lang)
    section_policy = _policy_for_prompt(_recommendation_policy(intelligence))

    risk_summary = intelligence.get("risk_summary") or {}
    longitudinal_text = _format_longitudinal_for_prompt(
        intelligence.get("longitudinal") or {},
        risk_summary,
        lang=lang,
    )

    if lang == "ur":
        return f"""آپ NeuroSense کے لیے warm، مختصر اور patient-friendly care guidance لکھ رہے ہیں۔
لکھتے وقت ایسے محسوس کریں جیسے آپ ایک تجربہ کار نیورولوجسٹ ہیں جو مریض کے ساتھ بیٹھے بات کر رہے ہیں۔

صرف CURATED RAG CONTEXT میں موجود advice استعمال کریں۔ کوئی نئی medicine، treatment، exercise duration، food example، fluid quantity یا diagnosis خود سے نہ بنائیں۔

اہم اصول:
1. Summary اور Meaning خود بخود بھر جائیں گے؛ آپ کے جواب میں summary اور meaning مختصر رکھیں۔
2. ہر section دکھائیں، مگر garbage filler نہ دیں۔ Active section میں 2-3 warm actions ہوں؛ low/watch section میں ایک مختصر واضح جملہ کافی ہے۔
3. Physical اور Speech sections میں user کے خود کرنے والے actions پہلے آئیں؛ professional referral اگر relevant ہو تو آخر میں ہو۔
4. Lifestyle صرف تب specific advice دے جب RAG میں واقعی reason ہو۔ Quantity یا food examples RAG میں ہوں تو exactly preserve کریں؛ generic "healthy food" یا "drink water" نہ لکھیں۔
5. Monitoring میں واضح لکھیں کہ کن changes کو note کرنا ہے، نرم اور دوستانہ لہجے میں۔
6. "Changes to Watch Closely" کے لیے unrelated emergency symptoms نہ بنائیں۔ صرف affected gait/speech area یا questionnaire میں واقعی report ہوئی supporting symptom استعمال کریں۔
7. یہ نہ کہیں کہ user کو doctor کی ضرورت نہیں۔ Routine doctor advice مختلف sections میں repeat نہ کریں۔
8. Follow-up میں next screening اور previous report comparison کی practical guidance دیں۔
9. confidence, threshold, score, model, fusion, discordance, certainty, RAG, algorithm, domain, longitudinal جیسے technical الفاظ patient-facing output میں نہ آئیں۔
10. percentages/decimals نہ لکھیں؛ probability page پر الگ دکھائی جاتی ہے۔
11. YouTube link نہ بنائیں؛ verified video code بعد میں attach کرے گا۔
12. Questionnaire-only elevation gait/speech therapy یا video trigger نہیں کرتا۔
13. Patient-facing جواب خالص اور آسان اردو میں لکھیں۔ Low/Medium/High، gait، speech، medical advice، diet، exercise، warning جیسے عام English الفاظ output میں نہ لکھیں۔
14. صرف verified video کا اصل title/provider بعد میں code کے ذریعے English میں رہ سکتا ہے۔
15. ہر point 10 سے 22 الفاظ کے درمیان ہو۔ ہر point میں 1-2 چھوٹے جملے ہوں۔
16. دوستانہ اور پرسکون لہجہ رکھیں، لیکن ضرورت سے زیادہ غیر رسمی یا جوشیلا نہ ہوں۔
17. "آپ" اور "ہم" کا استعمال کریں جیسے ڈاکٹر مریض سے بات کرتا ہے۔
18. Physical Therapy اور Speech کے سیکشنز میں جب کسی پیشہ ور (physiotherapist، speech therapist) کا ذکر کریں تو ایسے لکھیں کہ وہ مریض کے ساتھ کیا کرے گا۔
19. "Note whether..." کی جگہ "Notice if..." لکھیں۔ "Monitor..." کی جگہ "Keep an eye on..." لکھیں۔
20. اہم: کسی بھی سیکشن میں "گزشتہ اسکریننگ کے مقابلے میں" یا "پہلے جیسا ہی ہے" جیسی باتیں نہ لکھیں۔ ہر سیکشن صرف موجودہ اسکریننگ کی حالت بیان کرے۔

SECTION POLICY:
{section_policy}

INTERNAL PROFILE — patient کو copy نہ کریں:
{profile_hint}

PREVIOUS SCREENING COMPARISON:
{longitudinal_text}

CURATED RAG CONTEXT — یہی allowed recommendation source ہے:
{rag_context}

صرف valid JSON return کریں:
{{
  "summary": "1-2 warm short sentences (will be auto-replaced by system)",
  "meaning": ["point 1", "point 2"],
  "physical": ["warm action 1", "warm action 2", "optional warm action 3"],
  "speech": ["warm action 1", "warm action 2", "optional warm action 3"],
  "lifestyle": ["warm action 1", "warm action 2", "optional warm action 3"],
  "monitoring": ["warm action 1", "warm action 2", "optional warm action 3"],
  "medical": ["module-specific warm watch point"],
  "follow_up": ["warm action 1", "warm action 2", "optional warm action 3"]
}}
"""

    return f"""You write warm, concise patient guidance for NeuroSense.
Write as if you are a caring neurologist sitting with the patient and talking through their results.

Use ONLY the CURATED RAG CONTEXT below. Do not invent medicines, treatments, exercise durations, food examples, fluid quantities, diagnoses, or URLs.

TONE REQUIREMENTS:
- Speak directly to the patient using "you" and "we".
- Be warm, personal, and reassuring, but still clinical.
- Never sound like a report. Never sound detached. Never sound alarmist.
- Every point should feel like something a caring doctor would actually say.

RISK-AWARE TONE:
- If screening level is LOW: reassuring tone. "Looks steady", "nothing specific needed right now".
- If screening level is MEDIUM: attentive tone. "Worth keeping an eye on", "let's watch this together".
- If screening level is HIGH: serious but warm tone. "Needs some attention soon".
- Never panic the patient. Never understate either. Match the level.

STRICT RULES:
1. Summary and Meaning will be auto-filled by the system. In your JSON response, keep them short (they will be replaced) but still warm.
2. Keep all six sections. Never add filler just to fill space.
3. Active Physical/Speech sections: give 2-3 warm actions. Put something the user can do first; professional assessment, if supported, goes last.
4. Low/watch Physical/Speech sections: one warm neutral line is enough. Do not attach a video unless that module is >=50%.
5. Lifestyle: be symptom-driven. If RAG provides a fluid quantity or food examples, preserve them. Never write generic advice.
6. Symptom Monitoring: clearly name WHAT to watch or record, in warm, plain words.
7. Changes to Watch Closely: mention only the affected gait/speech area and explicitly reported supporting symptoms. Do not add unrelated emergencies.
8. Never say that the user does not need medical care. Avoid routine doctor advice across multiple sections.
9. Follow-up: give warm, practical next-screening guidance.
10. Do not repeat the same recommendation across sections.
11. NEVER expose these technical words: confidence, threshold, score, model, fusion, discordance, certainty, RAG, algorithm, domain, longitudinal.
12. Do not output percentages or decimals.
13. Do not output YouTube/resource URLs; verified resources are inserted by code.
14. Questionnaire-only elevation is supporting evidence and must not trigger gait/speech therapy or videos.
15. Use everyday English. Prefer short words and short sentences.
16. Avoid formal wording such as "maintain", "restrict fluids", "simultaneous changes", "clinical evaluation", "communication needs", or "persistent".
17. If you mention food, name actual food examples from the RAG context. If you mention water, keep the quantity exactly as given.
18. Each point should be 10 to 22 words, made of 1-2 short sentences.
19. Use "you" and "we". Speak like a doctor, not a system.
20. In Physical Therapy and Speech sections, phrase professional help as something they can DO WITH the patient.
21. Prefer "Notice if..." over "Note whether...". Prefer "Keep an eye on..." over "Monitor...".
22. IMPORTANT: Do NOT write anything like "compared to your last screening" or "about the same as last time" in ANY section. The system handles all comparisons. Each section should describe ONLY the current screening findings.

TONE EXAMPLES:

BAD (clinical, detached):
"There are mild gait changes. Exercise. Consult a doctor."

GOOD (warm, doctor-like):
"There are some mild changes in your walking that are worth keeping an eye on. No specific exercise video is needed yet."

BAD (clinical, high-risk):
"Multiple modules affected. Consult soon."

GOOD (warm, high-risk):
"This screening shows some changes that are worth paying attention to soon. Let's focus on what we can do next."

SECTION POLICY:
{section_policy}

INTERNAL PROFILE — DO NOT COPY THIS TECHNICAL WORDING:
{profile_hint}

PREVIOUS SCREENING COMPARISON (for your awareness only — do NOT include this in any section):
{longitudinal_text}

CURATED RAG CONTEXT — THIS IS THE ONLY ALLOWED RECOMMENDATION SOURCE:
{rag_context}

Return ONLY valid JSON:
{{
  "summary": "1-2 warm short sentences (will be auto-replaced by system)",
  "meaning": ["warm short point 1", "warm short point 2"],
  "physical": ["warm action 1", "warm action 2", "optional warm action 3"],
  "speech": ["warm action 1", "warm action 2", "optional warm action 3"],
  "lifestyle": ["warm action 1", "warm action 2", "optional warm action 3"],
  "monitoring": ["warm action 1", "warm action 2", "optional warm action 3"],
  "medical": ["module-specific warm watch point"],
  "follow_up": ["warm action 1", "warm action 2", "optional warm action 3"]
}}
"""


def _trim_words(text, limit):
    words = str(text or "").strip().split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "…"


def _clean_patient_text(text, limit=22):
    """Final safety pass for patient-facing wording."""
    text = str(text or "").strip()

    replacements = [
        (r"\bspeech-language pathologist\b", "speech therapist"),
        (r"\bclinical review\b", "doctor visit"),
        (r"\bclinical assessment\b", "doctor check-up"),
        (r"\bneurological evaluation\b", "check-up with a neurologist"),
        (r"\bmultidisciplinary care\b", "help from the right health professionals"),
        (r"\bmultidisciplinary\b", "different health"),
        (r"\bscreening domains?\b", "screening areas"),
        (r"\bmodel'?s?\b", ""),
        (r"\bthreshold\b", ""),
        (r"\bconfidence\b", ""),
        (r"\bdiscordance\b", ""),
        (r"\bcertainty\b", ""),
        (r"\bfusion\b", ""),
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
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r"\b0\.\d+\b", "", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*%", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-")

    return _trim_words(text, limit)


def _normalize_points(value, min_points=1, max_points=3):
    """Normalize Gemini JSON fields into up to three useful patient-facing points."""
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
        item = _clean_patient_text(item, limit=35)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= max_points:
            break

    return cleaned


def _clean_urdu_output_text(text):
    """Remove common English leftovers from patient-facing Urdu text."""
    text = str(text or "")

    replacements = [
        (r"\bMedium\b", "درمیانی"),
        (r"\bLow\b", "کم"),
        (r"\bHigh\b", "زیادہ"),
        (r"\bmedical advice\b", "طبی مشورہ"),
        (r"\bgait exercise\b", "چال کی ورزش"),
        (r"\bspeech exercise\b", "آواز یا بولنے کی مشق"),
        (r"\bspeech video\b", "آواز یا بولنے کی ویڈیو"),
        (r"\bnear-fall\b", "گرتے گرتے بچنا"),
        (r"\bbalance\b", "توازن"),
        (r"\bchoking\b", "دم گھٹنے یا نگلنے میں مشکل"),
        (r"\bdiet\b", "خوراک"),
        (r"\bsupplement\b", "غذائی اضافہ"),
        (r"\bmovement\b", "حرکت"),
        (r"\blifestyle\b", "روزمرہ طرزِ زندگی"),
        (r"\bgait\b", "چال"),
        (r"\bspeech\b", "آواز یا بولنے"),
        (r"\bwarning\b", "تنبیہی علامت"),
        (r"\bexercise\b", "ورزش"),
        (r"\bvideo\b", "ویڈیو"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def _format_action_report(payload, lang="en"):
    """Format JSON payload into the fixed six-section patient care plan."""
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
    summary = _clean_patient_text(payload.get("summary", ""), limit=65)
    lines.extend([f"## {headings[0][0]}", summary, ""])

    for heading, key in headings[1:]:
        points = _normalize_points(payload.get(key), min_points=1, max_points=3)
        if not points:
            raise ValueError(f"Recommendation section '{key}' returned no useful content.")

        lines.append(f"## {heading}")
        for point in points:
            lines.append(f"- {point}")
        lines.append("")

    return "\n".join(lines).strip()


# ================================================================
# RAG FALLBACK REPORT
# ================================================================
def _english_or_urdu_heading(section, lang):
    if lang != "ur":
        return SECTION_LABELS.get(section, section)

    urdu = {
        "physical_therapy_exercise": "1. فزیکل تھراپی اور ورزش",
        "speech_voice_therapy": "2. اسپیچ اور آواز کی تھراپی",
        "lifestyle_wellness": "3. طرز زندگی اور صحت",
        "symptom_monitoring": "4. علامات کی نگرانی",
        "medical_consultation": "5. قریب سے دیکھنے والی تبدیلیاں",
        "follow_up_schedule": "6. فالو اپ شیڈول",
    }
    return urdu.get(section, SECTION_LABELS.get(section, section))


def get_fallback_recommendations(
    intelligence,
    lang="en",
    retrieved=None,
    selected_resources=None,
):
    """Build fallback text from THIS screening's retrieved cards."""
    retrieved = retrieved or retrieve_report_recommendations(intelligence)

    summary, meaning = _summary_and_meaning(intelligence, lang=lang)
    policy = _recommendation_policy(intelligence)

    section_key_map = {
        "medical_consultation": "medical",
        "physical_therapy_exercise": "physical",
        "speech_voice_therapy": "speech",
        "lifestyle_wellness": "lifestyle",
        "symptom_monitoring": "monitoring",
        "follow_up_schedule": "follow_up",
    }
    payload = {"summary": summary, "meaning": meaning}

    for section, output_key in section_key_map.items():
        if output_key == "lifestyle":
            specific_lifestyle = _specific_lifestyle_points(policy, retrieved, lang=lang)
            if specific_lifestyle:
                payload[output_key] = specific_lifestyle[:3]
                continue

        neutral = _neutral_section_points(output_key, policy, lang=lang)
        if neutral:
            payload[output_key] = neutral
            continue

        points = _points_from_retrieved(section, retrieved, max_points=3)
        if not points:
            points = [
                "No additional targeted recommendation is needed for this area from the current screening."
                if lang != "ur"
                else "موجودہ اسکریننگ کے مطابق اس حصے کے لیے مزید مخصوص سفارش درکار نہیں۔"
            ]
        payload[output_key] = points

    payload = _apply_section_policy(payload, intelligence, retrieved, lang=lang)
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
    """Generate the existing NeuroSense recommendation report using:
    analyzers -> RAG -> optional verified resource -> Gemini wording layer.
    """
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
        counts = {
            section: len(retrieved.get(section) or [])
            for section in REPORT_SECTIONS
        }
        logger.info("RAG recommendations retrieved: %s", counts)
        logger.info(
            "RAG card IDs | screening=%s | %s",
            screening_id,
            {section: [card.get("id") for card in (retrieved.get(section) or [])] for section in REPORT_SECTIONS},
        )
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

    client = get_gemini_client()
    if not client:
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved=retrieved,
            selected_resources=selected_resources,
        )

    try:
        from google.genai import types

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.30,
                top_p=0.75,
                max_output_tokens=2000,
                response_mime_type="application/json",
            ),
        )

        raw_json = (response.text or "").strip()
        payload = json.loads(raw_json)

        required = {
            "summary", "meaning", "medical", "physical",
            "speech", "lifestyle", "monitoring", "follow_up"
        }
        if not required.issubset(payload):
            raise ValueError("Compact recommendation JSON is incomplete.")

        payload = _apply_section_policy(
            payload,
            intelligence=intelligence,
            retrieved=retrieved,
            lang=lang,
        )
        result = _format_action_report(payload, lang=lang)
        if lang == "ur":
            result = _clean_urdu_output_text(result)

        result = inject_resources_into_report(
            result,
            selected_resources,
            lang=lang,
        )
        return result

    except Exception as exc:
        logger.exception("Gemini language-layer error: %s", exc)
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved=retrieved,
            selected_resources=selected_resources,
        )