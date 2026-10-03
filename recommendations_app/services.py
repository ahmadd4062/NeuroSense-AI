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
    """
    Internal comparison summary for the language layer.

    Exact model values stay inside the system. The patient-facing output only
    describes the direction of change in simple words.
    """
    if not longitudinal or longitudinal.get("previous_pd_probability") is None:
        return (
            "پہلی اسکریننگ ہے؛ پچھلا موازنہ دستیاب نہیں۔"
            if lang == "ur"
            else "First screening; no previous comparison is available."
        )

    delta_dir = str(longitudinal.get("delta_direction", "")).lower()
    overall_trend = str(longitudinal.get("overall_trend", "")).lower()

    if "wors" in delta_dir or "wors" in overall_trend:
        return (
            "پچھلی اسکریننگ کے مقابلے میں کچھ تبدیلیاں زیادہ نمایاں ہیں۔"
            if lang == "ur"
            else "Some findings are more noticeable than in the previous screening."
        )

    if "improv" in delta_dir or "improv" in overall_trend:
        return (
            "پچھلی اسکریننگ کے مقابلے میں کچھ بہتری نظر آتی ہے۔"
            if lang == "ur"
            else "Some findings look better than in the previous screening."
        )

    return (
        "نتائج پچھلی اسکریننگ کے قریب ہیں۔"
        if lang == "ur"
        else "The results are broadly similar to the previous screening."
    )



def _build_locked_rag_context(
    retrieved: Dict[str, List[Dict[str, Any]]]
) -> str:
    """
    Convert selected RAG cards to a locked content block for Gemini.
    Sources are kept internally in retrieved metadata and are not required
    in the patient-facing report.
    """
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
    """Build the patient-facing content policy for this exact screening.

    Overall risk never hides an individual module. Gait and speech drive their
    own recommendations. Questionnaire findings are supporting evidence and can
    activate lifestyle advice only when an actual symptom flag is available.
    """
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

    # Speech >=50% can legitimately activate voice-care hydration guidance from
    # the curated knowledge base even when the questionnaire has no lifestyle flag.
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
        # The old medical-consultation slot is now a module-aware safety slot.
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


def _summary_and_meaning(intelligence, lang="en"):
    policy = _recommendation_policy(intelligence)
    risk = str((intelligence.get("risk_summary") or {}).get("risk_level") or "Low")

    gait = policy["gait"]
    speech = policy["speech"]
    questionnaire = policy["questionnaire"]

    if lang == "ur":
        if gait == "attention" and speech == "attention":
            summary = "چال اور آواز دونوں میں نمایاں تبدیلیاں نظر آئیں۔ ان دونوں حصوں پر اگلے اقدامات مرکوز ہونے چاہئیں۔"
            meaning = [
                f"مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔",
                "چلنے، توازن، آواز اور بولنے کی تبدیلیوں کو الگ الگ مانیٹر کریں۔",
            ]
        elif gait == "attention":
            summary = "چال یا توازن کے نتائج زیادہ توجہ چاہتے ہیں۔ آواز کے نتائج میں اتنی نمایاں تبدیلی نہیں ملی۔"
            meaning = [f"مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔", "چلنے، مڑنے اور توازن کی تبدیلیوں پر خاص نظر رکھیں۔"]
        elif speech == "attention":
            summary = "آواز یا بولنے کے نتائج زیادہ توجہ چاہتے ہیں۔ چال کے نتائج میں اتنی نمایاں تبدیلی نہیں ملی۔"
            meaning = [f"مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔", "آواز کی طاقت، وضاحت، وقفوں اور سمجھ آنے میں تبدیلی نوٹ کریں۔"]
        elif policy["questionnaire_only_elevated"]:
            summary = "چال اور آواز میں نمایاں تبدیلی نہیں ملی، لیکن سوالنامے میں زیادہ علامات رپورٹ ہوئیں۔ صرف وہ علامات مانیٹر کریں جو واقعی روزمرہ میں محسوس ہو رہی ہیں۔"
            meaning = [f"مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔", "سوالنامہ معاون معلومات ہے؛ اس اسکریننگ میں چال اور آواز نے وہی تبدیلیاں ظاہر نہیں کیں۔"]
        elif gait == "watch" or speech == "watch":
            areas = "چال" if gait == "watch" and speech != "watch" else "آواز" if speech == "watch" and gait != "watch" else "چال اور آواز"
            summary = f"مجموعی نتیجہ {_urdu_risk_label(risk)} ہے۔ {areas} میں ہلکی تبدیلیاں ہیں جنہیں ابھی مانیٹر کرنا بہتر ہے۔"
            meaning = ["اس وقت مخصوص تھراپی یا ویڈیو کی ضرورت نہیں دکھائی دیتی۔", "اگلی اسکریننگ میں انہی حصوں کا موازنہ کریں۔"]
        else:
            summary = "اس وقت چال یا آواز میں نمایاں تبدیلی نہیں ملی۔ ان حصوں کے لیے مخصوص سفارشات کی ضرورت نہیں ہے۔"
            meaning = [f"مجموعی اسکریننگ سطح {_urdu_risk_label(risk)} ہے۔", "صرف نئی یا مسلسل روزمرہ تبدیلیوں پر نظر رکھیں۔"]
    else:
        if gait == "attention" and speech == "attention":
            summary = "Your screening shows noticeable changes in both walking/balance and speech. These two areas should be the main focus of your next steps."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Track walking, balance, voice and speech changes separately instead of relying only on the overall result.",
            ]
        elif gait == "attention":
            summary = "Your walking or balance result needs more attention. Your speech result does not show the same level of change."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Focus on changes in walking, turning and balance rather than treating every screening area the same.",
            ]
        elif speech == "attention":
            summary = "Your voice or speech result needs more attention. Your gait result does not show the same level of change."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Focus on voice volume, clarity, pauses and communication changes rather than treating every screening area the same.",
            ]
        elif policy["questionnaire_only_elevated"]:
            summary = "Your gait and speech results do not show notable changes, but your questionnaire reported more symptoms. Focus on the symptoms you actually notice in daily life."
            meaning = [
                f"Your combined screening level is {risk}.",
                "The questionnaire adds supporting information, while gait and speech did not show matching changes in this screening.",
            ]
        elif gait == "watch" or speech == "watch":
            if gait == "watch" and speech == "watch":
                area = "gait and speech"
            elif gait == "watch":
                area = "gait"
            else:
                area = "speech"
            summary = f"Your combined screening level is {risk}. Mild changes in {area} are worth watching, but they do not need a targeted therapy plan yet."
            meaning = [
                "No module-specific video is needed at this stage.",
                "Compare the same area again in a future screening under similar conditions.",
            ]
        else:
            summary = "Your current screening does not show notable gait or speech changes. No targeted gait or speech recommendation is needed right now."
            meaning = [
                f"Your combined screening level is {risk}.",
                "Keep normal awareness of any new or persistent day-to-day changes.",
            ]

    if questionnaire == "attention" and not policy["questionnaire_only_elevated"]:
        if lang == "ur":
            meaning[-1] += " سوالنامے میں رپورٹ شدہ علامات کو بھی بطور معاون معلومات مانیٹر کریں۔"
        else:
            meaning[-1] += " Also track the symptoms reported in the questionnaire as supporting information."

    return summary, meaning


def _neutral_section_points(section_key, policy, lang="en"):
    mode = {
        "physical": policy["physical"],
        "speech": policy["speech_section"],
        "lifestyle": policy["lifestyle"],
    }.get(section_key, "active")

    if lang == "ur":
        if section_key == "physical" and mode == "low":
            return ["موجودہ چال کے نتیجے میں چال کی کسی مخصوص ورزش یا فزیکل تھراپی پلان کی ضرورت سامنے نہیں آئی۔"]
        if section_key == "physical" and mode == "watch":
            return ["چال میں ہلکی تبدیلیاں نوٹ کریں؛ ابھی چال کی مخصوص ورزش یا ویڈیو کی ضرورت نہیں۔"]
        if section_key == "speech" and mode == "low":
            return ["موجودہ آواز کے نتیجے میں آواز یا بولنے کی مخصوص مشق یا ویڈیو کی ضرورت سامنے نہیں آئی۔"]
        if section_key == "speech" and mode == "watch":
            return ["آواز میں ہلکی تبدیلیاں نوٹ کریں؛ ابھی آواز یا بولنے کی مخصوص مشق یا ویڈیو کی ضرورت نہیں۔"]
        if section_key == "lifestyle" and mode == "not_needed":
            if policy.get("questionnaire_only_elevated"):
                return ["سوالنامے کی موجودہ معلومات سے کوئی خاص خوراک، پانی یا نیند کی تبدیلی ثابت نہیں ہوتی؛ صرف واقعی محسوس ہونے والی علامات پر توجہ دیں۔"]
            if policy.get("gait") == "attention":
                return ["چال کے نتیجے سے کسی خاص خوراک یا غذائی اضافے کی ضرورت سامنے نہیں آتی؛ اپنی توجہ حرکت اور توازن سے متعلق اقدامات پر رکھیں۔"]
            return ["اس اسکریننگ سے روزمرہ طرزِ زندگی میں کسی خاص اضافی تبدیلی کی ضرورت سامنے نہیں آتی؛ غیر ضروری خوراک یا غذائی اضافے شامل نہ کریں۔"]
        return []

    if section_key == "physical" and mode == "low":
        return ["Your current gait result does not need a targeted exercise or physiotherapy plan."]
    if section_key == "physical" and mode == "watch":
        return ["Mild gait changes are worth watching; no targeted gait exercise plan or gait video is needed yet."]
    if section_key == "speech" and mode == "low":
        return ["Your current speech result does not need a targeted voice exercise plan or speech video."]
    if section_key == "speech" and mode == "watch":
        return ["Mild speech changes are worth watching; no targeted speech exercise or video is needed yet."]
    if section_key == "lifestyle" and mode == "not_needed":
        if policy.get("questionnaire_only_elevated"):
            return ["Your questionnaire does not identify a specific food, fluid, or sleep action from the stored symptoms; focus on symptoms you actually notice."]
        if policy.get("gait") == "attention":
            return ["Your gait result does not call for a special diet or supplement; focus on the movement and balance actions above."]
        return ["This screening does not add a specific lifestyle change; avoid unnecessary diet or supplement changes just because a section exists."]
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


def _watch_closely_points(intelligence, lang="en"):
    """Deterministic, module-aware safety wording.

    It never says that medical care is unnecessary and never adds unrelated
    emergency symptoms. Only affected gait/speech areas and explicitly reported
    supporting symptoms may appear.
    """
    policy = _recommendation_policy(intelligence)
    gait = policy["gait"]
    speech = policy["speech"]
    flags = _normalized_questionnaire_flags(intelligence)

    swallowing_reported = any("swallow" in f or "chok" in f for f in flags)
    balance_reported = any("fall" in f or "balance" in f for f in flags)

    if lang == "ur":
        module_points = []
        if gait == "attention":
            module_points.append("چھوٹے یا گھسٹتے قدم، پاؤں رکنے جیسے احساس، مڑنے میں مشکل، بڑھتا عدم توازن یا چلنے کے لیے زیادہ سہارے کی ضرورت نوٹ کریں۔")
        elif gait == "watch":
            module_points.append("قدموں کی لمبائی، گھسٹنے، مڑنے یا توازن میں بار بار ہونے والی تبدیلیاں نوٹ کریں اور دیکھیں کہ کیا وہ بڑھ رہی ہیں۔")

        if speech == "attention":
            module_points.append("آواز بہت نرم ہونے، الفاظ کم واضح ہونے، زیادہ ہچکچاہٹ، لمبے وقفوں یا دوسروں کو سمجھنے میں بڑھتی مشکل کو نوٹ کریں۔")
        elif speech == "watch":
            module_points.append("نرم آواز، کم واضح الفاظ، زیادہ ہچکچاہٹ یا لمبے وقفوں کو نوٹ کریں اور دیکھیں کہ کیا وہ مسلسل رہتے ہیں۔")

        support_point = ""
        if swallowing_reported:
            support_point = "آپ نے نگلنے یا دم گھٹنے جیسی مشکل رپورٹ کی ہے؛ اگر کھانے یا پینے کے ساتھ یہ مسئلہ بار بار ہو تو طبی مشورہ لیں۔"
        elif balance_reported and gait != "attention":
            support_point = "آپ نے توازن یا گرتے گرتے بچنے کی مشکل رپورٹ کی ہے؛ یہ کب اور کتنی بار ہوتا ہے، نوٹ کریں، خاص طور پر اگر یہ بڑھ رہا ہو۔"

        if gait == "attention" and speech == "attention":
            escalation = "اگر چلنے یا بولنے میں اچانک واضح بگاڑ آئے یا چلنا معمول کے مطابق محفوظ نہ رہے تو فوری طبی مشورہ لیں۔"
        elif gait == "attention":
            escalation = "اگر چلنا اچانک بہت مشکل ہو جائے، توازن کی وجہ سے بار بار گرتے گرتے بچیں، یا بغیر سہارے محفوظ چلنا ممکن نہ رہے تو فوری طبی مشورہ لیں۔"
        elif speech == "attention":
            escalation = "اگر بولنا اچانک معمول سے بہت کم واضح ہو جائے یا دوسروں کے لیے سمجھنا بہت مشکل ہو جائے تو فوری طبی مشورہ لیں۔"
        elif gait == "watch" or speech == "watch":
            escalation = "اگر یہ تبدیلیاں واضح طور پر بڑھیں یا روزمرہ کام متاثر کرنے لگیں تو طبی مشورہ لیں۔"
        elif policy.get("questionnaire_only_elevated"):
            escalation = "صرف وہی علامات نوٹ کریں جو آپ نے واقعی سوالنامے میں رپورٹ کیں؛ چال یا آواز سے متعلق نئی تنبیہی علامت خود سے فرض نہ کریں۔"
        else:
            escalation = "اس اسکریننگ میں چال یا آواز سے متعلق کوئی خاص تنبیہی علامت سامنے نہیں آئی؛ پھر بھی نئی یا مسلسل پریشان کن علامت کو نظر انداز نہ کریں۔"

        points = list(module_points)
        if support_point and len(points) < 2:
            points.append(support_point)
        points.append(escalation)
        return points[:3]

    module_points = []
    if gait == "attention":
        module_points.append("Watch for shorter or shuffling steps, feet seeming to stick, harder turning, increasing unsteadiness, or needing more support while walking.")
    elif gait == "watch":
        module_points.append("Watch for repeated changes in step length, shuffling, turning, or balance, and note whether they are becoming more frequent.")

    if speech == "attention":
        module_points.append("Watch for a much softer voice, less clear words, more hesitation, longer pauses, or increasing difficulty being understood.")
    elif speech == "watch":
        module_points.append("Watch for softer voice, unclear words, more hesitation, or longer pauses, and note whether these changes persist.")

    support_point = ""
    if swallowing_reported:
        support_point = "You reported swallowing or choking difficulty; if food or drink repeatedly causes this problem, seek medical advice rather than relying only on screening."
    elif balance_reported and gait != "attention":
        support_point = "You reported balance or near-fall difficulty; record when it happens and whether it is becoming more frequent."

    if gait == "attention" and speech == "attention":
        escalation = "If walking or speech suddenly becomes clearly worse than usual, or walking is no longer safe without support, seek prompt medical advice."
    elif gait == "attention":
        escalation = "If walking suddenly becomes much harder, balance causes repeated near-falls, or you cannot walk safely without support, seek prompt medical advice."
    elif speech == "attention":
        escalation = "If speech suddenly becomes much harder to understand than usual or changes sharply over a short time, seek prompt medical advice."
    elif gait == "watch" or speech == "watch":
        escalation = "If these changes clearly worsen or begin affecting daily activities, seek medical advice rather than waiting only for the next screening."
    elif policy.get("questionnaire_only_elevated"):
        escalation = "Watch only the symptoms you actually reported in the questionnaire; do not assume a gait or speech warning that was not detected."
    else:
        escalation = "No gait- or speech-specific warning change is highlighted here; still pay attention to any new or persistent symptom that worries you."

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
            limit=32,
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
            limit=30,
        )
        if text and text not in points:
            points.append(text)
        if len(points) >= max_points:
            break
    return points



def _limit_repeated_general_referrals(payload):
    """Keep routine doctor/neurologist referral near the end of the care plan.

    Physical- and speech-therapy referrals remain allowed when they belong to the
    affected module. Generic doctor/clinician wording is removed from earlier
    sections so the page does not open with "go see a doctor" advice.
    """
    payload = dict(payload or {})
    routine_terms = re.compile(
        r"\b(doctor|neurologist|clinician|healthcare professional|medical review|medical assessment|check-up)\b",
        re.IGNORECASE,
    )
    urgent_terms = re.compile(
        r"\b(urgent|sudden|prompt medical advice|choking|unsafe without support)\b",
        re.IGNORECASE,
    )

    # Earlier practical sections should not contain routine general referral.
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

    # The deterministic watch section may contain one module-specific escalation.
    # Follow-up may contain at most one additional routine referral when useful.
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
        # Urdu continues to use RAG-selected points so translation remains handled
        # by the existing language layer.
        return []

    # Speech-focused lifestyle support: keep this predictable and simple.
    if reason == "speech_support":
        return [
            "Drink about 6 to 8 glasses of water through the day, unless you have been told to limit fluids.",
            "For meals, choose foods such as beans or eggs/fish, brown rice or other whole grains, and fruit or vegetables such as apples, tomatoes or spinach.",
        ]

    # For questionnaire-driven lifestyle issues, use the exact simple patient
    # actions stored in the retrieved knowledge cards.
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
        neutral = _neutral_section_points(key, policy, lang=lang)
        if neutral:
            payload[key] = neutral
            continue

        if key == "lifestyle":
            specific_lifestyle = _specific_lifestyle_points(
                policy,
                retrieved,
                lang=lang,
            )
            if specific_lifestyle:
                payload[key] = specific_lifestyle[:3]
                continue

        points = _normalize_points(payload.get(key), min_points=1, max_points=3)
        if not points:
            points = _points_from_retrieved(rag_section, retrieved, max_points=3)

        # Active gait/speech sections must contain something the user can do,
        # not only a referral. The retriever reserves a self-care card when one
        # is available; use it if Gemini returned referral-only wording.
        if key in {"physical", "speech"} and points:
            if all(_professional_referral_point(point) for point in points):
                self_care = _self_care_point_from_retrieved(rag_section, retrieved)
                if self_care:
                    points.insert(0, self_care)

            # Practical actions first; professional assessment, when relevant,
            # is deliberately kept as the last point.
            points = sorted(points, key=lambda point: 1 if _professional_referral_point(point) else 0)

        if not points:
            points = [
                "No additional targeted recommendation is needed for this area from the current screening."
                if lang != "ur"
                else "موجودہ اسکریننگ کے مطابق اس حصے کے لیے مزید مخصوص سفارش درکار نہیں۔"
            ]
        payload[key] = points[:3]

    # The former Medical Consultation slot is now deterministic and module-aware.
    # Do not let Gemini insert a generic doctor-first recommendation here.
    payload["medical"] = _watch_closely_points(intelligence, lang=lang)

    payload = _limit_repeated_general_referrals(payload)

    # Referral filtering must never leave a required section empty.
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

    if lang == "ur":
        return f"""آپ NeuroSense کے لیے مختصر، practical اور patient-friendly care guidance لکھ رہے ہیں۔

صرف CURATED RAG CONTEXT میں موجود advice استعمال کریں۔ کوئی نئی medicine، treatment، exercise duration، food example، fluid quantity یا diagnosis خود سے نہ بنائیں۔

اہم اصول:
1. Summary صرف 2 چھوٹے جملے ہو۔
2. Results Meaning صرف 2 مختصر points ہوں۔
3. ہر section دکھائیں، مگر garbage filler نہ دیں۔ Active section میں 1–3 specific actions ہوں؛ low/watch section میں ایک مختصر واضح line کافی ہے۔
4. Physical اور Speech sections میں user کے خود کرنے والے actions پہلے آئیں؛ professional referral اگر relevant ہو تو آخر میں ہو۔
5. Lifestyle صرف تب specific advice دے جب RAG میں واقعی reason ہو۔ Quantity یا food examples RAG میں ہوں تو exactly preserve کریں؛ generic "healthy food" یا "drink water" نہ لکھیں۔
6. Monitoring میں واضح لکھیں کہ کن changes کو note کرنا ہے۔
7. "Changes to Watch Closely" کے لیے unrelated emergency symptoms نہ بنائیں۔ صرف affected gait/speech area یا questionnaire میں واقعی report ہوئی supporting symptom استعمال کریں۔
8. یہ نہ کہیں کہ user کو doctor کی ضرورت نہیں۔ Routine doctor advice مختلف sections میں repeat نہ کریں۔
9. Follow-up میں next screening اور previous report comparison کی practical guidance دیں۔
10. confidence, threshold, score, model, fusion, discordance, certainty, RAG, algorithm, domain, longitudinal جیسے technical الفاظ patient-facing output میں نہ آئیں۔
11. percentages/decimals نہ لکھیں؛ probability page پر الگ دکھائی جاتی ہے۔
12. YouTube link نہ بنائیں؛ verified video code بعد میں attach کرے گا۔
13. Questionnaire-only elevation gait/speech therapy یا video trigger نہیں کرتا۔
14. Patient-facing جواب خالص اور آسان اردو میں لکھیں۔ Low/Medium/High، gait، speech، medical advice، diet، exercise، warning جیسے عام English الفاظ output میں نہ لکھیں۔
15. صرف verified video کا اصل title/provider بعد میں code کے ذریعے English میں رہ سکتا ہے۔

SECTION POLICY:
{section_policy}

INTERNAL PROFILE — patient کو copy نہ کریں:
{profile_hint}

CURATED RAG CONTEXT — یہی allowed recommendation source ہے:
{rag_context}

صرف valid JSON return کریں:
{{
  "summary": "2 short sentences",
  "meaning": ["point 1", "point 2"],
  "physical": ["action 1", "action 2", "optional action 3"],
  "speech": ["action 1", "action 2", "optional action 3"],
  "lifestyle": ["action 1", "action 2", "optional action 3"],
  "monitoring": ["action 1", "action 2", "optional action 3"],
  "medical": ["module-specific watch point"],
  "follow_up": ["action 1", "action 2", "optional action 3"]
}}
"""

    return f"""You write concise, practical patient guidance for NeuroSense.

Use ONLY the CURATED RAG CONTEXT below. Do not invent medicines, treatments, exercise durations, food examples, fluid quantities, diagnoses, or URLs.

STRICT RULES:
1. Summary: exactly 2 short sentences.
2. Results Meaning: exactly 2 short points.
3. Keep all six sections, but never add filler just to fill space.
4. Active Physical/Speech sections: give 1–3 specific actions. Put something the user can do first; professional assessment, if supported, goes last.
5. Low/watch Physical/Speech sections: one clear neutral line is enough. Do not attach a video unless that module is >=50%.
6. Lifestyle: be symptom-driven. If RAG provides a fluid quantity or food examples, preserve them. Never write generic "eat healthy" or "drink water" advice.
7. Symptom Monitoring: clearly name WHAT to watch or record in simple words.
8. Changes to Watch Closely: mention only the affected gait/speech area and explicitly reported supporting symptoms. Do not add unrelated emergencies such as seizures, facial drooping, loss of consciousness, or injury scenarios.
9. Never say that the user does not need medical care. Avoid routine doctor advice across multiple sections; one module-specific escalation line is enough unless an explicitly reported safety symptom requires more.
10. Follow-up: give practical next-screening and previous-report comparison guidance.
11. Do not repeat the same recommendation across sections.
12. NEVER expose these technical words: confidence, threshold, score, model, fusion, discordance, certainty, RAG, algorithm, domain, longitudinal.
13. Do not output percentages or decimals; probability is displayed separately.
14. Do not output YouTube/resource URLs; verified resources are inserted by code.
15. Questionnaire-only elevation is supporting evidence and must not trigger gait/speech therapy or videos.
16. Use everyday English that a person without medical knowledge can understand. Prefer short words and short sentences.
17. Avoid formal wording such as "maintain", "restrict fluids", "simultaneous changes", "clinical evaluation", "communication needs", or "persistent" when a simpler phrase can say the same thing.
18. If you mention food, name actual food examples from the RAG context. If you mention water, keep the quantity exactly as given in the RAG context.

SECTION POLICY:
{section_policy}

INTERNAL PROFILE — DO NOT COPY THIS TECHNICAL WORDING:
{profile_hint}

CURATED RAG CONTEXT — THIS IS THE ONLY ALLOWED RECOMMENDATION SOURCE:
{rag_context}

Return ONLY valid JSON:
{{
  "summary": "exactly 2 short sentences",
  "meaning": ["short point 1", "short point 2"],
  "physical": ["action 1", "action 2", "optional action 3"],
  "speech": ["action 1", "action 2", "optional action 3"],
  "lifestyle": ["action 1", "action 2", "optional action 3"],
  "monitoring": ["action 1", "action 2", "optional action 3"],
  "medical": ["module-specific watch point"],
  "follow_up": ["action 1", "action 2", "optional action 3"]
}}
"""

def _trim_words(text, limit):
    words = str(text or "").strip().split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).rstrip(" ,;:.") + "…"


def _clean_patient_text(text, limit=20):
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
        (r"\bpersistent\b", "keeps happening"),
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
        item = _clean_patient_text(item, limit=30)
        if item and item not in cleaned:
            cleaned.append(item)
        if len(cleaned) >= max_points:
            break

    # Do not fabricate missing medical advice in code.
    # If Gemini returns fewer than 2, keep what is supported and fallback later.
    return cleaned


def _clean_urdu_output_text(text):
    """Remove common English leftovers from patient-facing Urdu text.

    Exact external resource titles are injected after this cleanup, so official
    video titles/provider names are preserved.
    """
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
    summary = _clean_patient_text(payload.get("summary", ""), limit=34)
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
    """Build fallback text from THIS screening's retrieved cards, never a static set."""
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
    """
    Generate the existing NeuroSense recommendation report using:
    analyzers -> RAG -> optional verified resource -> Gemini wording layer.
    """

    # ------------------------------------------------------------
    # 1. EXISTING NEUROSENSE INTELLIGENCE
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 2. CURATED RAG RETRIEVAL
    # ------------------------------------------------------------
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
        # Preserve the report structure even if the embedding layer fails.
        return get_fallback_recommendations(
            intelligence=intelligence,
            lang=lang,
            retrieved={section: [] for section in REPORT_SECTIONS},
            selected_resources={},
        )

    # ------------------------------------------------------------
    # 3. VERIFIED OPTIONAL RESOURCES
    # ------------------------------------------------------------
    try:
        selected_resources = select_report_resources(
            intelligence=intelligence,
            retrieved_report=retrieved,
        )
    except Exception as exc:
        logger.warning("Resource selection failed: %s", exc)
        selected_resources = {}

    # ------------------------------------------------------------
    # 4. GEMINI = SHORT LANGUAGE LAYER ONLY
    # ------------------------------------------------------------
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
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.15,
                top_p=0.75,
                max_output_tokens=1500,
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

        # Exact curated links are inserted by code, never selected by Gemini.
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
