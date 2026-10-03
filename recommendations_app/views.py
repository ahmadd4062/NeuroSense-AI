from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.urls import reverse
from urllib.parse import urlencode
import logging
import re

from .services import generate_ai_recommendations

logger = logging.getLogger(__name__)

# Change this whenever recommendation content/layout rules change.
RECOMMENDATION_PIPELINE_VERSION = "rag_urdu_clean_v11"
CACHE_SESSION_KEY = "recommendation_cache_v11"
MAX_CACHE_ITEMS = 12


def clean_markdown(text):
    """Convert the small recommendation Markdown response to safe display HTML."""
    if not text:
        return text

    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(
        r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)',
        r'<em>\1</em>',
        text
    )

    text = re.sub(
        r'^##\s+(.*?)$',
        r'<h2 class="rec-h2">\1</h2>',
        text,
        flags=re.MULTILINE
    )

    lines = text.splitlines()
    result = []
    in_list = False

    for raw_line in lines:
        line = raw_line.strip()

        bullet = re.match(r'^[\-\*•]\s+(.+)$', line)
        if bullet:
            if not in_list:
                result.append('<ul class="rec-list">')
                in_list = True
            result.append(f'<li>{bullet.group(1)}</li>')
            continue

        if in_list:
            result.append('</ul>')
            in_list = False

        if not line:
            continue

        if line.startswith('<h2'):
            result.append(line)
        elif line.startswith('<div') or line.startswith('<span'):
            result.append(line)
        else:
            result.append(f'<p class="rec-p">{line}</p>')

    if in_list:
        result.append('</ul>')

    return "\n".join(result)


def _fusion_model_to_dict(fusion, history=None):
    """Convert a saved FusionResult into the structure used by the recommendation analyzers."""
    return {
        "screening_id": fusion.screening_id,
        "final_prediction": fusion.final_prediction,
        "final_pd_probability": fusion.final_pd_probability,
        "fusion_confidence": fusion.fusion_confidence,
        "risk_level": fusion.risk_level,
        "created_at": fusion.created_at,
        "modules": {
            "gait": {
                "prediction": fusion.gait_prediction,
                "pd_probability": fusion.gait_pd_probability,
                "weight": fusion.gait_weight,
            },
            "speech": {
                "prediction": fusion.speech_prediction,
                "pd_probability": fusion.speech_pd_probability,
                "weight": fusion.speech_weight,
            },
            "questionnaire": {
                "prediction": fusion.questionnaire_prediction,
                "pd_probability": fusion.questionnaire_pd_probability,
                "weight": fusion.questionnaire_weight,
                "motor_score": getattr(history, "motor_score", None) if history else None,
            },
        },
    }


def _historical_questionnaire_stub(fusion, history=None):
    """
    Historical records may not retain the full questionnaire row ID.
    This small structure still gives the pattern detector the saved module result.
    """
    return {
        "prediction": fusion.questionnaire_prediction,
        "pd_probability": fusion.questionnaire_pd_probability,
        "motor_score": getattr(history, "motor_score", None) if history else None,
    }


def _probability_percent(fusion_result):
    value = fusion_result.get("final_pd_probability", 0) if fusion_result else 0
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        value = 0.0

    if value <= 1:
        value *= 100

    return max(0.0, min(100.0, round(value, 1)))


def _cache_key(screening_id, lang):
    return f"{RECOMMENDATION_PIPELINE_VERSION}:{screening_id}:{lang}"


def _trim_cache(cache):
    if len(cache) <= MAX_CACHE_ITEMS:
        return cache

    # Session dictionaries preserve insertion order in modern Python.
    while len(cache) > MAX_CACHE_ITEMS:
        first_key = next(iter(cache))
        cache.pop(first_key, None)
    return cache


def _resolve_language(request, explicit_lang=None):
    """
    Resolve the site's active language consistently.

    The NeuroSense navbar stores the selected language in:
        request.session["language"]
    and also in the "language" cookie.

    Session is intentionally checked BEFORE a query-string value so an old
    ?lang=en in the current URL cannot override a newly selected Urdu navbar
    preference after the language switch redirects back to the same page.
    """
    candidates = [
        request.session.get("language"),
        explicit_lang,
        request.GET.get("lang"),
        request.COOKIES.get("language"),
        getattr(request, "LANG", None),
        "en",
    ]

    for value in candidates:
        value = str(value or "").strip().lower()

        if value.startswith("ur"):
            return "ur"

        if value.startswith("en"):
            return "en"

    return "en"


@login_required
def recommendations_page(request):
    """
    Show recommendations for the EXACT screening requested.

    Historical mode:
        /recommendations/?screening_id=<id>

    Current mode:
        /recommendations/
        Uses the active screening stored in session.
    """
    lang = _resolve_language(request)
    selected_screening_id = request.GET.get("screening_id")

    questionnaire_data = None
    questionnaire_result_id = None
    historical_report = False

    # ---------------------------------------------------------
    # HISTORICAL SCREENING
    # ---------------------------------------------------------
    if selected_screening_id:
        try:
            from fusion_app.models import FusionResult
            from dashboard.models import ScreeningHistory

            history = ScreeningHistory.objects.filter(
                user=request.user,
                screening_id=selected_screening_id,
            ).first()

            if not history:
                raise ValueError("The selected screening could not be found.")

            saved_fusion = FusionResult.objects.filter(
                screening_id=selected_screening_id
            ).order_by("-created_at").first()

            if not saved_fusion:
                raise ValueError("The saved report for this screening is unavailable.")

            fusion_result = _fusion_model_to_dict(saved_fusion, history=history)
            questionnaire_data = _historical_questionnaire_stub(
                saved_fusion,
                history=history,
            )
            current_screening_id = str(selected_screening_id)
            historical_report = True

        except Exception as exc:
            logger.exception(
                "Could not load historical recommendation screening %s",
                selected_screening_id,
            )
            return render(
                request,
                "recommendations_app/recommendations.html",
                {
                    "error": str(exc),
                    "lang": lang,
                },
            )

    # ---------------------------------------------------------
    # CURRENT SCREENING
    # ---------------------------------------------------------
    else:
        fusion_result = request.session.get("fusion_result")
        questionnaire_data = request.session.get("questionnaire_result")
        questionnaire_result_id = request.session.get("questionnaire_result_id")

        current_screening_id = (
            request.session.get("screening_id")
            or (fusion_result.get("screening_id") if fusion_result else None)
        )

        if not fusion_result or not current_screening_id:
            error_msg = (
                "No screening data found. Please complete the screening first."
                if lang == "en"
                else "کوئی اسکریننگ ڈیٹا نہیں ملا۔ براہ کرم پہلے اسکریننگ مکمل کریں۔"
            )
            return render(
                request,
                "recommendations_app/recommendations.html",
                {
                    "error": error_msg,
                    "lang": lang,
                },
            )

    # ---------------------------------------------------------
    # SCREENING-SPECIFIC CACHE
    # ---------------------------------------------------------
    cache = request.session.get(CACHE_SESSION_KEY, {})
    if not isinstance(cache, dict):
        cache = {}

    key = _cache_key(current_screening_id, lang)
    recommendations = cache.get(key)

    if recommendations:
        logger.info(
            "Recommendation cache HIT | screening=%s | lang=%s | pipeline=%s",
            current_screening_id,
            lang,
            RECOMMENDATION_PIPELINE_VERSION,
        )

    if not recommendations:
        logger.info(
            "Recommendation cache MISS - generating fresh output | screening=%s | historical=%s | pipeline=%s",
            current_screening_id,
            historical_report,
            RECOMMENDATION_PIPELINE_VERSION,
        )

        user = request.user if request.user.is_authenticated else None

        raw_recommendations = generate_ai_recommendations(
            user=user,
            fusion_result=fusion_result,
            questionnaire_result=questionnaire_data,
            questionnaire_result_id=questionnaire_result_id,
            screening_id=current_screening_id,
            lang=lang,
        )

        recommendations = clean_markdown(raw_recommendations)

        cache[key] = recommendations
        request.session[CACHE_SESSION_KEY] = _trim_cache(cache)
        request.session.modified = True

    context = {
        "recommendations": recommendations,
        "fusion": fusion_result,
        "pd_probability_percent": _probability_percent(fusion_result),
        "screening_id": current_screening_id,
        "historical_report": historical_report,
        "lang": lang,
        "has_recommendations": bool(recommendations),
        "recommendations_pipeline_version": RECOMMENDATION_PIPELINE_VERSION,
    }

    return render(
        request,
        "recommendations_app/recommendations.html",
        context,
    )


@login_required
def regenerate_recommendations(request):
    """Regenerate only the recommendation belonging to the selected screening."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    screening_id = request.POST.get("screening_id") or request.session.get("screening_id")
    lang = _resolve_language(request, explicit_lang=request.POST.get("lang"))

    cache = request.session.get(CACHE_SESSION_KEY, {})
    if isinstance(cache, dict) and screening_id:
        screening_text = str(screening_id)
        keys_to_remove = [
            key for key in list(cache.keys())
            if screening_text in str(key)
        ]
        for key in keys_to_remove:
            cache.pop(key, None)

        request.session[CACHE_SESSION_KEY] = cache
        request.session.modified = True

        logger.info(
            "Cleared recommendation cache for screening=%s (%d entries)",
            screening_id,
            len(keys_to_remove),
        )

    url = reverse("recommendations_app:recommendations")
    params = {}

    if screening_id:
        params["screening_id"] = screening_id
    if lang:
        params["lang"] = lang

    if params:
        url = f"{url}?{urlencode(params)}"

    return redirect(url)


def save_recommendations(request):
    if request.method == "POST":
        pass
    return JsonResponse({"error": "Method not allowed"}, status=405)
