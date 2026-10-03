from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import logging
from django.contrib.auth.decorators import login_required
from django.utils import timezone

from .services import calculate_fusion, save_fusion_result

logger = logging.getLogger(__name__)


def _resolve_language(request):
    """
    Resolve NeuroSense language consistently with the navbar.

    The navbar stores the selected language in the session/cookie.
    Session is checked before an old ?lang=en value so switching to Urdu
    immediately updates this report as well.
    """
    candidates = [
        request.session.get('language'),
        request.GET.get('lang'),
        request.COOKIES.get('language'),
        getattr(request, 'LANG', None),
        'en',
    ]

    for value in candidates:
        value = str(value or '').strip().lower()

        if value.startswith('ur'):
            return 'ur'

        if value.startswith('en'):
            return 'en'

    return 'en'


# Report-only interpretation thresholds.
# These DO NOT change the fusion/model calculation.
MODULE_WATCH_THRESHOLD = 0.30
MODULE_NOTICEABLE_THRESHOLD = 0.50


def _safe_probability(value):
    """Return a module probability as a safe 0-1 float."""
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _module_attention_state(probability):
    """Classify a module for report interpretation only."""
    probability = _safe_probability(probability)

    if probability >= MODULE_NOTICEABLE_THRESHOLD:
        return 'noticeable'
    if probability >= MODULE_WATCH_THRESHOLD:
        return 'watch'
    return 'low'


def _build_report_insights(fusion_result, lang='en'):
    """
    Build patient-friendly report interpretation from the individual modules.

    The overall weighted PD probability is kept exactly as calculated by the
    fusion service. This helper only prevents a strong gait/speech signal from
    being hidden by a low combined score.

    Gait and speech are treated as the primary objective screening areas.
    Questionnaire findings are supportive context only.
    """
    modules = (fusion_result or {}).get('modules', {}) or {}

    gait_probability = _safe_probability(
        (modules.get('gait') or {}).get('pd_probability')
    )
    speech_probability = _safe_probability(
        (modules.get('speech') or {}).get('pd_probability')
    )
    questionnaire_probability = _safe_probability(
        (modules.get('questionnaire') or {}).get('pd_probability')
    )

    gait_state = _module_attention_state(gait_probability)
    speech_state = _module_attention_state(speech_probability)
    questionnaire_state = _module_attention_state(questionnaire_probability)

    risk_level = str((fusion_result or {}).get('risk_level') or 'Low')

    # Patient-facing Urdu label for the overall screening level.
    # The stored/model risk level remains unchanged.
    risk_level_ur = {
        'low': 'کم',
        'medium': 'درمیانہ',
        'high': 'زیادہ',
    }.get(risk_level.lower(), risk_level)

    gait_pct = round(gait_probability * 100)
    speech_pct = round(speech_probability * 100)
    questionnaire_pct = round(questionnaire_probability * 100)

    attention_items = []
    next_steps = []

    if lang == 'ur':
        if gait_state == 'noticeable':
            attention_items.append({
                'module': 'gait',
                'title': 'چال اور توازن',
                'status': 'توجہ درکار',
                'level': 'noticeable',
                'score': gait_pct,
                'show_score': True,
                'message': 'چلنے یا توازن میں نمایاں تبدیلیاں نظر آئی ہیں۔ پاؤں گھسیٹ کر چلنے، مڑنے، رفتار اور توازن پر نظر رکھیں۔',
            })
            next_steps.append('روزمرہ چلنے میں توازن، پاؤں گھسیٹ کر چلنے، سست رفتار یا مڑنے میں مشکل کو نوٹ کریں۔')
        elif gait_state == 'watch':
            attention_items.append({
                'module': 'gait',
                'title': 'چال اور توازن',
                'status': 'نگرانی کریں',
                'level': 'watch',
                'score': gait_pct,
                'show_score': True,
                'message': 'چال میں ہلکی تبدیلیاں نظر آئی ہیں۔ اگر توازن یا چلنے کے انداز میں مزید تبدیلی آئے تو نوٹ کریں۔',
            })
            next_steps.append('اگلی اسکریننگ تک چلنے، توازن اور موڑنے میں نئی تبدیلیوں پر نظر رکھیں۔')

        if speech_state == 'noticeable':
            attention_items.append({
                'module': 'speech',
                'title': 'آواز اور بولنے کا انداز',
                'status': 'توجہ درکار',
                'level': 'noticeable',
                'score': speech_pct,
                'show_score': True,
                'message': 'آواز یا بولنے میں نمایاں تبدیلیاں نظر آئی ہیں۔ آواز کی شدت، وضاحت، وقفوں اور رفتار پر نظر رکھیں۔',
            })
            next_steps.append('آواز نرم ہونے، الفاظ غیر واضح ہونے، وقفے بڑھنے یا بولنے کی رفتار بدلنے کو نوٹ کریں۔')
        elif speech_state == 'watch':
            attention_items.append({
                'module': 'speech',
                'title': 'آواز اور بولنے کا انداز',
                'status': 'نگرانی کریں',
                'level': 'watch',
                'score': speech_pct,
                'show_score': True,
                'message': 'آواز یا بولنے میں ہلکی تبدیلیاں نظر آئی ہیں۔ وقت کے ساتھ واضح تبدیلی آئے تو نوٹ کریں۔',
            })
            next_steps.append('اگلی اسکریننگ تک آواز کی شدت، وضاحت اور بولنے کے وقفوں میں تبدیلی پر نظر رکھیں۔')

        if not attention_items:
            attention_items.append({
                'module': 'overall',
                'title': 'کوئی نمایاں انفرادی تبدیلی نہیں',
                'status': 'معمول کی نگرانی',
                'level': 'low',
                'score': None,
                'show_score': False,
                'message': 'چال اور آواز کے حصوں میں فی الحال کوئی نمایاں انفرادی تبدیلی نظر نہیں آئی۔',
            })
            next_steps.append('مستقبل کی اسکریننگ اسی طرح کے حالات میں مکمل کریں تاکہ تبدیلیوں کا بہتر موازنہ ہو سکے۔')

        if questionnaire_state == 'noticeable':
            questionnaire_note = (
                f'سوالنامہ میں بھی کچھ معاون علامات رپورٹ ہوئی ہیں ({questionnaire_pct}٪)۔ '
                'اسے معاون معلومات کے طور پر دیکھا جاتا ہے، بنیادی اشارے کے طور پر نہیں۔'
            )
        elif questionnaire_state == 'watch':
            questionnaire_note = (
                'سوالنامہ میں چند ہلکی معاون علامات رپورٹ ہوئی ہیں۔ '
                'یہ چال اور آواز کے نتائج کے ساتھ معاون معلومات کے طور پر استعمال ہوتی ہیں۔'
            )
        else:
            questionnaire_note = (
                'سوالنامہ میں فی الحال کم معاون علامات رپورٹ ہوئی ہیں۔ '
                'اہم توجہ چال اور آواز کے حصوں پر رکھی جاتی ہے۔'
            )

        if gait_state == 'noticeable' and speech_state == 'noticeable':
            summary = (
                f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے، اور چال اور آواز دونوں میں نمایاں تبدیلیاں نظر آئی ہیں۔ '
                'دونوں حصوں پر اگلی اسکریننگ تک نظر رکھنا اہم ہے۔'
            )
        elif gait_state == 'noticeable':
            summary = (
                f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے، لیکن چال اور توازن کے نتائج میں نمایاں تبدیلیاں نظر آئی ہیں۔ '
                'مجموعی اسکور کم ہونے کی صورت میں بھی اس حصے کو نظر انداز نہیں کرنا چاہیے۔'
            )
        elif speech_state == 'noticeable':
            summary = (
                f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے، لیکن آواز اور بولنے کے نتائج میں نمایاں تبدیلیاں نظر آئی ہیں۔ '
                'مجموعی اسکور کم ہونے کی صورت میں بھی اس حصے پر نظر رکھنی چاہیے۔'
            )
        elif gait_state == 'watch' and speech_state == 'watch':
            summary = (
                f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے۔ چال اور آواز دونوں میں ہلکی تبدیلیاں ہیں جن پر وقت کے ساتھ نظر رکھنا مفید ہے۔'
            )
        elif gait_state == 'watch':
            summary = f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے۔ چال میں ہلکی تبدیلیاں ہیں جن پر وقت کے ساتھ نظر رکھنا بہتر ہے۔'
        elif speech_state == 'watch':
            summary = f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے۔ آواز یا بولنے میں ہلکی تبدیلیاں ہیں جن پر وقت کے ساتھ نظر رکھنا بہتر ہے۔'
        else:
            summary = f'آپ کی مجموعی اسکریننگ سطح {risk_level_ur} ہے اور چال یا آواز میں فی الحال کوئی نمایاں انفرادی تبدیلی نظر نہیں آئی۔'

    else:
        if gait_state == 'noticeable':
            attention_items.append({
                'module': 'gait',
                'title': 'Walking & Balance',
                'status': 'Needs attention',
                'level': 'noticeable',
                'score': gait_pct,
                'show_score': True,
                'message': 'Noticeable gait changes were detected. Watch balance, shuffling, turning and walking speed in daily activity.',
            })
            next_steps.append('Track balance, shuffling, slower walking, or difficulty turning during normal daily activity.')
        elif gait_state == 'watch':
            attention_items.append({
                'module': 'gait',
                'title': 'Walking & Balance',
                'status': 'Keep watching',
                'level': 'watch',
                'score': gait_pct,
                'show_score': True,
                'message': 'Mild gait changes were detected. Keep an eye on balance and changes in your walking pattern.',
            })
            next_steps.append('Watch for new changes in walking, balance, or turning before your next screening.')

        if speech_state == 'noticeable':
            attention_items.append({
                'module': 'speech',
                'title': 'Voice & Speech',
                'status': 'Needs attention',
                'level': 'noticeable',
                'score': speech_pct,
                'show_score': True,
                'message': 'Noticeable speech changes were detected. Watch voice volume, clarity, pauses and speaking speed.',
            })
            next_steps.append('Track softer voice, unclear words, longer pauses, or changes in speaking speed.')
        elif speech_state == 'watch':
            attention_items.append({
                'module': 'speech',
                'title': 'Voice & Speech',
                'status': 'Keep watching',
                'level': 'watch',
                'score': speech_pct,
                'show_score': True,
                'message': 'Mild speech changes were detected. Keep watching voice strength, clarity and pauses over time.',
            })
            next_steps.append('Watch voice volume, clarity, and speaking pauses until your next screening.')

        if not attention_items:
            attention_items.append({
                'module': 'overall',
                'title': 'No strong module-specific change',
                'status': 'Routine monitoring',
                'level': 'low',
                'score': None,
                'show_score': False,
                'message': 'Neither gait nor speech currently shows a strong individual signal that needs extra attention.',
            })
            next_steps.append('Repeat future screenings under similar conditions so changes can be compared more meaningfully.')

        if questionnaire_state == 'noticeable':
            questionnaire_note = (
                f'The questionnaire also reported supporting symptoms ({questionnaire_pct}%). '
                'These are treated as supporting context rather than the main screening signal.'
            )
        elif questionnaire_state == 'watch':
            questionnaire_note = (
                'The questionnaire reported some mild supporting symptoms. '
                'These are considered alongside the gait and speech results.'
            )
        else:
            questionnaire_note = (
                'The questionnaire currently shows fewer supporting symptoms. '
                'Primary attention is based mainly on gait and speech findings.'
            )

        if gait_state == 'noticeable' and speech_state == 'noticeable':
            summary = (
                f'Your combined screening level is {risk_level}, but both gait and speech show noticeable changes. '
                'Both areas should be monitored even if the combined score is lower than either module alone.'
            )
        elif gait_state == 'noticeable':
            summary = (
                f'Your combined screening level is {risk_level}, but gait and balance show noticeable changes. '
                'This module should still be monitored even if your overall score is low.'
            )
        elif speech_state == 'noticeable':
            summary = (
                f'Your combined screening level is {risk_level}, but voice and speech show noticeable changes. '
                'This module should still be monitored even if your overall score is low.'
            )
        elif gait_state == 'watch' and speech_state == 'watch':
            summary = (
                f'Your combined screening level is {risk_level}. Mild changes are present in both gait and speech, so both are worth watching over time.'
            )
        elif gait_state == 'watch':
            summary = f'Your combined screening level is {risk_level}. Gait shows mild changes that are worth monitoring over time.'
        elif speech_state == 'watch':
            summary = f'Your combined screening level is {risk_level}. Speech shows mild changes that are worth monitoring over time.'
        else:
            summary = f'Your combined screening level is {risk_level}, and neither gait nor speech currently shows a strong individual change.'

    return {
        'gait_state': gait_state,
        'speech_state': speech_state,
        'questionnaire_state': questionnaire_state,
        'gait_percent': gait_pct,
        'speech_percent': speech_pct,
        'questionnaire_percent': questionnaire_pct,
        'attention_items': attention_items,
        'questionnaire_note': questionnaire_note,
        'summary': summary,
        'next_steps': next_steps,
        'has_noticeable_gait': gait_state == 'noticeable',
        'has_noticeable_speech': speech_state == 'noticeable',
        'has_both_noticeable': (
            gait_state == 'noticeable' and speech_state == 'noticeable'
        ),
    }


def _fusion_model_to_report_dict(fusion, history=None):
    """
    Convert a saved FusionResult model instance into the same nested structure
    expected by fusion_app/report.html.
    """
    return {
        'screening_id': fusion.screening_id,
        'final_prediction': fusion.final_prediction,
        'final_pd_probability': fusion.final_pd_probability,
        'fusion_confidence': fusion.fusion_confidence,
        'risk_level': fusion.risk_level,
        'created_at': fusion.created_at,

        'modules': {
            'gait': {
                'prediction': fusion.gait_prediction,
                'pd_probability': fusion.gait_pd_probability,
                'weight': fusion.gait_weight,
            },
            'speech': {
                'prediction': fusion.speech_prediction,
                'pd_probability': fusion.speech_pd_probability,
                'weight': fusion.speech_weight,
            },
            'questionnaire': {
                'prediction': fusion.questionnaire_prediction,
                'pd_probability': fusion.questionnaire_pd_probability,
                'weight': fusion.questionnaire_weight,
                'motor_score': getattr(history, 'motor_score', None) if history else None,
            },
        },
    }


@login_required
def report_page(request):
    """
    Render the final fusion report.

    Two modes are supported:

    1) Historical report:
       /fusion/report/?screening_id=<id>
       Loads that exact saved screening from the database.

    2) Current screening:
       /fusion/report/
       Uses the current module data stored in the session and calculates/saves
       fusion exactly as before.
    """
    # Keep report language in sync with the NeuroSense navbar.
    # The navbar/session language takes priority over stale URL parameters.
    lang = _resolve_language(request)
    selected_screening_id = request.GET.get('screening_id')

    # ---------------------------------------------------------
    # HISTORICAL REPORT MODE
    # ---------------------------------------------------------
    # This MUST run before the session checks. Otherwise clicking "Full Report"
    # on an older dashboard record would still use the current/latest session.
    if selected_screening_id:
        try:
            from .models import FusionResult
            from dashboard.models import ScreeningHistory

            # Make sure the selected screening actually belongs to this user.
            history = ScreeningHistory.objects.filter(
                user=request.user,
                screening_id=selected_screening_id
            ).first()

            if not history:
                message = (
                    'یہ اسکریننگ ریکارڈ نہیں ملا۔'
                    if lang == 'ur'
                    else 'The selected screening record could not be found.'
                )
                return render(request, 'fusion_app/report.html', {
                    'error': message,
                    'lang': lang
                })

            # Load the exact fusion record matching the clicked screening.
            saved_fusion = FusionResult.objects.filter(
                screening_id=selected_screening_id
            ).order_by('-created_at').first()

            if not saved_fusion:
                message = (
                    'اس اسکریننگ کی مکمل رپورٹ دستیاب نہیں ہے۔'
                    if lang == 'ur'
                    else 'The full report for this screening is not available.'
                )
                return render(request, 'fusion_app/report.html', {
                    'error': message,
                    'lang': lang
                })

            fusion_result = _fusion_model_to_report_dict(
                saved_fusion,
                history=history
            )

            report_insights = _build_report_insights(
                fusion_result,
                lang=lang
            )

            return render(request, 'fusion_app/report.html', {
                'fusion': fusion_result,
                'report_insights': report_insights,
                'saved': True,
                'historical_report': True,
                'report_created_at': saved_fusion.created_at,
                'lang': lang
            })

        except Exception as e:
            logger.exception(
                "Failed to load historical fusion report for screening %s",
                selected_screening_id
            )

            message = (
                f'رپورٹ لوڈ کرنے میں خرابی: {str(e)}'
                if lang == 'ur'
                else f'Error loading report: {str(e)}'
            )

            return render(request, 'fusion_app/report.html', {
                'error': message,
                'lang': lang
            })

    # ---------------------------------------------------------
    # CURRENT SCREENING MODE
    # ---------------------------------------------------------
    # Existing workflow: use module results from the current session.
    gait_data = request.session.get('gait_result')
    speech_data = request.session.get('audio_module_result')
    questionnaire_data = request.session.get('questionnaire_result')

    error_messages = {
        'gait': {
            'en': 'Gait analysis not completed. Please complete the gait screening first.',
            'ur': 'چال کا تجزیہ مکمل نہیں ہوا۔ براہ کرم پہلے چال اسکریننگ مکمل کریں۔'
        },
        'speech': {
            'en': 'Speech analysis not completed. Please complete the speech screening first.',
            'ur': 'آواز کا تجزیہ مکمل نہیں ہوا۔ براہ کرم پہلے آواز اسکریننگ مکمل کریں۔'
        },
        'questionnaire': {
            'en': 'Questionnaire not completed. Please complete the questionnaire first.',
            'ur': 'سوالنامہ مکمل نہیں ہوا۔ براہ کرم پہلے سوالنامہ مکمل کریں۔'
        },
        'fusion': {
            'en': 'Error calculating fusion: ',
            'ur': 'فیوژن کیلکولیشن میں خرابی: '
        }
    }

    if not gait_data:
        return render(request, 'fusion_app/report.html', {
            'error': error_messages['gait'][lang],
            'lang': lang
        })

    if not speech_data:
        return render(request, 'fusion_app/report.html', {
            'error': error_messages['speech'][lang],
            'lang': lang
        })

    if not questionnaire_data:
        return render(request, 'fusion_app/report.html', {
            'error': error_messages['questionnaire'][lang],
            'lang': lang
        })

    try:
        user = request.user if request.user.is_authenticated else None

        # Existing fusion calculation
        fusion_result = calculate_fusion(
            gait_data,
            speech_data,
            questionnaire_data,
            user
        )

        # Existing database save
        saved_fusion = save_fusion_result(fusion_result, user)

        # Keep the report timestamp separate from fusion_result.
        # fusion_result is stored in the Django session, so adding a datetime
        # object to it would cause: "Object of type datetime is not JSON serializable".
        report_created_at = (
            saved_fusion.created_at
            if saved_fusion is not None
            else timezone.now()
        )

        # IMPORTANT:
        # Do not place Django datetime objects inside fusion_result because
        # fusion_result is stored in request.session below. Django's default
        # JSON session serializer cannot serialize datetime objects.
        #
        # The saved FusionResult already keeps created_at in the database, and
        # historical reports load it directly from the model when needed.

        try:
            from dashboard.models import ScreeningHistory

            screening_id = fusion_result.get('screening_id')

            if not ScreeningHistory.objects.filter(
                user=user,
                screening_id=screening_id
            ).exists():

                ScreeningHistory.objects.create(
                    user=user,
                    screening_id=screening_id,
                    risk_level=fusion_result.get('risk_level', 'Low'),
                    final_pd_probability=fusion_result.get(
                        'final_pd_probability', 0
                    ),
                    fusion_confidence=fusion_result.get(
                        'fusion_confidence', 0
                    ),
                    gait_prediction=fusion_result.get(
                        'modules', {}
                    ).get('gait', {}).get('prediction', 'N/A'),
                    speech_prediction=fusion_result.get(
                        'modules', {}
                    ).get('speech', {}).get('prediction', 'N/A'),
                    questionnaire_prediction=fusion_result.get(
                        'modules', {}
                    ).get('questionnaire', {}).get('prediction', 'N/A'),
                    gait_pd_probability=fusion_result.get(
                        'modules', {}
                    ).get('gait', {}).get('pd_probability', None),
                    speech_pd_probability=fusion_result.get(
                        'modules', {}
                    ).get('speech', {}).get('pd_probability', None),
                    questionnaire_pd_probability=fusion_result.get(
                        'modules', {}
                    ).get('questionnaire', {}).get('pd_probability', None),
                    motor_score=fusion_result.get(
                        'modules', {}
                    ).get('questionnaire', {}).get('motor_score', None),
                )

                logger.info(
                    "Screening history saved for user %s, screening: %s",
                    user.username,
                    screening_id
                )

                try:
                    profile = user.profile
                    profile.total_screenings += 1

                    profile.last_screening_date = timezone.now()
                    profile.save()

                except Exception as profile_error:
                    logger.warning(
                        "Failed to update user profile: %s",
                        str(profile_error)
                    )

        except ImportError:
            logger.warning(
                "Dashboard app not installed. Skipping screening history save."
            )

        except Exception as history_error:
            logger.error(
                "Failed to save screening history: %s",
                str(history_error)
            )

        # Keep original current-screening session behavior.
        request.session['fusion_result'] = fusion_result
        request.session['screening_id'] = fusion_result.get('screening_id')
        request.session.modified = True

        current_screening_id = fusion_result.get('screening_id')
        last_cached_screening_id = request.session.get(
            'recommendations_screening_id'
        )

        if (
            last_cached_screening_id
            and last_cached_screening_id != current_screening_id
        ):
            logger.info(
                "New screening detected (%s -> %s). "
                "Clearing stale recommendations cache.",
                last_cached_screening_id,
                current_screening_id
            )

            request.session.pop('recommendations', None)
            request.session.pop('recommendations_lang', None)
            request.session.pop('recommendations_screening_id', None)
            request.session.pop('generating_recommendations', None)
            request.session.modified = True

        report_insights = _build_report_insights(
            fusion_result,
            lang=lang
        )

        return render(request, 'fusion_app/report.html', {
            'fusion': fusion_result,
            'report_insights': report_insights,
            'gait': gait_data,
            'speech': speech_data,
            'questionnaire': questionnaire_data,
            'saved': saved_fusion is not None,
            'historical_report': False,
            'report_created_at': report_created_at,
            'lang': lang
        })

    except Exception as e:
        logger.exception("Fusion calculation error")

        return render(request, 'fusion_app/report.html', {
            'error': f"{error_messages['fusion'][lang]}{str(e)}",
            'lang': lang
        })


@login_required
@require_GET
def get_fusion_data(request, screening_id):
    """
    API endpoint to get fusion data by screening_id.

    Access is limited to screening records belonging to the logged-in user.
    """
    try:
        from .models import FusionResult
        from dashboard.models import ScreeningHistory

        # Ownership check
        if not ScreeningHistory.objects.filter(
            user=request.user,
            screening_id=screening_id
        ).exists():
            return JsonResponse(
                {'error': 'Screening not found.'},
                status=404
            )

        fusion = FusionResult.objects.filter(
            screening_id=screening_id
        ).order_by('-created_at').first()

        if not fusion:
            return JsonResponse(
                {'error': 'Fusion result not found.'},
                status=404
            )

        data = {
            'screening_id': fusion.screening_id,
            'final_prediction': fusion.final_prediction,
            'final_pd_probability': fusion.final_pd_probability,
            'fusion_confidence': fusion.fusion_confidence,
            'risk_level': fusion.risk_level,
            'gait': {
                'prediction': fusion.gait_prediction,
                'pd_probability': fusion.gait_pd_probability
            },
            'speech': {
                'prediction': fusion.speech_prediction,
                'pd_probability': fusion.speech_pd_probability
            },
            'questionnaire': {
                'prediction': fusion.questionnaire_prediction,
                'pd_probability': fusion.questionnaire_pd_probability
            },
            'weights': {
                'gait': fusion.gait_weight,
                'speech': fusion.speech_weight,
                'questionnaire': fusion.questionnaire_weight
            },
            'created_at': timezone.localtime(fusion.created_at).isoformat()
        }

        return JsonResponse(data, status=200)

    except Exception as e:
        logger.exception(
            "Failed to return fusion data for screening %s",
            screening_id
        )
        return JsonResponse({'error': str(e)}, status=404)
