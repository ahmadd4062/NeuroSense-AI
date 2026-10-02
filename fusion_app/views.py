from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import logging
from django.contrib.auth.decorators import login_required
from django.utils import timezone

from .services import calculate_fusion, save_fusion_result

logger = logging.getLogger(__name__)


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
    lang = request.GET.get('lang', 'en')
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

            return render(request, 'fusion_app/report.html', {
                'fusion': fusion_result,
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

        return render(request, 'fusion_app/report.html', {
            'fusion': fusion_result,
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
