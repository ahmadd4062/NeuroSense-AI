from django.shortcuts import render, redirect
from django.db import transaction
from django.contrib.auth.decorators import login_required
import uuid

from .models import QuestionnaireResult
from .services.questionnaire_service import predict_questionnaire


@login_required
def questionnaire_home(request):
    """Render the questionnaire home page with language support"""
    return render(
        request,
        "questionnaire_app/questionnaire_home.html",
        {
            'lang': request.LANG
        }
    )


@login_required
def questionnaire_form(request):
    """Render the questionnaire form page with language support"""

    if request.method == "GET":
        return render(
            request,
            "questionnaire_app/questionnaire_form.html",
            {
                'lang': request.LANG
            }
        )

    try:
        raw_values = {
            "AGE_AT_VISIT": request.POST.get("age"),
            "ANYFAMPD": request.POST.get("family_history"),
            "ARMLGSHK": request.POST.get("arm_leg_shaking"),
            "MVSLOW": request.POST.get("movement_slowness"),
            "FTSTUCK": request.POST.get("feet_stuck"),
            "SHUFFLE": request.POST.get("shuffling_walk"),
            "POORBAL": request.POST.get("poor_balance"),
            "WRTSMLR": request.POST.get("small_handwriting"),
            "LSSXPRSS": request.POST.get("reduced_facial_expression"),
            "VOICSFTR": request.POST.get("soft_voice"),
            "NP1COG": request.POST.get("cognitive_difficulty"),
        }

        result = predict_questionnaire(raw_values)

        # ============================================
        # SCREENING_ID — Create once per screening session
        # ============================================
        # Start of a new screening: clear previous screening data
        # so speech/gait use a fresh screening_id
        screening_id = uuid.uuid4().hex
        request.session['screening_id'] = screening_id
        request.session.modified = True

        # Clear stale data from any previous screening
        request.session.pop('audio_module_result', None)
        request.session.pop('gait_result', None)
        request.session.pop('fusion_result', None)
        request.session.pop('recommendations', None)
        request.session.pop('recommendations_lang', None)
        request.session.pop('recommendations_screening_id', None)
        request.session.modified = True

        with transaction.atomic():

            saved_result = QuestionnaireResult.objects.create(
                final_prediction=result[
                    "final_prediction"
                ],

                module_confidence=result[
                    "module_confidence"
                ],

                confidence_level=result[
                    "confidence_level"
                ],

                motor_score=result[
                    "motor_score"
                ],

                motor_pattern=result[
                    "motor_pattern"
                ],

                positive_motor_symptoms=result[
                    "positive_motor_symptoms"
                ],

                speech_cognition_score=result[
                    "speech_cognition_score"
                ],

                secondary_support_level=result[
                    "secondary_support_level"
                ],

                response_reliability=result[
                    "response_reliability"
                ],

                reliability_level=result[
                    "reliability_level"
                ],

                uncertain_answers=result[
                    "uncertain_answers"
                ],

                age_in_model_range=result[
                    "age_in_model_range"
                ],

                internal_model_score=result[
                    "internal_model_score"
                ],

                internal_model_zone=result[
                    "internal_model_zone"
                ],

                requires_fusion_review=result[
                    "requires_fusion_review"
                ],

                patient_interpretation=result[
                    "patient_interpretation"
                ],

                safety_flags=result[
                    "safety_flags"
                ],
            )

            request.session[
                "questionnaire_result_id"
            ] = saved_result.id

            request.session['questionnaire_result'] = {
                'prediction': result['final_prediction'],
                'pd_probability': result['internal_model_score'],
                'control_probability': 1 - result['internal_model_score'],
                'confidence': result['module_confidence'],
                'motor_score': result['motor_score'],
                'motor_pattern': result['motor_pattern'],
                'support_level': result['secondary_support_level'],
                'reliability': result['reliability_level'],
                'positive_motor_symptoms': result['positive_motor_symptoms'],
                'uncertain_answers': result['uncertain_answers'],
                'age_in_model_range': result['age_in_model_range'],
                'internal_model_zone': result['internal_model_zone'],
                'final_prediction_rule_based': result['final_prediction'],
            }

            request.session.modified = True

        return redirect(
            "speech_app:screening"
        )

    except (ValueError, TypeError) as error:

        return render(
            request,
            "questionnaire_app/questionnaire_form.html",
            {
                "error": str(error),
                'lang': request.LANG
            }
        )

    except Exception as error:

        print(
            "Questionnaire processing error:",
            error
        )

        return render(
            request,
            "questionnaire_app/questionnaire_form.html",
            {
                "error": (
                    "The questionnaire could not be processed. "
                    "Please try again."
                ),
                'lang': request.LANG
            }
        )