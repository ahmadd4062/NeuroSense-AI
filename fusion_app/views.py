from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import json
import logging
from django.contrib.auth.decorators import login_required

from .services import calculate_fusion, save_fusion_result

logger = logging.getLogger(__name__)

@login_required
def report_page(request):
    """
    Render the final fusion report page with language support
    """
    # Get language from request
    lang = request.GET.get('lang', 'en')
    
    # Get data from session
    gait_data = request.session.get('gait_result')
    speech_data = request.session.get('audio_module_result')
    questionnaire_data = request.session.get('questionnaire_result')
    
    # DEBUG: Print actual data
    print("=== GAIT DATA ===")
    print(gait_data)
    print("=== SPEECH DATA ===")
    print(speech_data)
    print("=== QUESTIONNAIRE DATA ===")
    print(questionnaire_data)
    print(f"=== LANGUAGE: {lang} ===")
    
    # Error messages based on language
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
    
    # Check if all data is available
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
        # Calculate fusion
        user = request.user if request.user.is_authenticated else None
        fusion_result = calculate_fusion(gait_data, speech_data, questionnaire_data, user)
        
        # Save to database
        saved_fusion = save_fusion_result(fusion_result, user)
        
        # ==========================================
        # SAVE TO SCREENING HISTORY (PROGRESS TRACKING)
        # ==========================================
        try:
            from dashboard.models import ScreeningHistory
            
            # Check if this screening already exists
            screening_id = fusion_result.get('screening_id')
            if not ScreeningHistory.objects.filter(screening_id=screening_id).exists():
                ScreeningHistory.objects.create(
                    user=user,
                    screening_id=screening_id,
                    risk_level=fusion_result.get('risk_level', 'Low'),
                    final_pd_probability=fusion_result.get('final_pd_probability', 0),
                    fusion_confidence=fusion_result.get('fusion_confidence', 0),
                    gait_prediction=fusion_result.get('modules', {}).get('gait', {}).get('prediction', 'N/A'),
                    speech_prediction=fusion_result.get('modules', {}).get('speech', {}).get('prediction', 'N/A'),
                    questionnaire_prediction=fusion_result.get('modules', {}).get('questionnaire', {}).get('prediction', 'N/A'),
                )
                logger.info(f"✅ Screening history saved for user {user.username}, screening: {screening_id}")
                
                # Update user profile stats
                try:
                    profile = user.profile
                    profile.total_screenings += 1
                    from django.utils import timezone
                    profile.last_screening_date = timezone.now()
                    profile.save()
                    logger.info(f"✅ User profile updated: total_screenings={profile.total_screenings}")
                except Exception as profile_error:
                    logger.warning(f"Failed to update user profile: {str(profile_error)}")
                    
        except ImportError:
            logger.warning("Dashboard app not installed. Skipping screening history save.")
        except Exception as history_error:
            logger.error(f"Failed to save screening history: {str(history_error)}")
        
        # Store in session
        request.session['fusion_result'] = fusion_result
        request.session.modified = True
        
        # Render report with language
        return render(request, 'fusion_app/report.html', {
            'fusion': fusion_result,
            'gait': gait_data,
            'speech': speech_data,
            'questionnaire': questionnaire_data,
            'saved': saved_fusion is not None,
            'lang': lang
        })
        
    except Exception as e:
        logger.error(f"Fusion calculation error: {str(e)}")
        return render(request, 'fusion_app/report.html', {
            'error': f"{error_messages['fusion'][lang]}{str(e)}",
            'lang': lang
        })


@require_GET
def get_fusion_data(request, screening_id):
    """
    API endpoint to get fusion data by screening_id
    """
    try:
        from .models import FusionResult
        fusion = FusionResult.objects.get(screening_id=screening_id)
        
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
            'created_at': fusion.created_at.isoformat()
        }
        return JsonResponse(data, status=200)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=404)