import google.generativeai as genai
import logging
import re
from django.conf import settings

logger = logging.getLogger(__name__)


def get_gemini_model():
    """Initialize and return Gemini model"""
    try:
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in settings")
        
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-3.5-flash')
    except Exception as e:
        logger.error(f"Failed to initialize Gemini: {str(e)}")
        return None


def generate_ai_recommendations(fusion_result, gait_data, speech_data, questionnaire_data, user=None):
    """
    Generate personalized AI recommendations using Gemini API
    """
    model = get_gemini_model()
    if not model:
        logger.error("Gemini model not available")
        raise Exception("Gemini AI model is not available. Please check your API key.")
    
    # Extract data from fusion_result (nested modules structure)
    modules = fusion_result.get('modules', {})
    
    gait_module = modules.get('gait', {})
    speech_module = modules.get('speech', {})
    questionnaire_module = modules.get('questionnaire', {})
    
    gait_prediction = gait_module.get('prediction', 'N/A')
    gait_risk = gait_module.get('pd_probability', 0)
    
    speech_prediction = speech_module.get('prediction', 'N/A')
    speech_risk = speech_module.get('pd_probability', 0)
    
    questionnaire_prediction = questionnaire_module.get('prediction', 'N/A')
    questionnaire_risk = questionnaire_module.get('pd_probability', 0)
    
    risk_level = fusion_result.get('risk_level', 'Low')
    confidence = fusion_result.get('fusion_confidence', 0)
    
    # Get user info
    user_name = user.get_full_name() if user and user.is_authenticated else "User"
    
    # Build the prompt
    prompt = f"""You are a compassionate clinical AI assistant. Generate detailed, personalized recommendations for a patient based on their screening results.

PATIENT: {user_name}
OVERALL RISK: {risk_level.upper()} (Confidence: {confidence:.1%})
GAIT: {gait_prediction} (Risk: {gait_risk:.2f})
SPEECH: {speech_prediction} (Risk: {speech_risk:.4f})
QUESTIONNAIRE: {questionnaire_prediction} (Risk: {questionnaire_risk:.2f})
MOTOR SCORE: {questionnaire_data.get('motor_score', 0):.2f}
SUPPORT LEVEL: {questionnaire_data.get('support_level', 'Low')}

Please provide COMPREHENSIVE, PERSONALIZED recommendations in this exact format:

## 1. MEDICAL CONSULTATION RECOMMENDATIONS
### Urgency Level: [immediate / within 2 weeks / within 3 months / routine]
### Type of Specialist:
### What to Discuss with the Doctor:
- Point 1
- Point 2
- Point 3

## 2. PHYSICAL THERAPY & EXERCISE RECOMMENDATIONS
### Specific Exercises:
- Exercise 1
- Exercise 2
- Exercise 3
### Frequency and Duration:
### Precautions to Take:

## 3. SPEECH & VOICE THERAPY RECOMMENDATIONS
### Specific Exercises if Needed:
### When to Seek Speech Therapy:

## 4. LIFESTYLE & WELLNESS RECOMMENDATIONS
### Diet and Nutrition Tips:
### Sleep and Stress Management:
### Social and Mental Engagement:

## 5. SYMPTOM MONITORING GUIDANCE
### What Symptoms to Track:
### How to Track Them:
### When to Seek Immediate Medical Attention:

## 6. FOLLOW-UP SCHEDULE
### When to Repeat Screening:
### When to Follow Up with a Specialist:

RULES:
- Be empathetic and reassuring
- Do NOT diagnose or prescribe medications
- Include a disclaimer at the beginning
- Use **bold** for key terms
- Be specific to this patient's results
- Provide practical, actionable advice"""

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 4096,
            }
        )
        
        result = response.text
        logger.info(f"Generated recommendations: {len(result)} characters")
        
        # Check if response is too short
        if not result or len(result) < 100:
            logger.error(f"Response too short: {result}")
            raise Exception("Generated recommendations are too short. Please try again.")
        
        return result
        
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        raise Exception(f"Failed to generate recommendations: {str(e)}")