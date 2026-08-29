import logging
import time
from .models import FusionResult

logger = logging.getLogger(__name__)

# Module weights
GAIT_WEIGHT = 0.45
SPEECH_WEIGHT = 0.35
QUESTIONNAIRE_WEIGHT = 0.20


def calculate_questionnaire_risk(questionnaire_data):
    """
    Calculate a more accurate questionnaire risk score
    by combining ML score with rule-based factors
    """
    # Get all available data
    q_ml_score = questionnaire_data.get('pd_probability', 0.5)
    q_prediction = questionnaire_data.get('prediction', 'Healthy')
    q_motor_score = questionnaire_data.get('motor_score', 0)
    q_motor_pattern = questionnaire_data.get('motor_pattern', 'Low')
    q_support_level = questionnaire_data.get('support_level', 'Low')
    q_reliability = questionnaire_data.get('reliability', 'Low')
    q_uncertain_answers = questionnaire_data.get('uncertain_answers', 0)
    q_internal_zone = questionnaire_data.get('internal_model_zone', 'Low')
    
    # 1. Rule-based risk based on final prediction
    if q_prediction == 'Elevated Parkinsonian Risk':
        rule_risk = 0.85
    elif q_prediction == 'Borderline':
        rule_risk = 0.50
    else:  # Healthy
        rule_risk = 0.15
    
    # 2. Motor score contribution (0-1 scale)
    motor_contribution = q_motor_score  # Already 0-1
    
    # 3. Support level contribution
    if q_support_level == 'Strong':
        support_contribution = 0.8
    elif q_support_level == 'Moderate':
        support_contribution = 0.5
    else:
        support_contribution = 0.2
    
    # 4. Reliability adjustment
    if q_reliability == 'High':
        reliability_factor = 1.0
    elif q_reliability == 'Moderate':
        reliability_factor = 0.8
    else:
        reliability_factor = 0.5
    
    # 5. Uncertain answers penalty
    uncertainty_penalty = min(q_uncertain_answers * 0.05, 0.3)  # Max 30% reduction
    
    # 6. Combined risk calculation
    # Weighted combination of ML score, rule-based score, and clinical factors
    base_risk = (0.35 * q_ml_score) + (0.25 * rule_risk) + (0.25 * motor_contribution) + (0.15 * support_contribution)
    
    # Apply reliability factor
    adjusted_risk = base_risk * reliability_factor
    
    # Apply uncertainty penalty
    final_risk = max(0, min(1, adjusted_risk - uncertainty_penalty))
    
    # Debug print
    print(f"🔍 QUESTIONNAIRE CALCULATION:")
    print(f"   ML Score: {q_ml_score:.4f}")
    print(f"   Prediction: {q_prediction} → Rule Risk: {rule_risk:.4f}")
    print(f"   Motor Score: {motor_contribution:.4f}")
    print(f"   Support Level: {q_support_level} → {support_contribution:.4f}")
    print(f"   Reliability: {q_reliability} → Factor: {reliability_factor:.2f}")
    print(f"   Uncertain Answers: {q_uncertain_answers} → Penalty: {uncertainty_penalty:.4f}")
    print(f"   Base Risk: {base_risk:.4f}")
    print(f"   Adjusted Risk: {adjusted_risk:.4f}")
    print(f"   FINAL RISK: {final_risk:.4f}")
    print("="*50)
    
    return final_risk


def get_prediction_from_risk(risk_score):
    """
    Get standardized prediction label based on risk score
    """
    if risk_score < 0.3:
        return "Control"
    elif risk_score < 0.6:
        return "Borderline"
    else:
        return "PD"


def map_speech_label(label):
    """
    Map speech module labels to standardized labels
    """
    if label == 'Healthy':
        return 'Control'
    elif label == 'Parkinson':
        return 'PD'
    else:  # Borderline or any other
        return label


def map_questionnaire_label(label):
    """
    Map questionnaire module labels to standardized labels
    """
    if label == 'Elevated Parkinsonian Risk':
        return 'Borderline'
    elif label == 'Healthy':
        return 'Control'
    else:  # Borderline or any other
        return label


def calculate_fusion(gait_data, speech_data, questionnaire_data, user=None):
    """
    Calculate fusion results from all three modules
    
    Args:
        gait_data: dict with 'prediction', 'pd_probability', 'control_probability'
        speech_data: dict with 'prediction', 'pd_probability', 'control_probability'
        questionnaire_data: dict with 'prediction', 'pd_probability', 'control_probability'
        user: User object (optional)
    
    Returns:
        dict: Fusion results
    """
    
    # Extract screening_id
    screening_id = gait_data.get('screening_id')
    if not screening_id:
        screening_id = f"fusion_{int(time.time())}"
    
    # Get PD probabilities
    gait_pd = gait_data.get('pd_probability', 0.5)
    speech_pd = speech_data.get('pd_probability', 0.5)
    
    # === IMPROVED: Calculate questionnaire risk using ALL data ===
    questionnaire_risk = calculate_questionnaire_risk(questionnaire_data)
    
    # DEBUG: Print what's being passed to fusion
    print(f"📊 FUSION INPUT:")
    print(f"   Gait PD: {gait_pd:.4f} (Weight: {GAIT_WEIGHT})")
    print(f"   Speech PD: {speech_pd:.4f} (Weight: {SPEECH_WEIGHT})")
    print(f"   Questionnaire Risk: {questionnaire_risk:.4f} (Weight: {QUESTIONNAIRE_WEIGHT})")
    print("="*50)
    
    # Get Control probabilities
    gait_control = gait_data.get('control_probability', 1 - gait_pd)
    speech_control = speech_data.get('control_probability', 1 - speech_pd)
    questionnaire_control = 1 - questionnaire_risk
    
    # ===== STANDARDIZE LABELS =====
    # Gait labels are already standardized (Control / PD)
    gait_prediction = gait_data.get('prediction')
    
    # Map speech labels to standardized format
    speech_prediction = map_speech_label(speech_data.get('prediction'))
    
    # Map questionnaire labels to standardized format
    questionnaire_prediction = get_prediction_from_risk(questionnaire_risk)
    
    # Calculate weighted final probability using improved questionnaire risk
    final_pd_probability = (
        (gait_pd * GAIT_WEIGHT) +
        (speech_pd * SPEECH_WEIGHT) +
        (questionnaire_risk * QUESTIONNAIRE_WEIGHT)
    )
    
    final_control_probability = 1 - final_pd_probability
    
    # Final prediction
    final_prediction = "PD" if final_pd_probability > 0.5 else "Control"
    
    # Confidence (max probability)
    fusion_confidence = max(final_pd_probability, final_control_probability)
    
    # Risk level
    if final_pd_probability < 0.3:
        risk_level = "Low"
    elif final_pd_probability < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    # Prepare result with all data
    result = {
        'screening_id': screening_id,
        'final_prediction': final_prediction,
        'final_pd_probability': final_pd_probability,
        'final_control_probability': final_control_probability,
        'fusion_confidence': fusion_confidence,
        'risk_level': risk_level,
        'modules': {
            'gait': {
                'prediction': gait_prediction,
                'pd_probability': gait_pd,
                'control_probability': gait_control,
                'weight': GAIT_WEIGHT
            },
            'speech': {
                'prediction': speech_prediction,
                'pd_probability': speech_pd,
                'control_probability': speech_control,
                'weight': SPEECH_WEIGHT
            },
            'questionnaire': {
                'prediction': questionnaire_prediction,
                'pd_probability': questionnaire_risk,
                'control_probability': questionnaire_control,
                'weight': QUESTIONNAIRE_WEIGHT,
                # Additional data for reference
                'motor_score': questionnaire_data.get('motor_score'),
                'motor_pattern': questionnaire_data.get('motor_pattern'),
                'support_level': questionnaire_data.get('support_level'),
                'reliability': questionnaire_data.get('reliability'),
                'internal_model_score': questionnaire_data.get('pd_probability')
            }
        }
    }
    
    # DEBUG: Print final result
    print(f"🎯 FINAL FUSION RESULT:")
    print(f"   Final PD Probability: {final_pd_probability:.4f}")
    print(f"   Final Prediction: {final_prediction}")
    print(f"   Risk Level: {risk_level}")
    print(f"   Questionnaire Risk used: {questionnaire_risk:.4f}")
    print("="*50)
    
    return result


def save_fusion_result(result, user=None):
    """Save fusion result to database"""
    try:
        # Use get_or_create to avoid UNIQUE constraint violation
        fusion, created = FusionResult.objects.get_or_create(
            screening_id=result['screening_id'],
            defaults={
                'user': user,
                'gait_prediction': result['modules']['gait']['prediction'],
                'gait_pd_probability': result['modules']['gait']['pd_probability'],
                'gait_control_probability': result['modules']['gait']['control_probability'],
                'speech_prediction': result['modules']['speech']['prediction'],
                'speech_pd_probability': result['modules']['speech']['pd_probability'],
                'speech_control_probability': result['modules']['speech']['control_probability'],
                'questionnaire_prediction': result['modules']['questionnaire']['prediction'],
                'questionnaire_pd_probability': result['modules']['questionnaire']['pd_probability'],
                'questionnaire_control_probability': result['modules']['questionnaire']['control_probability'],
                'final_prediction': result['final_prediction'],
                'final_pd_probability': result['final_pd_probability'],
                'final_control_probability': result['final_control_probability'],
                'fusion_confidence': result['fusion_confidence'],
                'risk_level': result['risk_level'],
                'gait_weight': GAIT_WEIGHT,
                'speech_weight': SPEECH_WEIGHT,
                'questionnaire_weight': QUESTIONNAIRE_WEIGHT
            }
        )
        
        if created:
            logger.info(f"✅ Fusion result saved for screening: {result['screening_id']}")
        else:
            logger.info(f"🔄 Fusion result already exists for screening: {result['screening_id']}")
            
        return fusion
        
    except Exception as e:
        logger.error(f"❌ Failed to save fusion result: {str(e)}")
        return None