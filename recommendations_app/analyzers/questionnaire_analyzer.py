"""
Questionnaire Analyzer

Extracts deep intelligence from the questionnaire module's sub-signals:
- Motor symptoms breakdown
- Non-motor symptoms (speech, cognition)
- Support level analysis
- Response reliability assessment
- Internal model zone
- Safety flags
- Positive motor symptoms count
- Uncertain answers penalty

This is UNIQUE to NeuroSense AI because it uses the multi-layered
questionnaire logic that is stored in YOUR database.

Data Source: QuestionnaireResult model (database)
"""

from typing import Dict, Any, List, Optional


def analyze_questionnaire(questionnaire_result_id: Optional[int]) -> Dict[str, Any]:
    """
    Analyze questionnaire from database record.
    
    Args:
        questionnaire_result_id: The ID of the QuestionnaireResult
        
    Returns:
        Dict with questionnaire analysis
    """
    
    # Empty result if no data
    empty_result = {
        'has_data': False,
        'final_prediction': 'Unknown',
        'module_confidence': 0.0,
        'confidence_level': 'Low',
        'motor_score': 0.0,
        'motor_pattern': 'Low',
        'positive_motor_symptoms': 0,
        'speech_cognition_score': 0.0,
        'secondary_support_level': 'Unknown',
        'response_reliability': 0.0,
        'reliability_level': 'Low',
        'uncertain_answers': 0,
        'age_in_model_range': False,
        'internal_model_score': 0.0,
        'internal_model_zone': 'Low',
        'requires_fusion_review': False,
        'patient_interpretation': '',
        'safety_flags': [],
        'insights': [],
    }
    
    if not questionnaire_result_id:
        return empty_result
    
    try:
        from questionnaire_app.models import QuestionnaireResult
        
        try:
            result = QuestionnaireResult.objects.get(id=questionnaire_result_id)
        except QuestionnaireResult.DoesNotExist:
            return empty_result
        
        # ============================================================
        # EXTRACT SUB-SIGNALS
        # ============================================================
        motor_score = result.motor_score or 0.0
        motor_pattern = result.motor_pattern or 'Low'
        positive_motor = result.positive_motor_symptoms or 0
        speech_cog = result.speech_cognition_score or 0.0
        support_level = result.secondary_support_level or 'Unknown'
        reliability = result.response_reliability or 0.0
        reliability_level = result.reliability_level or 'Low'
        uncertain = result.uncertain_answers or 0
        age_in_range = result.age_in_model_range
        internal_score = result.internal_model_score or 0.0
        internal_zone = result.internal_model_zone or 'Low'
        safety_flags = result.safety_flags or []
        interpretation = result.patient_interpretation or ''
        final_pred = result.final_prediction
        
        # ============================================================
        # MOTOR SYMPTOM ANALYSIS
        # ============================================================
        if motor_score >= 0.6:
            motor_severity = 'high'
        elif motor_score >= 0.35:
            motor_severity = 'moderate'
        elif motor_score >= 0.15:
            motor_severity = 'mild'
        else:
            motor_severity = 'minimal'
        
        # ============================================================
        # SUPPORT LEVEL ANALYSIS
        # ============================================================
        # Low support = more self-management emphasis
        support_map = {
            'Strong': {'score': 0.9, 'self_care_need': 'low'},
            'Moderate': {'score': 0.6, 'self_care_need': 'moderate'},
            'Low': {'score': 0.2, 'self_care_need': 'high'},
        }
        support_info = support_map.get(support_level, {'score': 0.5, 'self_care_need': 'unknown'})
        
        # ============================================================
        # RELIABILITY ANALYSIS
        # ============================================================
        if reliability_level == 'High':
            reliability_trust = 'high'
        elif reliability_level == 'Moderate':
            reliability_trust = 'moderate'
        else:
            reliability_trust = 'low'
        
        # ============================================================
        # GENERATE INSIGHTS
        # ============================================================
        insights = []
        
        # Motor symptoms insight
        if motor_severity == 'high':
            insights.append({
                'type': 'motor_high',
                'severity': 'high',
                'en': f"Your motor symptoms score ({round(motor_score, 2)}) indicates significant motor involvement with {positive_motor} positive symptoms.",
                'ur': f"آپ کے موٹر علامات کے اسکور ({round(motor_score, 2)}) میں نمایاں موٹر شمولیت کی نشاندہی ہوتی ہے، {positive_motor} مثبت علامات کے ساتھ۔"
            })
        elif motor_severity == 'moderate':
            insights.append({
                'type': 'motor_moderate',
                'severity': 'moderate',
                'en': f"Your motor symptoms score ({round(motor_score, 2)}) indicates mild motor involvement. Monitor for changes.",
                'ur': f"آپ کے موٹر علامات کے اسکور ({round(motor_score, 2)}) میں ہلکی موٹر شمولیت کی نشاندہی ہوتی ہے۔ تبدیلیوں کی نگرانی کریں۔"
            })
        elif motor_severity == 'minimal':
            insights.append({
                'type': 'motor_minimal',
                'severity': 'positive',
                'en': f"Your motor symptoms score ({round(motor_score, 2)}) is minimal — no significant motor concerns.",
                'ur': f"آپ کے موٹر علامات کے اسکور ({round(motor_score, 2)}) کم سے کم ہیں — کوئی اہم موٹر خدشات نہیں۔"
            })
        
        # Speech & cognition insight
        if speech_cog >= 0.5:
            insights.append({
                'type': 'speech_cog_high',
                'severity': 'moderate',
                'en': f"Your speech and cognition score ({round(speech_cog, 2)}) suggests some non-motor symptoms. Cognitive and speech monitoring is recommended.",
                'ur': f"آپ کے تقریر اور ادراک کے اسکور ({round(speech_cog, 2)}) میں کچھ غیر موٹر علامات کی نشاندہی ہوتی ہے۔ علمی اور تقریر کی نگرانی کی سفارش کی جاتی ہے۔"
            })
        
        # Support level insight
        if support_info['self_care_need'] == 'high':
            insights.append({
                'type': 'support_low',
                'severity': 'moderate',
                'en': "Your support level is low — this means you may be handling symptoms independently. Building a care network is strongly recommended.",
                'ur': "آپ کا سپورٹ لیول کم ہے — اس کا مطلب ہے کہ آپ خود مختاری سے علامات کو سنبھال رہے ہیں۔ نگہداشت کا نیٹ ورک بنانے کی سختی سے سفارش کی جاتی ہے۔"
            })
        elif support_info['self_care_need'] == 'low':
            insights.append({
                'type': 'support_strong',
                'severity': 'positive',
                'en': "You have a strong support network — this is a significant protective factor.",
                'ur': "آپ کے پاس مضبوط سپورٹ نیٹ ورک ہے — یہ ایک اہم حفاظتی عنصر ہے۔"
            })
        
        # Reliability insight
        if reliability_level == 'Low':
            insights.append({
                'type': 'reliability_low',
                'severity': 'moderate',
                'en': f"Your responses have low reliability with {uncertain} uncertain answers. Consider a repeat screening with more definite answers.",
                'ur': f"آپ کے جوابات میں کم وشوسنییتا ہے، {uncertain} غیر یقینی جوابات کے ساتھ۔ مزید واضح جوابات کے ساتھ دوبارہ اسکریننگ پر غور کریں۔"
            })
        
        # Uncertain answers insight
        if uncertain >= 3:
            insights.append({
                'type': 'uncertain_high',
                'severity': 'moderate',
                'en': f"You provided {uncertain} \"Not sure\" answers — this reduces the accuracy of the questionnaire module. Please answer more definitively next time.",
                'ur': f"آپ نے {uncertain} \"یقین نہیں\" جوابات دیے — یہ سوالنامہ ماڈیول کی درستگی کو کم کرتا ہے۔ براہ کرم اگلی بار زیادہ واضح جواب دیں۔"
            })
        
        # Age range insight
        if not age_in_range:
            insights.append({
                'type': 'age_out_of_range',
                'severity': 'info',
                'en': "Your age is outside the model's primary training range. Results should be interpreted with caution.",
                'ur': "آپ کی عمر ماڈل کی بنیادی تربیتی حد سے باہر ہے۔ نتائج کو احتیاط کے ساتھ تشریح کیا جانا چاہیے۔"
            })
        
        # Safety flags
        if safety_flags:
            insights.append({
                'type': 'safety_flags',
                'severity': 'high',
                'en': f"Safety concern detected: {', '.join(safety_flags) if isinstance(safety_flags, list) else safety_flags}",
                'ur': f"حفاظتی تشویش کا پتہ چلا: {', '.join(safety_flags) if isinstance(safety_flags, list) else safety_flags}"
            })
        
        # ============================================================
        # BUILD OUTPUT
        # ============================================================
        result_dict = {
            'has_data': True,
            'final_prediction': final_pred,
            'module_confidence': round(result.module_confidence or 0.0, 4),
            'confidence_level': result.confidence_level or 'Low',
            
            # Motor
            'motor_score': round(motor_score, 4),
            'motor_pattern': motor_pattern,
            'motor_severity': motor_severity,
            'positive_motor_symptoms': positive_motor,
            
            # Speech & Cognition
            'speech_cognition_score': round(speech_cog, 4),
            
            # Support
            'secondary_support_level': support_level,
            'support_score': support_info['score'],
            'self_care_need': support_info['self_care_need'],
            
            # Reliability
            'response_reliability': round(reliability, 4),
            'reliability_level': reliability_level,
            'reliability_trust': reliability_trust,
            'uncertain_answers': uncertain,
            
            # Age
            'age_in_model_range': age_in_range,
            
            # Internal model
            'internal_model_score': round(internal_score, 4),
            'internal_model_zone': internal_zone,
            
            # Fusion
            'requires_fusion_review': result.requires_fusion_review,
            'patient_interpretation': interpretation,
            'safety_flags': safety_flags,
            
            # Insights
            'insights': insights,
        }
        
        return result_dict
        
    except Exception as e:
        print(f"Questionnaire analysis error: {str(e)}")
        return empty_result


def get_questionnaire_summary(
    questionnaire_analysis: Dict[str, Any],
    lang: str = 'en'
) -> Dict[str, Any]:
    """
    Generate human-readable summary from questionnaire analysis.
    
    Args:
        questionnaire_analysis: Output from analyze_questionnaire()
        lang: 'en' or 'ur'
        
    Returns:
        Dict with summary text
    """
    
    if not questionnaire_analysis.get('has_data'):
        return {
            'has_data': False,
            'summary_text': (
                'Questionnaire data not available.' if lang == 'en'
                else 'سوالنامہ کا ڈیٹا دستیاب نہیں۔'
            ),
        }
    
    prediction = questionnaire_analysis['final_prediction']
    motor_severity = questionnaire_analysis['motor_severity']
    support = questionnaire_analysis['secondary_support_level']
    reliability = questionnaire_analysis['reliability_level']
    
    if lang == 'ur':
        motor_map = {
            'high': 'زیادہ',
            'moderate': 'درمیانی',
            'mild': 'ہلکی',
            'minimal': 'کم سے کم',
        }
        summary = f"سوالنامہ: {prediction} | موٹر علامات: {motor_map.get(motor_severity, motor_severity)} | سپورٹ: {support} | وشوسنییتا: {reliability}"
    else:
        summary = f"Questionnaire: {prediction} | Motor symptoms: {motor_severity} | Support: {support} | Reliability: {reliability}"
    
    return {
        'has_data': True,
        'summary_text': summary,
        'final_prediction': prediction,
        'motor_severity': motor_severity,
        'support_level': support,
        'reliability_level': reliability,
    }