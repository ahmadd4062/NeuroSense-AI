"""
Recommendation Synthesizer

Combines ALL analyzers into a single structured output that
captures the full unique intelligence of NeuroSense AI.
"""

from typing import Dict, Any, Optional

from .fusion_analyzer import analyze_fusion, get_fusion_summary
from .pattern_detector import detect_patterns, get_pattern_description, PATTERNS
from .longitudinal_analyzer import analyze_longitudinal, get_longitudinal_summary
from .speech_analyzer import analyze_speech, get_speech_summary
from .questionnaire_analyzer import analyze_questionnaire, get_questionnaire_summary


def synthesize_intelligence(
    user,
    fusion_result: Dict[str, Any],
    questionnaire_result: Optional[Dict[str, Any]] = None,
    questionnaire_result_id: Optional[int] = None,
    screening_id: Optional[str] = None,
    lang: str = 'en'
) -> Dict[str, Any]:
    """
    Run all analyzers and synthesize their outputs into unified intelligence.
    """


    # Fusion Analyzer
    fusion_analysis = analyze_fusion(fusion_result)
    fusion_summary = get_fusion_summary(fusion_analysis, lang)

    # Speech Analyzer
    if not screening_id:
        screening_id = fusion_result.get('screening_id')
    speech_analysis = analyze_speech(screening_id) if screening_id else {}
    speech_summary = get_speech_summary(speech_analysis, lang) if speech_analysis else {}

    # Questionnaire Analyzer
    questionnaire_analysis = analyze_questionnaire(questionnaire_result_id)
    questionnaire_summary = get_questionnaire_summary(questionnaire_analysis, lang)

    current_pd = fusion_result.get('final_pd_probability')

    longitudinal_analysis = analyze_longitudinal(
        current_user=user,
        current_screening_id=screening_id,
        current_pd_probability=current_pd,
    )
    longitudinal_summary = get_longitudinal_summary(longitudinal_analysis, lang)

    # Pattern Detector
    pattern_detection = detect_patterns(
        fusion_result=fusion_result,
        questionnaire_result=questionnaire_result,
        longitudinal_analysis=longitudinal_analysis,
        speech_analysis=speech_analysis,
    )
    pattern_description = get_pattern_description(pattern_detection, lang)


    all_insights = []

    if fusion_analysis.get('discordance_level') == 'high':
        all_insights.append({
            'source': 'fusion',
            'type': 'high_discordance',
            'severity': 'high',
            'en': fusion_summary.get('discordance_text', ''),
            'ur': fusion_summary.get('discordance_text', ''),
        })

    for insight in speech_analysis.get('insights', []):
        insight['source'] = 'speech'
        all_insights.append(insight)

    for insight in questionnaire_analysis.get('insights', []):
        insight['source'] = 'questionnaire'
        all_insights.append(insight)

    for insight in longitudinal_analysis.get('insights', []):
        insight['source'] = 'longitudinal'
        all_insights.append(insight)


    priority_score = pattern_detection.get('priority_score', 5)

    high_severity_count = sum(
        1 for ins in all_insights
        if ins.get('severity') == 'high'
    )

    if high_severity_count >= 2 or priority_score == 1:
        intelligence_priority = 'urgent'
    elif high_severity_count == 1 or priority_score == 2:
        intelligence_priority = 'high'
    elif priority_score == 3:
        intelligence_priority = 'moderate'
    elif priority_score == 4:
        intelligence_priority = 'low'
    else:
        intelligence_priority = 'routine'


    intelligence = {
        'screening_id': screening_id,
        'lang': lang,
        'intelligence_priority': intelligence_priority,

        # Pattern
        'primary_pattern': pattern_detection.get('primary_pattern'),
        'primary_pattern_label': pattern_description.get('label', ''),
        'primary_pattern_description': pattern_description.get('description', ''),
        'detected_patterns': pattern_detection.get('detected_patterns', []),

        # Fusion
        'fusion': {
            'analysis': fusion_analysis,
            'summary': fusion_summary,
            'dominant_module': fusion_analysis.get('dominant_module'),
            'dominant_share_pct': fusion_analysis.get('contribution_shares_pct', {}).get(
                fusion_analysis.get('dominant_module'), 0
            ),
            'discordance_level': fusion_analysis.get('discordance_level'),
            'certainty_level': fusion_analysis.get('certainty_level'),
        },

        # Speech
        'speech': {
            'analysis': speech_analysis,
            'summary': speech_summary,
            'consistency': speech_analysis.get('consistency'),
            'threshold_proximity': speech_analysis.get('threshold_proximity'),
        } if speech_analysis else None,

        # Questionnaire
        'questionnaire': {
            'analysis': questionnaire_analysis,
            'summary': questionnaire_summary,
            'motor_severity': questionnaire_analysis.get('motor_severity'),
            'support_level': questionnaire_analysis.get('secondary_support_level'),
            'reliability': questionnaire_analysis.get('reliability_level'),
        } if questionnaire_analysis.get('has_data') else None,

        # Longitudinal
        'longitudinal': {
            'analysis': longitudinal_analysis,
            'summary': longitudinal_summary,
            'overall_trend': longitudinal_analysis.get('overall_trend'),
            'delta_percentage': longitudinal_analysis.get('delta_percentage'),
            'delta_direction': longitudinal_analysis.get('delta_direction'),
            'delta_text': longitudinal_analysis.get('delta_text'),
            'previous_pd_probability': longitudinal_analysis.get('previous_pd_probability'),
            'previous_screening_date': longitudinal_analysis.get('previous_screening_date'),
        } if longitudinal_analysis.get('has_history') else None,

        # Insights
        'insights': all_insights,

        # Risk
        'risk_summary': {
            'risk_level': fusion_result.get('risk_level', 'Low'),
            'final_pd_probability': fusion_result.get('final_pd_probability', 0.0),
            'fusion_confidence': fusion_result.get('fusion_confidence', 0.0),
            'final_prediction': fusion_result.get('final_prediction', 'Control'),
        },

        # Modules
        'module_predictions': pattern_detection.get('module_summary', {}),
    }

    return intelligence


def build_gemini_prompt(intelligence: Dict[str, Any], lang: str = 'en') -> str:
    """
    Build a structured prompt for Gemini with EXACT numbers and anti-hallucination rules.
    """

    fusion = intelligence.get('fusion', {})
    speech = intelligence.get('speech')
    questionnaire = intelligence.get('questionnaire')
    longitudinal = intelligence.get('longitudinal')
    insights = intelligence.get('insights', [])
    risk = intelligence.get('risk_summary', {})


    insights_text = ""
    for ins in insights:
        text = ins.get(lang, ins.get('en', ''))
        if text:
            insights_text += f"- [{ins.get('source', 'system').upper()}] {text}\n"

    if not insights_text:
        insights_text = "- No specific insights detected."


    if longitudinal:
        delta_text = longitudinal.get('delta_text', 'No comparison available.')
        previous_pd = longitudinal.get('previous_pd_probability')
        current_pd = risk.get('final_pd_probability', 0)
        previous_date = longitudinal.get('previous_screening_date', 'N/A')
        delta_pct = longitudinal.get('delta_percentage')

        if delta_pct is not None:
            longitudinal_line = (
                f"- Previous screening ({previous_date}): PD {previous_pd:.3f}\n"
                f"- Current screening: PD {current_pd:.3f}\n"
                f"- DELTA: {delta_text}"
            )
        else:
            longitudinal_line = f"- {delta_text}"
    else:
        longitudinal_line = "- This is the patient's first screening."

    if lang == 'ur':
        prompt = f"""آپ ایک تجربہ کار نیورولوجسٹ ہیں جو اپنے مریض سے اس کے اسکریننگ نتائج پر بات کر رہے ہیں۔

⚠️ اہم ہدایات (لازمی پڑھیں):
1. آپ کو دیے گئے نمبروں کے علاوہ کوئی نیا نمبر یا فیصد ایجاد نہ کریں
2. اگر کوئی فیصد یہاں نہیں دی گئی تو اس کا ذکر نہ کریں
3. صرف وہی بصیرتیں استعمال کریں جو ذیل میں دی گئی ہیں
4. مریض سے بات کریں جیسے کلینک میں بیٹھے ہیں — گرمجوش اور ہمدرد

📊 مریض کا ڈیٹا:

**رسک:**
- مجموعی رسک: {risk.get('risk_level', 'Unknown')}
- پی ڈی رسک اسکور: {risk.get('final_pd_probability', 0):.3f}
- اعتماد: {risk.get('fusion_confidence', 0):.1%}

**پیٹرن:** {intelligence.get('primary_pattern_label', '')}
{intelligence.get('primary_pattern_description', '')}

**فیوژن:**
- غالب ماڈیول: {fusion.get('dominant_module', 'N/A')} ({fusion.get('dominant_share_pct', 0):.1f}%)
- اختلاف: {fusion.get('discordance_level', 'N/A')}
- یقین: {fusion.get('certainty_level', 'N/A')}

**آواز:** مستقل مزاجی = {speech.get('consistency', 'N/A') if speech else 'N/A'}, حد سے قربت = {speech.get('threshold_proximity', 'N/A') if speech else 'N/A'}

**سوالنامہ:** موٹر شدت = {questionnaire.get('motor_severity', 'N/A') if questionnaire else 'N/A'}, سپورٹ = {questionnaire.get('support_level', 'N/A') if questionnaire else 'N/A'}, وشوسنییتا = {questionnaire.get('reliability', 'N/A') if questionnaire else 'N/A'}

**طویل مدتی رجحان:**
{longitudinal_line}

**کلیدی بصیرتیں:**
{insights_text}

✍️ مریض سے بات کریں۔ مندرجہ ذیل ڈھانچے میں:

## خلاصہ
(ایک پیراگراف — صرف وہی نمبر استعمال کریں جو اوپر دیے گئے)

## آپ کے نتائج کا مطلب
(ہر بصیرت کو سمجھائیں)

## 1. طبی مشاورت
## 2. فزیکل تھراپی اور ورزش
## 3. اسپیچ اور آواز کی تھراپی
## 4. طرز زندگی اور صحت
## 5. علامات کی نگرانی
## 6. فالو اپ شیڈول

---
*یہ سفارشات نیوروسینس اے آئی کے تجزیے پر مبنی ہیں۔ کسی بھی طبی فیصلے سے پہلے مستند ڈاکٹر سے مشورہ کریں۔*"""

    else:
        prompt = f"""You are an experienced neurologist talking to your patient about their screening results.

⚠️ CRITICAL RULES (READ CAREFULLY):
1. Do NOT invent any numbers or percentages. Use ONLY the exact numbers provided below.
2. If a percentage is NOT provided in the data, do NOT mention any percentage.
3. Only use the insights listed below — do not add generic medical advice.
4. Write like you're sitting with the patient in a clinic — warm, empathetic, conversational.

📊 PATIENT DATA:

**Risk:**
- Overall Risk: {risk.get('risk_level', 'Unknown')}
- PD Risk Score: {risk.get('final_pd_probability', 0):.3f}
- Confidence: {risk.get('fusion_confidence', 0):.1%}

**Detected Pattern:** {intelligence.get('primary_pattern_label', '')}
{intelligence.get('primary_pattern_description', '')}

**Fusion Analysis:**
- Dominant Module: {fusion.get('dominant_module', 'N/A')} ({fusion.get('dominant_share_pct', 0):.1f}%)
- Discordance: {fusion.get('discordance_level', 'N/A')}
- Certainty: {fusion.get('certainty_level', 'N/A')}

**Speech:** Consistency = {speech.get('consistency', 'N/A') if speech else 'N/A'}, Threshold Proximity = {speech.get('threshold_proximity', 'N/A') if speech else 'N/A'}

**Questionnaire:** Motor Severity = {questionnaire.get('motor_severity', 'N/A') if questionnaire else 'N/A'}, Support = {questionnaire.get('support_level', 'N/A') if questionnaire else 'N/A'}, Reliability = {questionnaire.get('reliability', 'N/A') if questionnaire else 'N/A'}

**Longitudinal (Compare with previous screening):**
{longitudinal_line}

**Key Insights:**
{insights_text}

✍️ Now write a conversation with the patient. Use this exact structure:

## Your Summary
(One paragraph — use ONLY the exact numbers from above)

## What Your Results Mean
(Walk through each insight. When mentioning the delta from previous screening, use the EXACT number.)

## 1. Medical Consultation
## 2. Physical Therapy & Exercise
## 3. Speech & Voice Therapy
## 4. Lifestyle & Wellness
## 5. Symptom Monitoring
## 6. Follow-up Schedule

---
*These recommendations are based on your NeuroSense AI screening. Consult a qualified healthcare provider before making any medical decisions.*"""

    return prompt