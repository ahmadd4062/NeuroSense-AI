"""
Pattern Detector

Analyzes the cross-module relationships to identify specific
clinical patterns that unlock personalized recommendations.

Patterns are UNIQUE to NeuroSense AI because they combine:
- Module probabilities (from YOUR models)
- Module predictions (labels)
- Fusion weights (YOUR 45/35/20)
- Questionnaire sub-signals (motor_score, support_level, etc.)
- Speech per-recording data
- Longitudinal trends

Each pattern has specific clinical meaning and specific recommendations.

Data Sources:
- fusion_result (session)
- questionnaire_result (session)
- ScreeningHistory (database - for longitudinal)
- ScreeningPrediction + RecordingPrediction (database - for speech)
"""

from typing import Dict, Any, List, Optional


# ================================================================
# PATTERN DEFINITIONS
# ================================================================
# Each pattern has:
#   - id: internal identifier
#   - label: human-readable label (en/ur)
#   - description: short clinical description (en/ur)
#   - priority: 1 = highest urgency, 5 = lowest
#   - triggers: what combination of signals triggers this pattern
# ================================================================

PATTERNS = {
    'gait_only_pd': {
        'label': {
            'en': 'Motor-Isolated Pattern',
            'ur': 'موٹر-الگ تھلگ پیٹرن'
        },
        'description': {
            'en': 'Gait is the primary concern while speech and cognition remain normal. This isolated motor involvement often points to early-stage Parkinsonism.',
            'ur': 'چال بنیادی تشویش ہے جبکہ آواز اور ادراک معمول پر ہیں۔ یہ الگ تھلگ موٹر شمولیت اکثر ابتدائی مرحلے کی پارکنسنزم کی طرف اشارہ کرتی ہے۔'
        },
        'priority': 2,
    },
    'speech_only_pd': {
        'label': {
            'en': 'Voice-Isolated Pattern',
            'ur': 'آواز-الگ تھلگ پیٹرن'
        },
        'description': {
            'en': 'Speech shows abnormal patterns while gait and cognition remain normal. This isolated vocal involvement warrants a focused speech and neurological evaluation.',
            'ur': 'آواز میں غیر معمولی پیٹرن ظاہر ہوئے ہیں جبکہ چال اور ادراک معمول پر ہیں۔ یہ الگ تھلگ آواز کی شمولیت ایک مرکوز تقریر اور اعصابی جائزے کی ضرورت ہے۔'
        },
        'priority': 3,
    },
    'all_pd': {
        'label': {
            'en': 'Multi-Domain Pattern',
            'ur': 'کثیر الجہتی پیٹرن'
        },
        'description': {
            'en': 'All three screening domains show Parkinsonian indicators. This comprehensive involvement requires prompt multi-disciplinary evaluation.',
            'ur': 'تینوں اسکریننگ ڈومینز پارکنسنین اشارے ظاہر کرتے ہیں۔ یہ جامع شمولیت فوری کثیر الشعبہ جائزے کی ضرورت ہے۔'
        },
        'priority': 1,
    },
    'gait_questionnaire_pd': {
        'label': {
            'en': 'Motor + Non-Motor Pattern',
            'ur': 'موٹر + غیر موٹر پیٹرن'
        },
        'description': {
            'en': 'Gait and questionnaire responses show Parkinsonian indicators, while speech remains normal. This suggests combined motor and non-motor involvement.',
            'ur': 'چال اور سوالنامہ کے جوابات پارکنسنین اشارے ظاہر کرتے ہیں، جبکہ آواز معمول پر ہے۔ یہ مشترکہ موٹر اور غیر موٹر شمولیت کی نشاندہی کرتا ہے۔'
        },
        'priority': 2,
    },
    'speech_questionnaire_pd': {
        'label': {
            'en': 'Non-Motor + Voice Pattern',
            'ur': 'غیر موٹر + آواز پیٹرن'
        },
        'description': {
            'en': 'Speech and questionnaire responses show Parkinsonian indicators, while gait remains normal. This pattern points to voice and non-motor involvement.',
            'ur': 'آواز اور سوالنامہ کے جوابات پارکنسنین اشارے ظاہر کرتے ہیں، جبکہ چال معمول پر ہے۔ یہ پیٹرن آواز اور غیر موٹر شمولیت کی طرف اشارہ کرتا ہے۔'
        },
        'priority': 3,
    },
    'all_borderline': {
        'label': {
            'en': 'Borderline Across All Domains',
            'ur': 'تمام ڈومینز میں سرحدی'
        },
        'description': {
            'en': 'All three domains show borderline results. No single domain shows a clear pattern — this warrants monitoring and follow-up.',
            'ur': 'تینوں ڈومینز سرحدی نتائج دکھاتے ہیں۔ کوئی ایک ڈومین واضح پیٹرن نہیں دکھاتا — یہ نگرانی اور فالو اپ کا متقاضی ہے۔'
        },
        'priority': 4,
    },
    'mixed_pd': {
        'label': {
            'en': 'Mixed-Domain Pattern',
            'ur': 'مخلوط ڈومین پیٹرن'
        },
        'description': {
            'en': 'Two of the three domains show Parkinsonian indicators while the third remains normal. This pattern requires focused evaluation of the affected domains.',
            'ur': 'تین ڈومینز میں سے دو پارکنسنین اشارے ظاہر کرتے ہیں جبکہ تیسرا معمول پر ہے۔ یہ پیٹرن متاثرہ ڈومینز کے مرکوز جائزے کی ضرورت ہے۔'
        },
        'priority': 2,
    },
    'all_control': {
        'label': {
            'en': 'All Domains Normal',
            'ur': 'تمام ڈومینز معمول پر'
        },
        'description': {
            'en': 'All three domains show normal results. Continue routine monitoring and healthy lifestyle practices.',
            'ur': 'تینوں ڈومینز معمول کے نتائج دکھاتے ہیں۔ معمول کی نگرانی اور صحت مند طرز زندگی جاری رکھیں۔'
        },
        'priority': 5,
    },
    'questionnaire_only': {
        'label': {
            'en': 'Symptom-Only Pattern',
            'ur': 'علامات-صرف پیٹرن'
        },
        'description': {
            'en': 'Only the questionnaire shows Parkinsonian indicators while gait and speech remain normal. This may reflect early non-motor symptoms or other conditions.',
            'ur': 'صرف سوالنامہ پارکنسنین اشارے ظاہر کرتا ہے جبکہ چال اور آواز معمول پر ہیں۔ یہ ابتدائی غیر موٹر علامات یا دیگر حالات کی عکاسی کر سکتا ہے۔'
        },
        'priority': 4,
    },
    'low_reliability': {
        'label': {
            'en': 'Low-Reliability Screening',
            'ur': 'کم اعتماد اسکریننگ'
        },
        'description': {
            'en': 'Your responses contain significant uncertainty, reducing the reliability of this screening. A repeat screening is recommended.',
            'ur': 'آپ کے جوابات میں نمایاں غیر یقینی صورتحال ہے، جو اس اسکریننگ کی وشوسنییتا کو کم کرتی ہے۔ دوبارہ اسکریننگ کی سفارش کی جاتی ہے۔'
        },
        'priority': 3,
    },
    'worsening_trend': {
        'label': {
            'en': 'Worsening Trend Detected',
            'ur': 'بگڑتا ہوا رجحان پایا گیا'
        },
        'description': {
            'en': 'Your screening results show a consistent worsening trend compared to previous screenings. This warrants timely medical attention.',
            'ur': 'آپ کے اسکریننگ نتائج پچھلی اسکریننگز کے مقابلے میں مسلسل بگڑتا ہوا رجحان دکھاتے ہیں۔ یہ بروقت طبی توجہ کی ضرورت ہے۔'
        },
        'priority': 1,
    },
}


def detect_patterns(
    fusion_result: Dict[str, Any],
    questionnaire_result: Optional[Dict[str, Any]] = None,
    longitudinal_analysis: Optional[Dict[str, Any]] = None,
    speech_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect all applicable patterns from the screening data.
    
    Args:
        fusion_result: Fusion result from session
        questionnaire_result: Questionnaire result from session
        longitudinal_analysis: Output from longitudinal_analyzer (optional)
        speech_analysis: Output from speech_analyzer (optional)
        
    Returns:
        Dict with detected patterns, primary pattern, and priority score
    """
    
    modules = fusion_result.get('modules', {})
    gait = modules.get('gait', {})
    speech = modules.get('speech', {})
    questionnaire = modules.get('questionnaire', {})
    
    gait_pred = gait.get('prediction', 'Control')
    speech_pred = speech.get('prediction', 'Control')
    questionnaire_pred = questionnaire.get('prediction', 'Control')
    
    gait_pd = gait.get('pd_probability', 0.0)
    speech_pd = speech.get('pd_probability', 0.0)
    questionnaire_pd = questionnaire.get('pd_probability', 0.0)
    
    # ============================================================
    # DETECT PATTERNS
    # ============================================================
    detected = []
    
    # Pattern: All PD
    if gait_pred == 'PD' and speech_pred == 'PD' and questionnaire_pred == 'PD':
        detected.append('all_pd')
    
    # Pattern: Gait only
    elif gait_pred == 'PD' and speech_pred == 'Control' and questionnaire_pred in ['Control', 'Borderline']:
        detected.append('gait_only_pd')
    
    # Pattern: Speech only
    elif speech_pred == 'PD' and gait_pred == 'Control' and questionnaire_pred in ['Control', 'Borderline']:
        detected.append('speech_only_pd')
    
    # Pattern: Gait + Questionnaire
    elif gait_pred == 'PD' and questionnaire_pred in ['PD', 'Borderline'] and speech_pred == 'Control':
        detected.append('gait_questionnaire_pd')
    
    # Pattern: Speech + Questionnaire
    elif speech_pred == 'PD' and questionnaire_pred in ['PD', 'Borderline'] and gait_pred == 'Control':
        detected.append('speech_questionnaire_pd')
    
    # Pattern: Questionnaire only
    elif questionnaire_pred in ['PD', 'Borderline'] and gait_pred == 'Control' and speech_pred == 'Control':
        detected.append('questionnaire_only')
    
    # Pattern: All borderline
    if gait_pred == 'Borderline' and speech_pred == 'Borderline' and questionnaire_pred == 'Borderline':
        detected.append('all_borderline')
    
    # Pattern: Mixed PD (any 2 of 3)
    pd_count = sum([gait_pred == 'PD', speech_pred == 'PD', questionnaire_pred == 'PD'])
    if pd_count == 2 and 'mixed_pd' not in detected:
        # Only add if not already covered by gait_questionnaire or speech_questionnaire
        if not any(p in detected for p in ['gait_questionnaire_pd', 'speech_questionnaire_pd']):
            detected.append('mixed_pd')
    
    # Pattern: All control
    if gait_pred == 'Control' and speech_pred == 'Control' and questionnaire_pred == 'Control':
        detected.append('all_control')
    
    # Pattern: Low reliability
    if questionnaire_result:
        reliability = questionnaire_result.get('reliability', 'High')
        uncertain = questionnaire_result.get('uncertain_answers', 0)
        if reliability == 'Low' or uncertain >= 3:
            detected.append('low_reliability')
    
    # Pattern: Worsening trend (from longitudinal analysis)
    if longitudinal_analysis:
        trend = longitudinal_analysis.get('overall_trend', 'stable')
        if trend in ['worsening', 'significantly_worsening']:
            detected.append('worsening_trend')
    
    # ============================================================
    # DETERMINE PRIMARY PATTERN
    # ============================================================
    # Sort by priority (lower = more urgent)
    if detected:
        sorted_patterns = sorted(
            detected,
            key=lambda p: PATTERNS.get(p, {}).get('priority', 5)
        )
        primary_pattern = sorted_patterns[0]
        priority_score = PATTERNS[primary_pattern]['priority']
    else:
        primary_pattern = None
        priority_score = 5
    
    # ============================================================
    # BUILD OUTPUT
    # ============================================================
    result = {
        'detected_patterns': detected,
        'primary_pattern': primary_pattern,
        'priority_score': priority_score,
        'primary_pattern_info': PATTERNS.get(primary_pattern, {}) if primary_pattern else {},
        'all_pattern_info': {
            p: PATTERNS[p] for p in detected if p in PATTERNS
        },
        'module_summary': {
            'gait': gait_pred,
            'speech': speech_pred,
            'questionnaire': questionnaire_pred,
        },
        'pd_count': pd_count,
        'elevated_count': sum([
            gait_pd > 0.5,
            speech_pd > 0.5,
            questionnaire_pd > 0.5
        ]),
    }
    
    return result


def get_pattern_description(
    pattern_detection: Dict[str, Any],
    lang: str = 'en'
) -> Dict[str, str]:
    """
    Get human-readable description of the primary pattern.
    
    Args:
        pattern_detection: Output from detect_patterns()
        lang: 'en' or 'ur'
        
    Returns:
        Dict with label and description in requested language
    """
    
    primary = pattern_detection.get('primary_pattern')
    if not primary:
        return {
            'label': 'Screening Complete' if lang == 'en' else 'اسکریننگ مکمل',
            'description': '' 
        }
    
    pattern_info = PATTERNS.get(primary, {})
    label_dict = pattern_info.get('label', {})
    desc_dict = pattern_info.get('description', {})
    
    return {
        'label': label_dict.get(lang, label_dict.get('en', '')),
        'description': desc_dict.get(lang, desc_dict.get('en', '')),
        'pattern_id': primary,
        'priority': pattern_info.get('priority', 5),
    }