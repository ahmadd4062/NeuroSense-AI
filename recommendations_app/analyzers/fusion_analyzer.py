"""
Fusion Analyzer

Extracts unique intelligence from the fusion result:
- Weighted contribution of each module to final risk
- Dominant module (which drove the risk)
- Discordance score (how much modules disagree)
- Confidence-weighted insights

This data is UNIQUE to NeuroSense AI because it uses the
system's own fusion weights (45/35/20) and thresholds.

Data Source: fusion_result (from session)
"""

from typing import Dict, Any


def analyze_fusion(fusion_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze fusion result and extract unique intelligence.
    
    Args:
        fusion_result: The fusion result dict from session
        
    Returns:
        Dict with unique fusion insights
    """
    

    modules = fusion_result.get('modules', {})
    
    gait = modules.get('gait', {})
    speech = modules.get('speech', {})
    questionnaire = modules.get('questionnaire', {})
    
    gait_pd = gait.get('pd_probability', 0.0)
    speech_pd = speech.get('pd_probability', 0.0)
    questionnaire_pd = questionnaire.get('pd_probability', 0.0)
    
    gait_weight = gait.get('weight', 0.45)
    speech_weight = speech.get('weight', 0.35)
    questionnaire_weight = questionnaire.get('weight', 0.20)
    
    final_pd = fusion_result.get('final_pd_probability', 0.0)
    confidence = fusion_result.get('fusion_confidence', 0.0)
    risk_level = fusion_result.get('risk_level', 'Low')
    

    # How much does each module contribute to the final risk?
    gait_contribution = gait_pd * gait_weight
    speech_contribution = speech_pd * speech_weight
    questionnaire_contribution = questionnaire_pd * questionnaire_weight
    
    total_contribution = gait_contribution + speech_contribution + questionnaire_contribution
    
    # What percentage of the total risk comes from each module?
    if total_contribution > 0:
        gait_share = (gait_contribution / total_contribution) * 100
        speech_share = (speech_contribution / total_contribution) * 100
        questionnaire_share = (questionnaire_contribution / total_contribution) * 100
    else:
        gait_share = speech_share = questionnaire_share = 0.0
    

    contributions = {
        'gait': gait_contribution,
        'speech': speech_contribution,
        'questionnaire': questionnaire_contribution
    }
    
    dominant_module = max(contributions, key=contributions.get)
    dominant_value = contributions[dominant_module]
    
    # Is the dominant module truly dominant or just slightly ahead?
    sorted_contributions = sorted(contributions.values(), reverse=True)
    if sorted_contributions[1] > 0:
        dominance_ratio = sorted_contributions[0] / sorted_contributions[1]
    else:
        dominance_ratio = float('inf') if sorted_contributions[0] > 0 else 0
    

    # How much do the modules disagree with each other?
    probabilities = [gait_pd, speech_pd, questionnaire_pd]
    max_prob = max(probabilities)
    min_prob = min(probabilities)
    discordance = max_prob - min_prob
    
    # Classify discordance level
    if discordance >= 0.5:
        discordance_level = 'high'
    elif discordance >= 0.25:
        discordance_level = 'moderate'
    else:
        discordance_level = 'low'
 
    # Are modules in agreement or disagreement?
    is_gait_elevated = gait_pd > 0.5
    is_speech_elevated = speech_pd > 0.5
    is_questionnaire_elevated = questionnaire_pd > 0.5
    
    elevated_count = sum([is_gait_elevated, is_speech_elevated, is_questionnaire_elevated])
    
    # Compare each pair
    gait_vs_speech_gap = abs(gait_pd - speech_pd)
    gait_vs_questionnaire_gap = abs(gait_pd - questionnaire_pd)
    speech_vs_questionnaire_gap = abs(speech_pd - questionnaire_pd)
    

    # Given the confidence level, how much should we trust the result?
    if confidence >= 0.85:
        certainty_level = 'high'
    elif confidence >= 0.7:
        certainty_level = 'moderate'
    else:
        certainty_level = 'low'
    

    # Classify the overall pattern (for use by pattern_detector)
    if elevated_count == 3:
        risk_pattern = 'all_modules_elevated'
    elif elevated_count == 2:
        risk_pattern = 'mixed_elevated'
    elif elevated_count == 1:
        if is_gait_elevated:
            risk_pattern = 'gait_dominant'
        elif is_speech_elevated:
            risk_pattern = 'speech_dominant'
        else:
            risk_pattern = 'questionnaire_dominant'
    else:
        risk_pattern = 'no_module_elevated'
  
    result = {
        # Raw values
        'module_probabilities': {
            'gait': round(gait_pd, 4),
            'speech': round(speech_pd, 4),
            'questionnaire': round(questionnaire_pd, 4),
        },
        'module_weights': {
            'gait': gait_weight,
            'speech': speech_weight,
            'questionnaire': questionnaire_weight,
        },
        
        # Weighted contributions
        'weighted_contributions': {
            'gait': round(gait_contribution, 4),
            'speech': round(speech_contribution, 4),
            'questionnaire': round(questionnaire_contribution, 4),
        },
        'contribution_shares_pct': {
            'gait': round(gait_share, 1),
            'speech': round(speech_share, 1),
            'questionnaire': round(questionnaire_share, 1),
        },
        
        # Dominant module
        'dominant_module': dominant_module,
        'dominant_value': round(dominant_value, 4),
        'dominance_ratio': round(dominance_ratio, 2) if dominance_ratio != float('inf') else None,
        
        # Discordance
        'discordance_score': round(discordance, 4),
        'discordance_level': discordance_level,
        
        # Module alignment
        'elevated_modules': {
            'gait': is_gait_elevated,
            'speech': is_speech_elevated,
            'questionnaire': is_questionnaire_elevated,
        },
        'elevated_count': elevated_count,
        'module_gaps': {
            'gait_vs_speech': round(gait_vs_speech_gap, 4),
            'gait_vs_questionnaire': round(gait_vs_questionnaire_gap, 4),
            'speech_vs_questionnaire': round(speech_vs_questionnaire_gap, 4),
        },
        
        # Overall
        'final_pd_probability': round(final_pd, 4),
        'fusion_confidence': round(confidence, 4),
        'risk_level': risk_level,
        'certainty_level': certainty_level,
        'risk_pattern': risk_pattern,
    }
    
    return result


def get_fusion_summary(fusion_analysis: Dict[str, Any], lang: str = 'en') -> Dict[str, Any]:
    """
    Generate a human-readable summary from the fusion analysis.
    
    Args:
        fusion_analysis: Output from analyze_fusion()
        lang: 'en' or 'ur'
        
    Returns:
        Dict with summary text in the requested language
    """
    
    dominant = fusion_analysis['dominant_module']
    dominant_share = fusion_analysis['contribution_shares_pct'][dominant]
    discordance_level = fusion_analysis['discordance_level']
    certainty = fusion_analysis['certainty_level']
    elevated_count = fusion_analysis['elevated_count']
    
    # Module names for display
    module_names_en = {
        'gait': 'gait analysis',
        'speech': 'speech analysis',
        'questionnaire': 'questionnaire responses'
    }
    module_names_ur = {
        'gait': 'چال کا تجزیہ',
        'speech': 'آواز کا تجزیہ',
        'questionnaire': 'سوالنامہ کے جوابات'
    }
    
    if lang == 'ur':
        module_names = module_names_ur
        summary = {
            'dominant_text': f"آپ کے مجموعی رسک کا {dominant_share}% {module_names[dominant]} سے آتا ہے",
            'discordance_text': {
                'high': "آپ کے ماڈیولز کے نتائج میں نمایاں فرق ہے — کچھ علاقے متاثر ہیں جبکہ دیگر محفوظ ہیں",
                'moderate': "آپ کے ماڈیولز میں کچھ فرق ہے",
                'low': "آپ کے تمام ماڈیولز میں مستقل نتائج ہیں",
            }[discordance_level],
            'certainty_text': {
                'high': "اسکریننگ کی اعتماد کی سطح زیادہ ہے — ڈیٹا قابل اعتماد ہے",
                'moderate': "اسکریننگ کی اعتماد کی سطح درمیانہ ہے",
                'low': "اسکریننگ کی اعتماد کی سطح کم ہے — دوبارہ اسکریننگ کی ضرورت ہو سکتی ہے",
            }[certainty],
        }
    else:
        module_names = module_names_en
        summary = {
            'dominant_text': f"{dominant_share}% of your overall risk comes from {module_names[dominant]}",
            'discordance_text': {
                'high': "Your modules show significant disagreement — some areas are affected while others are preserved",
                'moderate': "Your modules show moderate disagreement",
                'low': "Your modules show consistent results across all areas",
            }[discordance_level],
            'certainty_text': {
                'high': "Screening confidence is high — the data is reliable",
                'moderate': "Screening confidence is moderate",
                'low': "Screening confidence is low — a repeat screening may be helpful",
            }[certainty],
        }
    
    return summary