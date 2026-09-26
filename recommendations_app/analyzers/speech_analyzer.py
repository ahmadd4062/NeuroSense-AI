"""
Speech Analyzer

Extracts deep intelligence from the speech module's database records:
- Per-recording predictions and probabilities
- Voice consistency across recordings
- Threshold proximity (how close to PD threshold)
- Recording quality indicators
- Speech stability metrics

This is UNIQUE to NeuroSense AI because it uses the per-recording
data that is stored in YOUR database.

Data Source: ScreeningPrediction + RecordingPrediction models (database)
"""

from typing import Dict, Any, List, Optional


def analyze_speech(screening_id: str) -> Dict[str, Any]:
    """
    Analyze speech screening from database records.
    
    Args:
        screening_id: The screening ID to look up
        
    Returns:
        Dict with speech analysis
    """
    
    # Empty result if no data
    empty_result = {
        'has_data': False,
        'screening_id': screening_id,
        'final_score': 0.0,
        'final_label': 'Unknown',
        'successful_recordings': 0,
        'failed_recordings': 0,
        'consistency': 'unknown',
        'consistency_score': 0.0,
        'average_probability': 0.0,
        'probability_spread': 0.0,
        'threshold_proximity': 'unknown',
        'parkinson_threshold': 0.27,
        'healthy_threshold': 0.23,
        'recordings': [],
        'insights': [],
    }
    
    if not screening_id:
        return empty_result
    
    try:
        from speech_app.models import ScreeningPrediction
        
        try:
            prediction = ScreeningPrediction.objects.get(screening_id=screening_id)
        except ScreeningPrediction.DoesNotExist:
            return empty_result
        
        # ============================================================
        # BASIC SCREENING DATA
        # ============================================================
        final_score = prediction.final_score
        final_label = prediction.final_label
        successful = prediction.successful_recordings
        failed = prediction.failed_recordings
        parkinson_threshold = prediction.parkinson_threshold
        healthy_threshold = prediction.healthy_threshold
        
        # ============================================================
        # PER-RECORDING ANALYSIS
        # ============================================================
        recordings_qs = prediction.recording_predictions.all().order_by('slot_number')
        recordings = []
        probabilities = []
        
        for rec in recordings_qs:
            rec_data = {
                'slot': rec.slot_number,
                'decision_score': round(rec.decision_score, 4) if rec.decision_score is not None else None,
                'predicted_label': rec.predicted_label,
                'probability': round(rec.probability, 4) if rec.probability is not None else None,
                'extraction_successful': rec.extraction_successful,
                'prediction_successful': rec.prediction_successful,
                'feature_count': rec.feature_count,
                'error': rec.error if rec.error else None,
            }
            recordings.append(rec_data)
            
            if rec.probability is not None:
                probabilities.append(rec.probability)
        
        # ============================================================
        # CONSISTENCY ANALYSIS
        # ============================================================
        if len(probabilities) >= 2:
            avg_prob = sum(probabilities) / len(probabilities)
            spread = max(probabilities) - min(probabilities)
            
            # Consistency classification
            if spread <= 0.10:
                consistency = 'high'
                consistency_score = 0.9
            elif spread <= 0.20:
                consistency = 'moderate'
                consistency_score = 0.6
            else:
                consistency = 'low'
                consistency_score = 0.3
        else:
            avg_prob = final_score
            spread = 0.0
            consistency = 'unknown'
            consistency_score = 0.0
        
        # ============================================================
        # THRESHOLD PROXIMITY
        # ============================================================
        # How close is the score to the PD threshold?
        distance_to_pd = parkinson_threshold - final_score
        distance_to_healthy = final_score - healthy_threshold
        
        if final_score >= parkinson_threshold:
            threshold_proximity = 'above_parkinson'
        elif final_score >= healthy_threshold:
            threshold_proximity = 'borderline_zone'
        elif final_score >= (healthy_threshold - 0.05):
            threshold_proximity = 'near_healthy_boundary'
        else:
            threshold_proximity = 'healthy_zone'
        
        # ============================================================
        # GENERATE INSIGHTS
        # ============================================================
        insights = []
        
        # Consistency insight
        if consistency == 'high' and len(probabilities) >= 2:
            insights.append({
                'type': 'voice_consistency_high',
                'severity': 'positive',
                'en': f"Your voice showed high consistency across all {len(probabilities)} recordings — this is a reliable result.",
                'ur': f"آپ کی آواز نے تمام {len(probabilities)} ریکارڈنگز میں اعلیٰ مستقل مزاجی ظاہر کی — یہ ایک قابل اعتماد نتیجہ ہے۔"
            })
        elif consistency == 'low' and len(probabilities) >= 2:
            insights.append({
                'type': 'voice_consistency_low',
                'severity': 'moderate',
                'en': f"Your voice varied significantly across recordings (spread: {round(spread, 2)}). A repeat screening may help confirm the result.",
                'ur': f"آپ کی آواز ریکارڈنگز میں نمایاں طور پر مختلف تھی (فرق: {round(spread, 2)})۔ دوبارہ اسکریننگ نتیجے کی تصدیق میں مدد کر سکتی ہے۔"
            })
        
        # Threshold proximity insight
        if threshold_proximity == 'above_parkinson':
            insights.append({
                'type': 'score_above_threshold',
                'severity': 'high',
                'en': f"Your voice score ({round(final_score, 3)}) is above the Parkinson's threshold ({parkinson_threshold}). Speech therapy evaluation is recommended.",
                'ur': f"آپ کا آواز کا اسکور ({round(final_score, 3)}) پارکنسن کی حد ({parkinson_threshold}) سے اوپر ہے۔ اسپیچ تھراپی کے جائزے کی سفارش کی جاتی ہے۔"
            })
        elif threshold_proximity == 'borderline_zone':
            distance = parkinson_threshold - final_score
            insights.append({
                'type': 'score_borderline',
                'severity': 'moderate',
                'en': f"Your voice score ({round(final_score, 3)}) is in the borderline zone — only {round(distance, 3)} below the Parkinson's threshold. Monitor monthly.",
                'ur': f"آپ کا آواز کا اسکور ({round(final_score, 3)}) سرحدی زون میں ہے — پارکنسن کی حد سے صرف {round(distance, 3)} نیچے۔ ماہانہ نگرانی کریں۔"
            })
        elif threshold_proximity == 'near_healthy_boundary':
            insights.append({
                'type': 'score_near_healthy',
                'severity': 'info',
                'en': f"Your voice score is near the healthy boundary. Continue vocal maintenance exercises.",
                'ur': f"آپ کا آواز کا اسکور صحت مند حد کے قریب ہے۔ آواز کی بحالی کی مشقیں جاری رکھیں۔"
            })
        else:
            insights.append({
                'type': 'score_healthy',
                'severity': 'positive',
                'en': f"Your voice score is well within the healthy range. No immediate speech concerns.",
                'ur': f"آپ کا آواز کا اسکور صحت مند حد میں ہے۔ تقریر کے کوئی فوری خدشات نہیں۔"
            })
        
        # Recording quality insight
        if failed > 0:
            insights.append({
                'type': 'recording_issues',
                'severity': 'info',
                'en': f"{failed} of {successful + failed} recordings had technical issues and were excluded from analysis.",
                'ur': f"{successful + failed} میں سے {failed} ریکارڈنگز میں تکنیکی مسائل تھے اور انہیں تجزیے سے خارج کر دیا گیا۔"
            })
        
        # ============================================================
        # BUILD OUTPUT
        # ============================================================
        result = {
            'has_data': True,
            'screening_id': screening_id,
            'final_score': round(final_score, 4),
            'final_label': final_label,
            'successful_recordings': successful,
            'failed_recordings': failed,
            
            # Consistency
            'consistency': consistency,
            'consistency_score': round(consistency_score, 2),
            'average_probability': round(avg_prob, 4),
            'probability_spread': round(spread, 4),
            
            # Thresholds
            'parkinson_threshold': parkinson_threshold,
            'healthy_threshold': healthy_threshold,
            'threshold_proximity': threshold_proximity,
            
            # Per-recording data
            'recordings': recordings,
            
            # Insights
            'insights': insights,
        }
        
        return result
        
    except Exception as e:
        print(f"Speech analysis error: {str(e)}")
        return empty_result


def get_speech_summary(speech_analysis: Dict[str, Any], lang: str = 'en') -> Dict[str, Any]:
    """
    Generate human-readable summary from speech analysis.
    
    Args:
        speech_analysis: Output from analyze_speech()
        lang: 'en' or 'ur'
        
    Returns:
        Dict with summary text
    """
    
    if not speech_analysis.get('has_data'):
        return {
            'has_data': False,
            'summary_text': (
                'Speech analysis data not available.' if lang == 'en'
                else 'آواز کے تجزیے کا ڈیٹا دستیاب نہیں۔'
            ),
        }
    
    label = speech_analysis['final_label']
    score = speech_analysis['final_score']
    consistency = speech_analysis['consistency']
    successful = speech_analysis['successful_recordings']
    
    if lang == 'ur':
        summary = f"آپ کی آواز کا تجزیہ ({successful} ریکارڈنگز): {label} (اسکور: {score})"
        
        consistency_map = {
            'high': ' — اعلیٰ مستقل مزاجی',
            'moderate': ' — درمیانی مستقل مزاجی',
            'low': ' — کم مستقل مزاجی',
            'unknown': '',
        }
        summary += consistency_map.get(consistency, '')
    else:
        summary = f"Your speech analysis ({successful} recordings): {label} (score: {score})"
        
        consistency_map = {
            'high': ' — high consistency',
            'moderate': ' — moderate consistency',
            'low': ' — low consistency',
            'unknown': '',
        }
        summary += consistency_map.get(consistency, '')
    
    return {
        'has_data': True,
        'summary_text': summary,
        'final_label': label,
        'final_score': score,
        'consistency': consistency,
    }