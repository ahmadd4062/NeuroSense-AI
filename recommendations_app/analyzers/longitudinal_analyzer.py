"""
Longitudinal Analyzer

Analyzes the user's screening history to detect trends over time.
"""

from typing import Dict, Any, List, Optional
from datetime import timedelta


def analyze_longitudinal(
    current_user,
    current_screening_id: str,
    current_pd_probability: Optional[float] = None,
    max_history: int = 10
) -> Dict[str, Any]:
    """
    Analyze the user's screening history for trends.

    Args:
        current_user: The logged-in user
        current_screening_id: The current screening ID (to exclude from history)
        current_pd_probability: The current screening's PD probability (for delta)
        max_history: Maximum number of past screenings to analyze

    Returns:
        Dict with longitudinal analysis
    """

    empty_result = {
        'has_history': False,
        'total_screenings': 0,
        'previous_screenings': [],
        'overall_trend': 'no_history',
        'gait_trend': 'no_history',
        'speech_trend': 'no_history',
        'questionnaire_trend': 'no_history',
        'motor_score_trend': 'no_history',
        'confidence_trend': 'no_history',
        'time_span_days': 0,
        'worsening_rate_per_month': 0.0,
        'insights': [],
        'previous_risk_level': None,
        'previous_pd_probability': None,
        'previous_screening_date': None,
        'current_pd_probability': None,
        'delta_from_previous': None,
        'delta_percentage': None,
        'delta_direction': 'none',
        'delta_text': 'No previous screening available for comparison.',
    }

    if not current_user or not getattr(current_user, 'is_authenticated', False):
        return empty_result

    try:
        from dashboard.models import ScreeningHistory

        # Step 1: Find the exact current screening timestamp.
        # Historical reports must only compare against screenings that happened
        # BEFORE the selected report. Otherwise future reports contaminate the
        # trend and make many historical recommendations look identical.
        current_record = ScreeningHistory.objects.filter(
            user=current_user,
            screening_id=current_screening_id,
        ).first()

        history_qs = ScreeningHistory.objects.filter(user=current_user)

        if current_record is not None:
            history_qs = history_qs.filter(created_at__lt=current_record.created_at)
        else:
            history_qs = history_qs.exclude(screening_id=current_screening_id)

        history = history_qs.order_by('-created_at')[:max_history]

        # Step 2: Convert to list and reverse to get chronological order (oldest → newest)
        history_list = list(history)
        history_list.reverse()

        if len(history_list) == 0:
            return empty_result


        total = len(history_list)
        first = history_list[0]
        last = history_list[-1]
        time_span = (last.created_at - first.created_at).days


        def compute_trend(values: List[float], threshold: float = 0.03) -> str:
            if len(values) < 2:
                return 'stable'

            clean_values = [v for v in values if v is not None]
            if len(clean_values) < 2:
                return 'stable'

            change = clean_values[-1] - clean_values[0]

            if change >= threshold * 2:
                return 'significantly_worsening'
            elif change >= threshold:
                return 'worsening'
            elif change <= -threshold * 2:
                return 'significantly_improving'
            elif change <= -threshold:
                return 'improving'
            else:
                return 'stable'

        risk_values = [h.final_pd_probability for h in history_list]
        gait_values = [h.gait_pd_probability for h in history_list if h.gait_pd_probability is not None]
        speech_values = [h.speech_pd_probability for h in history_list if h.speech_pd_probability is not None]
        questionnaire_values = [h.questionnaire_pd_probability for h in history_list if h.questionnaire_pd_probability is not None]
        motor_values = [h.motor_score for h in history_list if h.motor_score is not None]
        confidence_values = [h.fusion_confidence for h in history_list]


        overall_trend = compute_trend(risk_values)
        gait_trend = compute_trend(gait_values)
        speech_trend = compute_trend(speech_values)
        questionnaire_trend = compute_trend(questionnaire_values)
        motor_score_trend = compute_trend(motor_values)
        confidence_trend = compute_trend(confidence_values)


        previous = history_list[-1]
        previous_pd = previous.final_pd_probability
        previous_risk_level = previous.risk_level
        previous_date = previous.created_at.strftime('%d %b %Y, %H:%M')


        delta_from_previous = None
        delta_percentage = None
        delta_direction = 'none'
        delta_text = 'No comparison available.'

        if current_pd_probability is not None and previous_pd is not None:
            delta_from_previous = current_pd_probability - previous_pd
            delta_percentage = delta_from_previous * 100

            if delta_percentage > 0.5:
                delta_direction = 'worsening'
                delta_text = (
                    f"Your overall risk INCREASED by {abs(delta_percentage):.1f}% "
                    f"since your last screening on {previous_date}."
                )
            elif delta_percentage < -0.5:
                delta_direction = 'improving'
                delta_text = (
                    f"Your overall risk DECREASED by {abs(delta_percentage):.1f}% "
                    f"since your last screening on {previous_date}."
                )
            else:
                delta_direction = 'stable'
                delta_text = (
                    f"Your overall risk remained stable "
                    f"since your last screening on {previous_date}."
                )


        worsening_rate = 0.0
        if time_span > 0 and len(risk_values) >= 2:
            total_change = risk_values[-1] - risk_values[0]
            worsening_rate = (total_change / time_span) * 30


        previous_screenings = []
        for h in history_list[-3:]:
            previous_screenings.append({
                'screening_id': h.screening_id,
                'date': h.created_at.strftime('%d %b %Y'),
                'risk_level': h.risk_level,
                'final_pd_probability': round(h.final_pd_probability, 4),
                'gait': h.gait_prediction,
                'speech': h.speech_prediction,
                'questionnaire': h.questionnaire_prediction,
            })


        insights = []

        # Delta insight (most important)
        if delta_percentage is not None:
            if delta_direction == 'worsening':
                insights.append({
                    'type': 'delta_worsening',
                    'severity': 'high' if delta_percentage > 10 else 'moderate',
                    'en': f"Compared to your last screening, your overall risk increased by {abs(delta_percentage):.1f}%.",
                    'ur': f"آپ کی پچھلی اسکریننگ کے مقابلے میں، آپ کا مجموعی رسک {abs(delta_percentage):.1f}% بڑھ گیا ہے۔"
                })
            elif delta_direction == 'improving':
                insights.append({
                    'type': 'delta_improving',
                    'severity': 'positive',
                    'en': f"Compared to your last screening, your overall risk decreased by {abs(delta_percentage):.1f}%.",
                    'ur': f"آپ کی پچھلی اسکریننگ کے مقابلے میں، آپ کا مجموعی رسک {abs(delta_percentage):.1f}% کم ہوا ہے۔"
                })
            else:
                insights.append({
                    'type': 'delta_stable',
                    'severity': 'info',
                    'en': f"Your overall risk remained stable since your last screening.",
                    'ur': f"آپ کی پچھلی اسکریننگ کے بعد سے آپ کا مجموعی رسک مستحکم رہا ہے۔"
                })

        # Long-term trend
        if overall_trend in ['worsening', 'significantly_worsening']:
            long_change = (risk_values[-1] - risk_values[0]) * 100
            insights.append({
                'type': 'overall_worsening',
                'severity': 'high' if overall_trend == 'significantly_worsening' else 'moderate',
                'en': f"Across your last {total} screenings, your risk has changed by {long_change:+.1f}%.",
                'ur': f"آپ کی آخری {total} اسکریننگز میں، آپ کا رسک {long_change:+.1f}% تبدیل ہوا ہے۔"
            })

        if overall_trend in ['improving', 'significantly_improving']:
            long_change = (risk_values[-1] - risk_values[0]) * 100
            insights.append({
                'type': 'overall_improving',
                'severity': 'positive',
                'en': f"Across your last {total} screenings, your risk has changed by {long_change:+.1f}%.",
                'ur': f"آپ کی آخری {total} اسکریننگز میں، آپ کا رسک {long_change:+.1f}% تبدیل ہوا ہے۔"
            })

        if gait_trend in ['worsening', 'significantly_worsening']:
            insights.append({
                'type': 'gait_worsening',
                'severity': 'moderate',
                'en': "Your gait analysis shows a worsening trend across screenings.",
                'ur': "آپ کے چال کے تجزیے میں اسکریننگز کے دوران بگڑتا ہوا رجحان ظاہر ہوا ہے۔"
            })

        if speech_trend in ['worsening', 'significantly_worsening']:
            insights.append({
                'type': 'speech_worsening',
                'severity': 'moderate',
                'en': "Your speech analysis shows a worsening trend across screenings.",
                'ur': "آپ کے آواز کے تجزیے میں اسکریننگز کے دوران بگڑتا ہوا رجحان ظاہر ہوا ہے۔"
            })

        if motor_score_trend in ['worsening', 'significantly_worsening']:
            insights.append({
                'type': 'motor_worsening',
                'severity': 'moderate',
                'en': "Your motor symptoms score has increased across screenings.",
                'ur': "آپ کے موٹر علامات کے اسکور میں اسکریننگز کے دوران اضافہ ہوا ہے۔"
            })

        result = {
            'has_history': True,
            'total_screenings': total,
            'previous_screenings': previous_screenings,
            'time_span_days': time_span,

            # Trends
            'overall_trend': overall_trend,
            'gait_trend': gait_trend,
            'speech_trend': speech_trend,
            'questionnaire_trend': questionnaire_trend,
            'motor_score_trend': motor_score_trend,
            'confidence_trend': confidence_trend,

            # Rate of change
            'worsening_rate_per_month': round(worsening_rate, 4),

            # Previous screening
            'previous_risk_level': previous_risk_level,
            'previous_pd_probability': round(previous_pd, 4),
            'previous_screening_date': previous_date,

            # EXACT DELTA
            'current_pd_probability': round(current_pd_probability, 4) if current_pd_probability is not None else None,
            'delta_from_previous': round(delta_from_previous, 4) if delta_from_previous is not None else None,
            'delta_percentage': round(delta_percentage, 2) if delta_percentage is not None else None,
            'delta_direction': delta_direction,
            'delta_text': delta_text,

            # Insights
            'insights': insights,
        }

        return result

    except Exception as e:
        print(f"Longitudinal analysis error: {str(e)}")
        return empty_result


def get_longitudinal_summary(
    longitudinal_analysis: Dict[str, Any],
    lang: str = 'en'
) -> Dict[str, Any]:
    """
    Generate human-readable summary from longitudinal analysis.
    """

    if not longitudinal_analysis.get('has_history'):
        return {
            'has_history': False,
            'summary_text': (
                'This is your first screening.' if lang == 'en'
                else 'یہ آپ کی پہلی اسکریننگ ہے۔'
            ),
        }

    total = longitudinal_analysis['total_screenings']
    days = longitudinal_analysis['time_span_days']
    delta_text = longitudinal_analysis.get('delta_text', '')

    return {
        'has_history': True,
        'summary_text': delta_text,
        'total_screenings': total,
        'time_span_days': days,
        'delta_percentage': longitudinal_analysis.get('delta_percentage'),
        'delta_direction': longitudinal_analysis.get('delta_direction'),
    }