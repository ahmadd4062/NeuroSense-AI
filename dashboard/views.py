from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Min, Max, Count
from django.utils import timezone
from datetime import timedelta
import json
import logging

from .models import ScreeningHistory

logger = logging.getLogger(__name__)


# ================================================================
# HELPER: Compute trend direction
# ================================================================
def _compute_trend(values):
    """Return ('up'|'down'|'stable', change_value) based on last two values."""
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return 'stable', 0.0

    delta = clean[-1] - clean[-2]
    if delta > 0.01:
        return 'up', round(delta * 100, 2)
    elif delta < -0.01:
        return 'down', round(delta * 100, 2)
    return 'stable', 0.0


# ================================================================
# HELPER: Health Score (0-100)
# ================================================================
def _compute_health_score(screenings_list):
    """
    Compute a 0-100 health score from user's screenings.
    Higher = better (lower risk + improving trend + data quality).
    """
    if not screenings_list:
        return 0, 'stable'

    avg_risk = sum(s.final_pd_probability for s in screenings_list) / len(screenings_list)
    risk_score = (1 - avg_risk) * 70

    if len(screenings_list) >= 2:
        recent = screenings_list[0].final_pd_probability
        older = screenings_list[-1].final_pd_probability
        trend_delta = recent - older
        trend_score = max(0, min(20, -trend_delta * 100))
    else:
        trend_score = 10
        trend_delta = 0

    avg_conf = sum(s.fusion_confidence for s in screenings_list) / len(screenings_list)
    confidence_score = avg_conf * 10

    total = round(risk_score + trend_score + confidence_score)

    if trend_delta < -0.02:
        direction = 'improving'
    elif trend_delta > 0.02:
        direction = 'worsening'
    else:
        direction = 'stable'

    return min(100, max(0, total)), direction


# ================================================================
# HELPER: Generate Insights
# ================================================================
def _generate_insights(screenings_list, lang='en', total=0):
    """Generate personalized insights from screening history."""
    insights = []

    if not screenings_list:
        return insights

    latest = screenings_list[0]

    # 1. Trend insight (compare latest vs previous)
    if len(screenings_list) >= 2:
        prev = screenings_list[1]
        delta = (latest.final_pd_probability - prev.final_pd_probability) * 100
        if delta <= -5:
            insights.append({
                'icon': '📉',
                'type': 'good',
                'en': f"Your risk improved by {abs(delta):.2f}% since your last screening.",
                'ur': f"آپ کا رسک پچھلی اسکریننگ سے {abs(delta):.2f}% بہتر ہوا ہے۔",
            })
        elif delta >= 5:
            insights.append({
                'icon': '📈',
                'type': 'warning',
                'en': f"Your risk increased by {delta:.2f}% since your last screening. Consider a follow-up.",
                'ur': f"آپ کا رسک پچھلی اسکریننگ سے {delta:.2f}% بڑھا ہے۔ فالو اپ پر غور کریں۔",
            })
        else:
            insights.append({
                'icon': '➡️',
                'type': 'info',
                'en': f"Your risk remained stable since your last screening.",
                'ur': f"آپ کا رسک پچھلی اسکریننگ سے مستحکم رہا ہے۔",
            })

    # 2. Module insight - which module is dominant (BY WEIGHTED CONTRIBUTION)
    if latest.gait_pd_probability is not None and latest.speech_pd_probability is not None and latest.questionnaire_pd_probability is not None:
        # Apply fusion weights
        gait_contribution = latest.gait_pd_probability * 0.45
        speech_contribution = latest.speech_pd_probability * 0.35
        questionnaire_contribution = latest.questionnaire_pd_probability * 0.20

        contributions = {
            'gait': gait_contribution,
            'speech': speech_contribution,
            'questionnaire': questionnaire_contribution,
        }
        dominant = max(contributions, key=contributions.get)
        dominant_names_en = {'gait': 'gait', 'speech': 'speech', 'questionnaire': 'questionnaire'}
        dominant_names_ur = {'gait': 'چال', 'speech': 'آواز', 'questionnaire': 'سوالنامہ'}
        insights.append({
            'icon': '🎯',
            'type': 'info',
            'en': f"Your {dominant_names_en[dominant]} module is the dominant factor in your latest screening.",
            'ur': f"آپ کی تازہ اسکریننگ میں {dominant_names_ur[dominant]} ماڈیول غالب عنصر ہے۔",
        })

    # 3. Time-based insight
    if latest:
        days_since = (timezone.now() - latest.created_at).days
        if days_since > 90:
            insights.append({
                'icon': '⏰',
                'type': 'warning',
                'en': f"It has been {days_since} days since your last screening. Consider a new one.",
                'ur': f"آپ کی آخری اسکریننگ سے {days_since} دن گزر چکے ہیں۔ نئی اسکریننگ پر غور کریں۔",
            })
        elif days_since == 0:
            insights.append({
                'icon': '✅',
                'type': 'good',
                'en': f"You are actively tracking. Your last screening was today.",
                'ur': f"آپ فعال ٹریکنگ کر رہے ہیں۔ آپ کی آخری اسکریننگ آج تھی۔",
            })
        elif days_since == 1:
            insights.append({
                'icon': '✅',
                'type': 'good',
                'en': f"You are actively tracking. Your last screening was yesterday.",
                'ur': f"آپ فعال ٹریکنگ کر رہے ہیں۔ آپ کی آخری اسکریننگ کل تھی۔",
            })
        else:
            insights.append({
                'icon': '✅',
                'type': 'good',
                'en': f"You are actively tracking with a screening {days_since} days ago.",
                'ur': f"آپ فعال ٹریکنگ کر رہے ہیں، {days_since} دن پہلے اسکریننگ کی۔",
            })

    # 4. Consistency insight (uses real total, not capped 10)
    if total >= 3:
        insights.append({
            'icon': '🏆',
            'type': 'good',
            'en': f"You have completed {total} screenings. Great consistency!",
            'ur': f"آپ نے {total} اسکریننگز مکمل کی ہیں۔ بہت اچھی مستقل مزاجی!",
        })

    return insights


# ================================================================
# MAIN DASHBOARD VIEW
# ================================================================
@login_required
def dashboard(request):
    """Main dashboard showing screening history and statistics."""
    lang = request.GET.get('lang', 'en')

    # ---------------------------------------------
    # FILTERS (search + risk level)
    # ---------------------------------------------
    search_query = request.GET.get('q', '').strip()
    risk_filter = request.GET.get('risk', '').strip()

    all_screenings = ScreeningHistory.objects.filter(user=request.user)

    filtered = all_screenings
    if search_query:
        filtered = filtered.filter(screening_id__icontains=search_query)
    if risk_filter in ['Low', 'Medium', 'High']:
        filtered = filtered.filter(risk_level=risk_filter)

    total_screenings = all_screenings.count()

    # ---------------------------------------------
    # STATS
    # ---------------------------------------------
    stats = {
        'total': total_screenings,
        'average_risk': all_screenings.aggregate(Avg('final_pd_probability'))['final_pd_probability__avg'] or 0,
        'highest_risk': all_screenings.aggregate(Max('final_pd_probability'))['final_pd_probability__max'] or 0,
        'lowest_risk': all_screenings.aggregate(Min('final_pd_probability'))['final_pd_probability__min'] or 0,
    }

    # ---------------------------------------------
    # STAT TRENDS (compare latest two)
    # ---------------------------------------------
    screenings_list = list(all_screenings[:10])  # newest first

    stat_trends = {
        'average': 'stable',
        'average_change': 0.0,
    }
    if len(screenings_list) >= 2:
        latest_risk = screenings_list[0].final_pd_probability
        prev_risk = screenings_list[1].final_pd_probability
        delta = (latest_risk - prev_risk) * 100
        stat_trends['average'] = 'up' if delta > 1 else ('down' if delta < -1 else 'stable')
        stat_trends['average_change'] = round(delta, 2)

    # ---------------------------------------------
    # RISK DISTRIBUTION
    # ---------------------------------------------
    risk_distribution = {
        'Low': all_screenings.filter(risk_level='Low').count(),
        'Medium': all_screenings.filter(risk_level='Medium').count(),
        'High': all_screenings.filter(risk_level='High').count(),
    }

    # ---------------------------------------------
    # RECENT SCREENINGS (filtered, up to 5)
    # ---------------------------------------------
    recent_screenings = list(filtered[:5])

    # ---------------------------------------------
    # TREND DATA (last 10, oldest first)
    # ---------------------------------------------
    recent_10 = list(all_screenings[:10])
    recent_10.reverse()

    trend_data = []
    module_trend_data = {'gait': [], 'speech': [], 'questionnaire': []}

    for s in recent_10:
        trend_data.append({
            'date': s.created_at.strftime('%Y-%m-%d'),
            'risk': round(s.final_pd_probability * 100, 2),
            'level': s.risk_level,
            'screening_id': s.screening_id,
        })

        date_label = s.created_at.strftime('%Y-%m-%d')
        module_trend_data['gait'].append({
            'date': date_label,
            'value': round((s.gait_pd_probability or 0) * 100, 2),
        })
        module_trend_data['speech'].append({
            'date': date_label,
            'value': round((s.speech_pd_probability or 0) * 100, 2),
        })
        module_trend_data['questionnaire'].append({
            'date': date_label,
            'value': round((s.questionnaire_pd_probability or 0) * 100, 2),
        })

    # ---------------------------------------------
    # LAST SCREENING
    # ---------------------------------------------
    last_screening = all_screenings.first()
    last_screening_date = last_screening.created_at if last_screening else None

    # ---------------------------------------------
    # USER PROFILE
    # ---------------------------------------------
    try:
        profile = request.user.profile
        total_from_profile = profile.total_screenings
    except Exception:
        total_from_profile = total_screenings

    # ---------------------------------------------
    # INSIGHTS (pass real total, not capped 10)
    # ---------------------------------------------
    insights = _generate_insights(
        screenings_list,
        lang=lang,
        total=total_screenings,   # ← real total
    )

    # ---------------------------------------------
    # HEALTH SCORE
    # ---------------------------------------------
    health_score, health_direction = _compute_health_score(screenings_list)

    # ---------------------------------------------
    # CONTEXT
    # ---------------------------------------------
    context = {
        'lang': lang,
        'screenings': all_screenings,
        'recent_screenings': recent_screenings,
        'total_screenings': total_screenings,
        'total_from_profile': total_from_profile,
        'stats': stats,
        'stat_trends': stat_trends,
        'risk_distribution': risk_distribution,
        'trend_data': json.dumps(trend_data),
        'module_trend_data': json.dumps(module_trend_data),
        'last_screening_date': last_screening_date,
        'has_screenings': total_screenings > 0,
        'insights': insights,
        'health_score': health_score,
        'health_direction': health_direction,
        'search_query': search_query,
        'risk_filter': risk_filter,
    }

    return render(request, 'dashboard/dashboard.html', context)


# ================================================================
# OTHER VIEWS (unchanged)
# ================================================================
@login_required
def screening_detail(request, screening_id):
    """View details of a specific screening."""
    lang = request.GET.get('lang', 'en')

    screening = get_object_or_404(
        ScreeningHistory,
        screening_id=screening_id,
        user=request.user,
    )

    all_screenings = ScreeningHistory.objects.filter(user=request.user)

    context = {
        'lang': lang,
        'screening': screening,
        'all_screenings': all_screenings,
    }

    return render(request, 'dashboard/screening_detail.html', context)


@login_required
def compare_screenings(request):
    """Compare two screenings side by side."""
    lang = request.GET.get('lang', 'en')
    screening_ids = request.GET.getlist('ids')

    if len(screening_ids) != 2:
        screenings = ScreeningHistory.objects.filter(user=request.user)
        return render(request, 'dashboard/compare_select.html', {
            'lang': lang,
            'screenings': screenings,
        })

    screening1 = get_object_or_404(
        ScreeningHistory,
        screening_id=screening_ids[0],
        user=request.user,
    )
    screening2 = get_object_or_404(
        ScreeningHistory,
        screening_id=screening_ids[1],
        user=request.user,
    )

    risk_change = screening2.final_pd_probability - screening1.final_pd_probability

    if risk_change > 0.01:
        risk_trend = 'up'
        risk_change_display = f'+{risk_change * 100:.2f}%'
    elif risk_change < -0.01:
        risk_trend = 'down'
        risk_change_display = f'{risk_change * 100:.2f}%'
    else:
        risk_trend = 'same'
        risk_change_display = '0.00%'

    context = {
        'lang': lang,
        'screening1': screening1,
        'screening2': screening2,
        'risk_change': risk_change,
        'risk_change_display': risk_change_display,
        'risk_trend': risk_trend,
    }

    return render(request, 'dashboard/compare.html', context)


@login_required
def progress_chart_data(request):
    """API endpoint for chart data (last 10, oldest to newest)."""
    from django.http import JsonResponse

    screenings = list(ScreeningHistory.objects.filter(user=request.user)[:10])
    screenings.reverse()

    data = {
        'dates': [s.created_at.strftime('%Y-%m-%d') for s in screenings],
        'risks': [round(s.final_pd_probability * 100, 2) for s in screenings],
        'levels': [s.risk_level for s in screenings],
        'ids': [s.screening_id for s in screenings],
    }

    return JsonResponse(data, status=200)