from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Min, Max, Count
from django.utils import timezone
from datetime import timedelta
import json
import logging

from .models import ScreeningHistory

logger = logging.getLogger(__name__)


@login_required
def dashboard(request):
    """
    Main dashboard showing screening history and statistics
    """
    lang = request.GET.get('lang', 'en')
    
    # Get all screenings for the user
    screenings = ScreeningHistory.objects.filter(user=request.user)
    total_screenings = screenings.count()
    
    # Calculate statistics
    stats = {
        'total': total_screenings,
        'average_risk': screenings.aggregate(Avg('final_pd_probability'))['final_pd_probability__avg'] or 0,
        'highest_risk': screenings.aggregate(Max('final_pd_probability'))['final_pd_probability__max'] or 0,
        'lowest_risk': screenings.aggregate(Min('final_pd_probability'))['final_pd_probability__min'] or 0,
    }
    
    # Risk level distribution
    risk_distribution = {
        'Low': screenings.filter(risk_level='Low').count(),
        'Medium': screenings.filter(risk_level='Medium').count(),
        'High': screenings.filter(risk_level='High').count(),
    }
    
    # Recent screenings (last 5)
    recent_screenings = screenings[:5]
    
    # Trend data (risk over time)
    trend_data = []
    for screening in screenings.reverse()[:10]:  # Last 10 screenings
        trend_data.append({
            'date': screening.created_at.strftime('%Y-%m-%d'),
            'risk': round(screening.final_pd_probability * 100, 1),
            'level': screening.risk_level,
            'screening_id': screening.screening_id
        })
    
    # Last screening date
    last_screening = screenings.first()
    last_screening_date = last_screening.created_at if last_screening else None
    
    # Get user profile for total screenings
    try:
        profile = request.user.profile
        total_from_profile = profile.total_screenings
    except:
        total_from_profile = total_screenings
    
    context = {
        'lang': lang,
        'screenings': screenings,
        'recent_screenings': recent_screenings,
        'total_screenings': total_screenings,
        'total_from_profile': total_from_profile,
        'stats': stats,
        'risk_distribution': risk_distribution,
        'trend_data': json.dumps(trend_data),
        'last_screening_date': last_screening_date,
        'has_screenings': total_screenings > 0,
    }
    
    return render(request, 'dashboard/dashboard.html', context)


@login_required
def screening_detail(request, screening_id):
    """
    View details of a specific screening
    """
    lang = request.GET.get('lang', 'en')
    
    screening = get_object_or_404(
        ScreeningHistory, 
        screening_id=screening_id, 
        user=request.user
    )
    
    # Get all screenings for comparison
    all_screenings = ScreeningHistory.objects.filter(user=request.user)
    
    context = {
        'lang': lang,
        'screening': screening,
        'all_screenings': all_screenings,
    }
    
    return render(request, 'dashboard/screening_detail.html', context)


@login_required
def compare_screenings(request):
    """
    Compare two screenings side by side
    """
    lang = request.GET.get('lang', 'en')
    screening_ids = request.GET.getlist('ids')
    
    if len(screening_ids) != 2:
        # Show selection page
        screenings = ScreeningHistory.objects.filter(user=request.user)
        return render(request, 'dashboard/compare_select.html', {
            'lang': lang,
            'screenings': screenings,
        })
    
    # Get the two screenings
    screening1 = get_object_or_404(
        ScreeningHistory, 
        screening_id=screening_ids[0], 
        user=request.user
    )
    screening2 = get_object_or_404(
        ScreeningHistory, 
        screening_id=screening_ids[1], 
        user=request.user
    )
    
    # Calculate the difference
    risk_change = screening2.final_pd_probability - screening1.final_pd_probability
    
    # Determine if risk went up, down, or stayed the same
    if risk_change > 0.01:
        risk_trend = 'up'
        risk_change_display = f'+{risk_change:.1f}%'
    elif risk_change < -0.01:
        risk_trend = 'down'
        risk_change_display = f'{risk_change:.1f}%'
    else:
        risk_trend = 'same'
        risk_change_display = '0.0%'
    
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
    """
    API endpoint for chart data
    """
    from django.http import JsonResponse
    
    screenings = ScreeningHistory.objects.filter(user=request.user)
    
    data = {
        'dates': [s.created_at.strftime('%Y-%m-%d') for s in screenings],
        'risks': [round(s.final_pd_probability * 100, 1) for s in screenings],
        'levels': [s.risk_level for s in screenings],
        'ids': [s.screening_id for s in screenings],
    }
    
    return JsonResponse(data, status=200)