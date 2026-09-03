from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Main dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Screening details
    path('screening/<str:screening_id>/', views.screening_detail, name='screening_detail'),
    
    # Compare screenings
    path('compare/', views.compare_screenings, name='compare_screenings'),
    
    # API endpoint for chart data
    path('api/chart-data/', views.progress_chart_data, name='chart_data'),
]