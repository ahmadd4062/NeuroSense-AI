from django.urls import path
from . import views

app_name = 'gait_app'

urlpatterns = [
    path('predict/', views.gait_predict, name='predict'),
    # feature-info endpoint REMOVED - no longer needed
    
    # Gait screening with optional language parameter
    path('gait-screening/', views.gait_screening_page, name='gait_screening'),
    path('gait-screening/<str:lang>/', views.gait_screening_page, name='gait_screening_lang'),
    
    # Gait results with optional language parameter
    path('gait-results/', views.gait_results_page, name='gait_results'),
    path('gait-results/<str:lang>/', views.gait_results_page, name='gait_results_lang'),
]