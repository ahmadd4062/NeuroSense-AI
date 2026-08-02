from django.urls import path
from . import views

app_name = 'gait_app'

urlpatterns = [
    path('predict/', views.gait_predict, name='predict'),
    path('gait-screening/', views.gait_screening_page, name='gait_screening'),
    path('gait-results/', views.gait_results_page, name='gait_results'),
]