from django.urls import path
from . import views

app_name = 'recommendations_app'

urlpatterns = [
    path('', views.recommendations_page, name='recommendations'),
    path('regenerate/', views.regenerate_recommendations, name='regenerate'),
    path('save/', views.save_recommendations, name='save'),
]