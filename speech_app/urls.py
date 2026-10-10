# speech_app/urls.py

from django.urls import path

from . import views
from django.shortcuts import redirect


app_name = "speech_app"

urlpatterns = [
    path(
        "screening/",
        views.screening_page,
        name="screening",
    ),
    path(
        "api/screening/analyze/",
        views.analyze_screening,
        name="analyze_screening",
    ),
    path(
        "api/screening/health/",
        views.screening_health,
        name="screening_health",
    ),
    path(
        "module-three/",
        lambda request: redirect('gait_app:gait_screening'),
        name="module_three"),
    path("", views.home, name="home"),
    path("privacy-policy/", views.privacy_policy, name="privacy_policy"),
    path("terms-and-conditions/", views.terms_conditions, name="terms_conditions"),
    path('set-language/', views.set_language, name='set_language'),

]