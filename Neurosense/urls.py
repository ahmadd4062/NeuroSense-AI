
from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView, TemplateView
from speech_app.views import set_language

urlpatterns = [
    path('admin/', admin.site.urls),

    # Public Parkinson's education page
    path(
        'understanding-parkinsons/',
        TemplateView.as_view(
            template_name='understanding_parkinsons.html'
        ),
        name='understanding_parkinsons'
    ),

    path('', include('speech_app.urls')),
    path('api/gait/', include('gait_app.urls')),
    path('gait/', include('gait_app.urls')),
    path('questionnaire/', include('questionnaire_app.urls')),
    path('chatbot/', include('chatbot_app.urls')),
    path('fusion/', include('fusion_app.urls')),
    path('recommendations/', include('recommendations_app.urls')),
    path('set-language/', set_language, name='set_language'),
    path('accounts/', include('accounts.urls')),
    path('accounts/', include('allauth.urls')),
    path('dashboard/', include('dashboard.urls')),
]
