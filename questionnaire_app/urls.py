from django.urls import path
from . import views

app_name = "questionnaire_app"

urlpatterns = [
    path(
        "questionnaire/",
        views.questionnaire_home,
        name="home"
    ),
    path(
        "questionnaire/assessment/",
        views.questionnaire_form,
        name="form"
    ),
]