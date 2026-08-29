from django.urls import path
from . import views

app_name = "chatbot_app"

urlpatterns = [
    path("", views.chatbot_page, name="chatbot"),
    path("api/chat/", views.chat_api, name="chat_api"),
]