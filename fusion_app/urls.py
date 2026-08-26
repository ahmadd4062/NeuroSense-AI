from django.urls import path
from . import views

app_name = 'fusion_app'

urlpatterns = [
    path('report/', views.report_page, name='report'),
    path('api/fusion/<str:screening_id>/', views.get_fusion_data, name='fusion_data'),
]