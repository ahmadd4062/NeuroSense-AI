from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'accounts'

urlpatterns = [

    # Registration
    path('register/', views.register, name='register'),
    
    # Login/Logout
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # Profile
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
    
    # Consent Management
    path('profile/consent/', views.manage_consent, name='manage_consent'),
    
    # Delete Account
    path('profile/delete/', views.delete_account, name='delete_account'),  
    # Username and email availability checks
    path('check-username/', views.check_username_availability, name='check_username'),
    path('check-email/', views.check_email_availability, name='check_email'),
]