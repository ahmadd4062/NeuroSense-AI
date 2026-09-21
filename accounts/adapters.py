from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.account.adapter import DefaultAccountAdapter
from .models import UserProfile


class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Custom adapter to auto-create UserProfile for Google users
    """
    def save_user(self, request, sociallogin, form=None):
        # Call parent save_user to create the Django User
        user = super().save_user(request, sociallogin, form)
        
        # Create or get the UserProfile
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        # Set preferred language from session (if set)
        if request.session.get('language'):
            profile.preferred_language = request.session['language']
        
        # Get first name and last name from Google
        if sociallogin.account.extra_data:
            extra = sociallogin.account.extra_data
            
            if not user.first_name and extra.get('given_name'):
                user.first_name = extra['given_name']
            
            if not user.last_name and extra.get('family_name'):
                user.last_name = extra['family_name']
            
            # If first/last name not present, use name field
            if not user.first_name and extra.get('name'):
                parts = extra['name'].split(' ', 1)
                user.first_name = parts[0]
                if len(parts) > 1:
                    user.last_name = parts[1]
            
            user.save()
        
        # Mark data consent as granted (Google signup)
        profile.data_consent = True
        from django.utils import timezone
        profile.consent_date = timezone.now()
        profile.save()
        
        return user


class CustomAccountAdapter(DefaultAccountAdapter):
    """
    Custom adapter for regular signups
    """
    def save_user(self, request, user, form, commit=True):
        user = super().save_user(request, user, form, commit)
        
        # Ensure profile exists for all new users
        UserProfile.objects.get_or_create(user=user)
        
        return user