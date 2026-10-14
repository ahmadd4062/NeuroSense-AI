import logging

from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.account.adapter import DefaultAccountAdapter
from django.utils import timezone

from .models import UserProfile

logger = logging.getLogger(__name__)


class CustomSocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Custom adapter for Google sign-in.

    Handles:
      - creating/fetching the UserProfile
      - populating first/last name from Google data
      - storing preferred language from session
      - marking data consent

    Every step is wrapped so a failure in one step does not 500 the
    entire login flow, and every failure is logged so we can see what
    actually happened in the Azure log stream.
    """

    def save_user(self, request, sociallogin, form=None):
        # Step 1: create the Django User (allauth's own logic)
        try:
            user = super().save_user(request, sociallogin, form)
        except Exception:
            logger.exception("ALLAUTH save_user FAILED at super().save_user()")
            raise

        # Step 2: populate name from Google
        try:
            extra = sociallogin.account.extra_data or {}
            changed = False
            if not user.first_name and extra.get('given_name'):
                user.first_name = extra['given_name']
                changed = True
            if not user.last_name and extra.get('family_name'):
                user.last_name = extra['family_name']
                changed = True
            if not user.first_name and extra.get('name'):
                parts = extra['name'].split(' ', 1)
                user.first_name = parts[0]
                if len(parts) > 1:
                    user.last_name = parts[1]
                changed = True
            if changed:
                user.save()
        except Exception:
            logger.exception("ALLAUTH save_user FAILED at name population")

        # Step 3: create/fetch the UserProfile
        try:
            profile, _ = UserProfile.objects.get_or_create(user=user)
        except Exception:
            logger.exception("ALLAUTH save_user FAILED at UserProfile.get_or_create")
            return user

        # Step 4: preferred language from session
        try:
            session_lang = request.session.get('language') if hasattr(request, 'session') else None
            if session_lang and hasattr(profile, 'preferred_language'):
                profile.preferred_language = session_lang
        except Exception:
            logger.exception("ALLAUTH save_user FAILED at session language read")

        # Step 5: consent
        try:
            if hasattr(profile, 'data_consent'):
                profile.data_consent = True
            if hasattr(profile, 'consent_date'):
                profile.consent_date = timezone.now()
            profile.save()
        except Exception:
            logger.exception("ALLAUTH save_user FAILED at consent save")

        return user


class CustomAccountAdapter(DefaultAccountAdapter):
    """
    Custom adapter for regular email/password signups.
    """

    def save_user(self, request, user, form, commit=True):
        try:
            user = super().save_user(request, user, form, commit)
        except Exception:
            logger.exception("ALLAUTH CustomAccountAdapter.save_user FAILED at super()")
            raise

        try:
            UserProfile.objects.get_or_create(user=user)
        except Exception:
            logger.exception("ALLAUTH CustomAccountAdapter.save_user FAILED at get_or_create")

        return user