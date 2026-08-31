from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """
    Extended user profile model to store additional user information
    and screening preferences.
    """
    # Link to Django's built-in User model (One-to-One relationship)
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'
    )
    
    # Personal Information
    date_of_birth = models.DateField(
        null=True,
        blank=True,
        help_text="User's date of birth"
    )
    
    gender = models.CharField(
        max_length=10,
        choices=[
            ('M', 'Male'),
            ('F', 'Female'),
            ('O', 'Other'),
            ('P', 'Prefer not to say'),
        ],
        default='P',
        help_text="User's gender"
    )
    
    phone_number = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        help_text="Contact phone number"
    )
    
    # Language Preference (for the centralized language toggle)
    preferred_language = models.CharField(
        max_length=2,
        choices=[
            ('en', 'English'),
            ('ur', 'Urdu'),
        ],
        default='en',
        help_text="User's preferred language"
    )
    
    # Screening History Summary
    total_screenings = models.PositiveIntegerField(
        default=0,
        help_text="Total number of screenings completed"
    )
    
    last_screening_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Date of the last screening"
    )
    
    # Account Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When the profile was created"
    )
    
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="When the profile was last updated"
    )
    
    # Privacy Settings
    data_consent = models.BooleanField(
        default=False,
        help_text="Whether the user has consented to data collection"
    )
    
    consent_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the user provided consent"
    )
    
    # Email notifications
    email_notifications = models.BooleanField(
        default=True,
        help_text="Whether to send email notifications"
    )
    
    # Metadata
    is_active = models.BooleanField(
        default=True,
        help_text="Whether the profile is active"
    )
    
    class Meta:
        db_table = 'user_profiles'
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username}'s Profile"
    
    @property
    def full_name(self):
        """Get the user's full name"""
        return self.user.get_full_name() or self.user.username
    
    @property
    def email(self):
        """Get the user's email"""
        return self.user.email
    
    def increment_screenings(self):
        """Increment the total screenings count"""
        self.total_screenings += 1
        self.save()
    
    def update_last_screening(self):
        """Update the last screening date to now"""
        from django.utils import timezone
        self.last_screening_date = timezone.now()
        self.save()
    
    def save(self, *args, **kwargs):
        """Override save to handle date formatting"""
        # Ensure the profile is always active
        if self.is_active is None:
            self.is_active = True
        super().save(*args, **kwargs)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Automatically create a UserProfile when a new User is created.
    This signal ensures every user has a profile.
    """
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """
    Automatically save the UserProfile when the User is saved.
    """
    try:
        instance.profile.save()
    except UserProfile.DoesNotExist:
        # If profile doesn't exist, create it
        UserProfile.objects.create(user=instance)