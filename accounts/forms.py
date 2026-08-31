from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.utils.translation import gettext_lazy as _
from .models import UserProfile
import re


class RegistrationForm(UserCreationForm):
    """
    Custom registration form with additional fields for user profile.
    """
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'Enter your email address',
            'autocomplete': 'email'
        })
    )
    
    first_name = forms.CharField(
        required=True,
        max_length=30,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Enter your first name',
            'autocomplete': 'given-name'
        })
    )
    
    last_name = forms.CharField(
        required=True,
        max_length=30,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Enter your last name',
            'autocomplete': 'family-name'
        })
    )
    
    phone_number = forms.CharField(
        required=False,
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Enter your phone number (optional)',
            'autocomplete': 'tel'
        })
    )
    
    date_of_birth = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'class': 'form-input',
            'type': 'date',
            'placeholder': 'YYYY-MM-DD'
        }),
        help_text="Optional: Enter your date of birth"
    )
    
    gender = forms.ChoiceField(
        required=False,
        choices=[
            ('', 'Select your gender (optional)'),
            ('M', 'Male'),
            ('F', 'Female'),
            ('O', 'Other'),
            ('P', 'Prefer not to say'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-input'
        })
    )
    
    preferred_language = forms.ChoiceField(
        required=True,
        initial='en',
        choices=[
            ('en', 'English'),
            ('ur', 'Urdu'),
        ],
        widget=forms.Select(attrs={
            'class': 'form-input'
        })
    )
    
    data_consent = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-checkbox',
            'id': 'data_consent'
        }),
        error_messages={
            'required': 'You must agree to the data consent to continue.'
        }
    )
    
    class Meta:
        model = User
        fields = [
            'username',
            'first_name',
            'last_name',
            'email',
            'password1',
            'password2',
            'phone_number',
            'date_of_birth',
            'gender',
            'preferred_language',
            'data_consent',
        ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom attributes to default fields
        self.fields['username'].widget.attrs.update({
            'class': 'form-input',
            'placeholder': 'Choose a username',
            'autocomplete': 'username'
        })
        
        self.fields['password1'].widget.attrs.update({
            'class': 'form-input',
            'placeholder': 'Enter a password',
            'autocomplete': 'new-password'
        })
        
        self.fields['password2'].widget.attrs.update({
            'class': 'form-input',
            'placeholder': 'Confirm your password',
            'autocomplete': 'new-password'
        })
    
    def clean_email(self):
        """Validate that email is unique"""
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError(
                'A user with this email already exists.'
            )
        return email
    
    def clean_username(self):
        """Validate username (no special characters except underscore)"""
        username = self.cleaned_data.get('username')
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            raise forms.ValidationError(
                'Username can only contain letters, numbers, and underscores.'
            )
        return username
    
    def clean_date_of_birth(self):
        """Validate date of birth (user must be at least 18 years old)"""
        from datetime import date
        dob = self.cleaned_data.get('date_of_birth')
        if dob:
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            if age < 18:
                raise forms.ValidationError(
                    'You must be at least 18 years old to register.'
                )
        return dob
    
    def save(self, commit=True):
        """Save the user and create/update the profile"""
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        
        if commit:
            user.save()
            # Create or update profile
            profile = user.profile
            profile.phone_number = self.cleaned_data.get('phone_number')
            profile.date_of_birth = self.cleaned_data.get('date_of_birth')
            profile.gender = self.cleaned_data.get('gender')
            profile.preferred_language = self.cleaned_data.get('preferred_language')
            profile.data_consent = self.cleaned_data.get('data_consent')
            
            if profile.data_consent:
                from django.utils import timezone
                profile.consent_date = timezone.now()
            
            profile.save()
        
        return user


class LoginForm(AuthenticationForm):
    """
    Custom login form with styled fields.
    """
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Username or Email',
            'autocomplete': 'username'
        })
    )
    
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-input',
            'placeholder': 'Enter your password',
            'autocomplete': 'current-password'
        })
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove the default error messages to show custom ones
        self.error_messages['invalid_login'] = (
            'Please enter a correct username and password. '
            'Note that both fields may be case-sensitive.'
        )


class ProfileForm(forms.ModelForm):
    """
    Form for updating user profile information.
    """
    first_name = forms.CharField(
        max_length=30,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'First name',
            'autocomplete': 'given-name'
        })
    )
    
    last_name = forms.CharField(
        max_length=30,
        widget=forms.TextInput(attrs={
            'class': 'form-input',
            'placeholder': 'Last name',
            'autocomplete': 'family-name'
        })
    )
    
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-input',
            'placeholder': 'Email address',
            'autocomplete': 'email'
        })
    )
    
    class Meta:
        model = UserProfile
        fields = [
            'phone_number',
            'date_of_birth',
            'gender',
            'preferred_language',
            'email_notifications',
        ]
        widgets = {
            'date_of_birth': forms.DateInput(attrs={
                'class': 'form-input',
                'type': 'date'
            }),
            'phone_number': forms.TextInput(attrs={
                'class': 'form-input',
                'placeholder': 'Phone number',
                'autocomplete': 'tel'
            }),
            'gender': forms.Select(attrs={
                'class': 'form-input'
            }),
            'preferred_language': forms.Select(attrs={
                'class': 'form-input'
            }),
            'email_notifications': forms.CheckboxInput(attrs={
                'class': 'form-checkbox'
            }),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set initial values from user model
        if self.instance and self.instance.user:
            self.fields['first_name'].initial = self.instance.user.first_name
            self.fields['last_name'].initial = self.instance.user.last_name
            self.fields['email'].initial = self.instance.user.email
    
    def clean_email(self):
        """Validate email is unique (except for current user)"""
        email = self.cleaned_data.get('email')
        if User.objects.exclude(id=self.instance.user.id).filter(email=email).exists():
            raise forms.ValidationError(
                'A user with this email already exists.'
            )
        return email
    
    def save(self, commit=True):
        """Save the profile and update the user"""
        profile = super().save(commit=False)
        
        # Update user fields
        user = profile.user
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.email = self.cleaned_data['email']
        
        if commit:
            user.save()
            profile.save()
        
        return profile


class ConsentForm(forms.ModelForm):
    """
    Form for updating data consent.
    """
    data_consent = forms.BooleanField(
        required=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-checkbox'
        }),
        label="I consent to the collection and processing of my data for screening purposes."
    )
    
    class Meta:
        model = UserProfile
        fields = ['data_consent']
    
    def save(self, commit=True):
        """Save the consent and update the consent date"""
        profile = super().save(commit=False)
        if profile.data_consent:
            from django.utils import timezone
            profile.consent_date = timezone.now()
        else:
            profile.consent_date = None
        
        if commit:
            profile.save()
        
        return profile