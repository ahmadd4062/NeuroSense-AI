from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
from django.utils import timezone
from django.urls import reverse

from .forms import RegistrationForm, LoginForm, ProfileForm, ConsentForm
from .models import UserProfile


def register(request):
    """
    User registration view.
    Creates a new user account with profile.
    """
    # If user is already logged in, redirect to home
    if request.user.is_authenticated:
        return redirect('speech_app:home')
    
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            # Save the user (profile is created automatically via signal)
            user = form.save()
            
            # Log the user in immediately after registration
            login(request, user)
            
            # Set language preference in session
            preferred_lang = form.cleaned_data.get('preferred_language', 'en')
            request.session['language'] = preferred_lang
            request.session.modified = True
            
            # Store screening_id from session if exists (to continue screening)
            screening_id = request.session.get('screening_id')
            
            messages.success(
                request,
                f'Welcome {user.get_full_name() or user.username}! '
                'Your account has been created successfully.'
            )
            
            # Redirect to home or continue screening
            if screening_id:
                return redirect('speech_app:screening')
            return redirect('speech_app:home')
        else:
            # Show form errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = RegistrationForm()
    
    return render(request, 'registration/register.html', {
        'form': form,
        'lang': request.LANG,
    })


def login_view(request):
    """
    User login view.
    """
    # If user is already logged in, redirect to home
    if request.user.is_authenticated:
        return redirect('speech_app:home')
    
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            
            if user is not None:
                login(request, user)
                
                # Set language from profile if available
                try:
                    preferred_lang = user.profile.preferred_language
                    request.session['language'] = preferred_lang
                    request.session.modified = True
                except UserProfile.DoesNotExist:
                    pass
                
                messages.success(
                    request,
                    f'Welcome back, {user.get_full_name() or user.username}!'
                )
                
                # Redirect to next parameter or home
                next_url = request.GET.get('next')
                if next_url:
                    return redirect(next_url)
                return redirect('speech_app:home')
        else:
            messages.error(
                request,
                'Invalid username or password. Please try again.'
            )
    else:
        form = LoginForm()
    
    return render(request, 'registration/login.html', {
        'form': form,
        'lang': request.LANG,
    })


def logout_view(request):
    """
    User logout view.
    """
    logout(request)
    messages.info(request, 'You have been logged out successfully.')
    return redirect('speech_app:home')


@login_required
def profile_view(request):
    """
    View user profile.
    """
    profile = request.user.profile
    
    # Get screening history from session or database
    screening_history = []
    screening_id = request.session.get('screening_id')
    
    # Add screening history from session
    if screening_id:
        screening_history.append({
            'screening_id': screening_id,
            'date': request.session.get('screening_date'),
        })
    
    # Get user's screening results from database (if stored)
    # This will depend on your database models
    # For now, we'll show session data
    
    return render(request, 'profile/profile.html', {
        'profile': profile,
        'user': request.user,
        'screening_history': screening_history,
        'lang': request.LANG,
    })


@login_required
def edit_profile(request):
    """
    Edit user profile.
    """
    profile = request.user.profile
    
    if request.method == 'POST':
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your profile has been updated successfully!')
            return redirect('accounts:profile')
    else:
        form = ProfileForm(instance=profile)
    
    return render(request, 'profile/edit_profile.html', {
        'form': form,
        'user': request.user,
        'lang': request.LANG,
    })


@login_required
def manage_consent(request):
    """
    Manage data consent.
    """
    profile = request.user.profile
    
    if request.method == 'POST':
        form = ConsentForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your consent settings have been updated.')
            return redirect('accounts:profile')
    else:
        form = ConsentForm(instance=profile)
    
    return render(request, 'profile/manage_consent.html', {
        'form': form,
        'profile': profile,
        'lang': request.LANG,
    })


@login_required
def delete_account(request):
    """
    Delete user account (with confirmation).
    """
    if request.method == 'POST':
        user = request.user
        # Logout first
        logout(request)
        # Delete the user (profile will be deleted automatically via cascade)
        user.delete()
        messages.info(request, 'Your account has been deleted successfully.')
        return redirect('speech_app:home')
    
    return render(request, 'profile/delete_account.html', {
        'user': request.user,
        'lang': request.LANG,
    })


# ============================================================
# HELPER VIEWS (For AJAX or API calls)
# ============================================================

def check_username_availability(request):
    """
    Check if username is available (for AJAX validation)
    """
    from django.http import JsonResponse
    
    username = request.GET.get('username', '')
    if username:
        exists = User.objects.filter(username=username).exists()
        return JsonResponse({'available': not exists})
    return JsonResponse({'error': 'No username provided'}, status=400)


def check_email_availability(request):
    """
    Check if email is available (for AJAX validation)
    """
    from django.http import JsonResponse
    
    email = request.GET.get('email', '')
    if email:
        exists = User.objects.filter(email=email).exists()
        return JsonResponse({'available': not exists})
    return JsonResponse({'error': 'No email provided'}, status=400)