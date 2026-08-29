from django.shortcuts import render, redirect
from django.http import JsonResponse
import json
import logging
import re

from .services import generate_ai_recommendations

logger = logging.getLogger(__name__)


def clean_markdown(text):
    """
    Convert Markdown to HTML for structured display
    """
    if not text:
        return text
    
    # Fix bold with spaces after them
    text = re.sub(r'\*\*([A-Z]+)\*\*\s+([a-z])', r'<strong>\1</strong>\2', text)
    text = re.sub(r'\*\*([A-Z]+)\*\*\s+ly', r'<strong>\1</strong>ly', text)
    
    # Convert ALL bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert italic
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    
    # Convert headings
    text = re.sub(r'^####\s+(.*?)$', r'<h4 class="rec-h4">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*?)$', r'<h3 class="rec-h3">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'<h2 class="rec-h2">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'<h1 class="rec-h1">\1</h1>', text, flags=re.MULTILINE)
    
    # Process line by line for lists
    lines = text.split('\n')
    result = []
    in_list = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for bullet point
        bullet_match = re.match(r'^[\s]*[\*\-•]\s+(.+)$', line)
        
        if bullet_match:
            if not in_list:
                result.append('<ul class="rec-list">')
                in_list = True
            content = bullet_match.group(1)
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            result.append(f'<li>{content}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            
            if line_stripped == '':
                result.append('')
                continue
            
            lower_line = line_stripped.lower()
            
            # Disclaimer block
            if 'disclaimer' in lower_line or 'not a substitute' in lower_line:
                line_stripped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line_stripped)
                result.append(f'<div class="disclaimer-block"><i class="fa-regular fa-circle-exclamation"></i><div>{line_stripped}</div></div>')
            else:
                line_stripped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line_stripped)
                line_stripped = re.sub(r'\*(.*?)\*', r'<em>\1</em>', line_stripped)
                result.append(f'<p class="rec-p">{line_stripped}</p>')
    
    if in_list:
        result.append('</ul>')
    
    text = '\n'.join(result)
    text = re.sub(r'<p class="rec-p">\s*</p>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Highlight risk levels and urgency
    text = re.sub(r'(Medium Risk)', r'<span class="risk-tag risk-medium">\1</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'(High Risk)', r'<span class="risk-tag risk-high">\1</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'(Low Risk)', r'<span class="risk-tag risk-low">\1</span>', text, flags=re.IGNORECASE)
    
    text = re.sub(r'(Within 2 weeks)', r'<span class="urgency-tag urgency-soon">\1</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'(Immediate)', r'<span class="urgency-tag urgency-urgent">\1</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'(Within 3 months|Routine)', r'<span class="urgency-tag urgency-routine">\1</span>', text, flags=re.IGNORECASE)
    
    return text


def recommendations_page(request):
    """
    Render personalized recommendations page
    """
    # Get data from session
    gait_data = request.session.get('gait_result')
    speech_data = request.session.get('audio_module_result')
    questionnaire_data = request.session.get('questionnaire_result')
    fusion_result = request.session.get('fusion_result')
    
    if not fusion_result:
        return render(request, 'recommendations_app/recommendations.html', {
            'error': 'No screening data found. Please complete the full screening first.'
        })
    
    try:
        recommendations = request.session.get('recommendations')
        generating = request.session.get('generating_recommendations', False)
        
        if not recommendations and not generating:
            request.session['generating_recommendations'] = True
            request.session.modified = True
            
            user = request.user if request.user.is_authenticated else None
            
            # Generate recommendations
            raw_recommendations = generate_ai_recommendations(
                fusion_result, 
                gait_data, 
                speech_data, 
                questionnaire_data, 
                user
            )
            
            # Log for debugging
            logger.info(f"Raw recommendations length: {len(raw_recommendations)}")
            
            # Convert to HTML
            recommendations = clean_markdown(raw_recommendations)
            
            request.session['recommendations'] = recommendations
            request.session['generating_recommendations'] = False
            request.session.modified = True
        
        # Double-check if recommendations need cleaning
        if recommendations and ('**' in recommendations or '##' in recommendations or '###' in recommendations):
            recommendations = clean_markdown(recommendations)
            request.session['recommendations'] = recommendations
            request.session.modified = True
        
        return render(request, 'recommendations_app/recommendations.html', {
            'recommendations': recommendations,
            'fusion': fusion_result,
            'is_generating': generating,
            'has_recommendations': bool(recommendations)
        })
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        request.session['generating_recommendations'] = False
        request.session.modified = True
        
        return render(request, 'recommendations_app/recommendations.html', {
            'error': str(e),
            'fusion': fusion_result
        })


def regenerate_recommendations(request):
    """
    Force regenerate recommendations
    """
    if request.method == 'POST':
        request.session.pop('recommendations', None)
        request.session.pop('generating_recommendations', None)
        request.session.modified = True
        return redirect('recommendations_app:recommendations')
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def save_recommendations(request):
    if request.method == 'POST':
        pass
    return JsonResponse({'error': 'Method not allowed'}, status=405)