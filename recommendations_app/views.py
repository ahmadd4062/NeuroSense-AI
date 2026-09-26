from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import logging
import re

from .services import generate_ai_recommendations

logger = logging.getLogger(__name__)


def clean_markdown(text):
    """
    Convert Markdown to HTML for structured display.
    """
    if not text:
        return text

    # Fix bold with spaces after them
    text = re.sub(r'\*\*([A-Z]+)\*\*\s+([a-z])', r'<strong>\1</strong>\2', text)
    text = re.sub(r'\*\*([A-Z]+)\*\*\s+ly', r'<strong>\1</strong>ly', text)

    # Convert ALL bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

    # Convert italic (with negative lookahead to avoid bold markers)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)

    # Convert headings
    text = re.sub(r'^####\s+(.*?)$', r'<h4 class="rec-h4">\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*?)$', r'<h3 class="rec-h3">\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'<h2 class="rec-h2">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'<h1 class="rec-h1">\1</h1>', text, flags=re.MULTILINE)

    # Process line by line for lists
    lines = text.split('\n')
    result = []
    in_list = False
    in_ordered_list = False

    for line in lines:
        line_stripped = line.strip()

        numbered_match = re.match(r'^[\s]*(\d+)\.\s+(.+)$', line)
        bullet_match = re.match(r'^[\s]*[\*\-•]\s+(.+)$', line)

        if numbered_match:
            if not in_ordered_list:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                result.append('<ol class="rec-list rec-ol">')
                in_ordered_list = True
            content = numbered_match.group(2)
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', content)
            result.append(f'<li>{content}</li>')

        elif bullet_match:
            if in_ordered_list:
                result.append('</ol>')
                in_ordered_list = False
            if not in_list:
                result.append('<ul class="rec-list">')
                in_list = True
            content = bullet_match.group(1)
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', content)
            result.append(f'<li>{content}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            if in_ordered_list:
                result.append('</ol>')
                in_ordered_list = False

            if line_stripped == '':
                result.append('')
                continue

            lower_line = line_stripped.lower()

            # Disclaimer block
            if 'disclaimer' in lower_line or 'not a substitute' in lower_line:
                line_stripped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line_stripped)
                result.append(
                    f'<div class="disclaimer-block">'
                    f'<i class="fa-regular fa-circle-exclamation"></i>'
                    f'<div>{line_stripped}</div></div>'
                )
            # Greeting block
            elif re.match(r'^(Hello|Thank you|First,|I want to|Assalam|السلام)', line_stripped, re.IGNORECASE):
                line_stripped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line_stripped)
                result.append(f'<div class="greeting-block">{line_stripped}</div>')
            else:
                line_stripped = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line_stripped)
                line_stripped = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<em>\1</em>', line_stripped)
                result.append(f'<p class="rec-p">{line_stripped}</p>')

    if in_list:
        result.append('</ul>')
    if in_ordered_list:
        result.append('</ol>')

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


@login_required
def recommendations_page(request):
    """
    Render personalized recommendations page.
    Uses system intelligence (analyzers) + Gemini polish.

    Cache is tied to a specific screening_id — new screening = fresh recommendations.
    """
    lang = getattr(request, 'LANG', 'en')

    fusion_result = request.session.get('fusion_result')
    questionnaire_data = request.session.get('questionnaire_result')
    questionnaire_result_id = request.session.get('questionnaire_result_id')

    # Current screening ID (source of truth)
    current_screening_id = (
        request.session.get('screening_id')
        or (fusion_result.get('screening_id') if fusion_result else None)
    )

    if not fusion_result:
        error_msg = (
            'No screening data found. Please complete the full screening first.'
            if lang == 'en' else
            'کوئی اسکریننگ ڈیٹا نہیں ملا۔ براہ کرم پہلے مکمل اسکریننگ مکمل کریں۔'
        )
        return render(request, 'recommendations_app/recommendations.html', {
            'error': error_msg,
            'lang': lang,
        })

    try:
        recommendations = request.session.get('recommendations')
        generating = request.session.get('generating_recommendations', False)

        cached_screening_id = request.session.get('recommendations_screening_id')

        if recommendations and cached_screening_id and cached_screening_id != current_screening_id:
            logger.info(
                f"🔄 Stale recommendations detected "
                f"(cached for {cached_screening_id}, current is {current_screening_id}). "
                f"Clearing cache."
            )
            request.session.pop('recommendations', None)
            request.session.pop('recommendations_lang', None)
            request.session.pop('recommendations_screening_id', None)
            request.session.pop('generating_recommendations', None)
            request.session.modified = True
            recommendations = None
            generating = False

        logger.info(
            f"Recommendations page loaded | "
            f"cached={bool(recommendations)} | "
            f"generating={generating} | "
            f"lang={lang} | "
            f"screening_id={current_screening_id}"
        )


        if not recommendations and not generating:
            logger.info("🔄 Generating new recommendations via intelligence pipeline...")
            request.session['generating_recommendations'] = True
            request.session.modified = True

            user = request.user if request.user.is_authenticated else None

            try:
                raw_recommendations = generate_ai_recommendations(
                    user=user,
                    fusion_result=fusion_result,
                    questionnaire_result=questionnaire_data,
                    questionnaire_result_id=questionnaire_result_id,
                    screening_id=current_screening_id,
                    lang=lang,
                )
                logger.info(f"Recommendations generated ({len(raw_recommendations)} chars)")

            except Exception as gen_error:
                logger.warning(f"Gemini polish failed: {gen_error}. Using fallback.")
                try:
                    from .analyzers.synthesizer import synthesize_intelligence
                    from .services import get_fallback_recommendations

                    intelligence = synthesize_intelligence(
                        user=user,
                        fusion_result=fusion_result,
                        questionnaire_result=questionnaire_data,
                        questionnaire_result_id=questionnaire_result_id,
                        screening_id=current_screening_id,
                        lang=lang,
                    )
                    raw_recommendations = get_fallback_recommendations(intelligence, lang=lang)
                    logger.info(f"Fallback recommendations generated ({len(raw_recommendations)} chars)")
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise


            recommendations = clean_markdown(raw_recommendations)

            request.session['recommendations'] = recommendations
            request.session['recommendations_lang'] = lang
            request.session['recommendations_screening_id'] = current_screening_id  # ← KEY LINE
            request.session['generating_recommendations'] = False
            request.session.modified = True

        cached_lang = request.session.get('recommendations_lang', 'en')
        if recommendations and cached_lang != lang:
            logger.info(f"Language changed ({cached_lang} → {lang}). Regenerating.")
            request.session.pop('recommendations', None)
            request.session.pop('recommendations_lang', None)
            request.session.pop('recommendations_screening_id', None)
            request.session.modified = True
            return redirect('recommendations_app:recommendations')

        if recommendations and '##' in recommendations:
            logger.warning("Markdown symbols found in cached HTML. Re-cleaning.")
            recommendations = clean_markdown(recommendations)
            request.session['recommendations'] = recommendations
            request.session.modified = True

        return render(request, 'recommendations_app/recommendations.html', {
            'recommendations': recommendations,
            'fusion': fusion_result,
            'is_generating': generating,
            'has_recommendations': bool(recommendations),
            'lang': lang,
        })

    except Exception as e:
        logger.exception(f"Error generating recommendations: {str(e)}")
        request.session['generating_recommendations'] = False
        request.session.modified = True

        error_msg = (
            f'Error generating recommendations: {str(e)}'
            if lang == 'en' else
            f'سفارشات تیار کرنے میں خرابی: {str(e)}'
        )
        return render(request, 'recommendations_app/recommendations.html', {
            'error': error_msg,
            'fusion': fusion_result,
            'lang': lang,
        })


def regenerate_recommendations(request):
    """Force regenerate recommendations."""
    if request.method == 'POST':
        request.session.pop('recommendations', None)
        request.session.pop('recommendations_lang', None)
        request.session.pop('recommendations_screening_id', None)
        request.session.pop('generating_recommendations', None)
        request.session.modified = True
        return redirect('recommendations_app:recommendations')

    return JsonResponse({'error': 'Method not allowed'}, status=405)

def save_recommendations(request):
    if request.method == 'POST':
        pass
    return JsonResponse({'error': 'Method not allowed'}, status=405)