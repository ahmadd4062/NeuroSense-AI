"""
NeuroSense AI - Recommendations Services (google-genai)
"""

import logging
from django.conf import settings

from .analyzers.synthesizer import synthesize_intelligence

logger = logging.getLogger(__name__)


# ================================================================
# GEMINI CLIENT
# ================================================================
def get_gemini_client():
    """Initialize and return the new google-genai client."""
    try:
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in settings")

        from google import genai
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        return None


# ================================================================
# FORMATTERS
# ================================================================
def _format_insights_for_prompt(insights, lang):
    """Format insights as a readable list for the prompt."""
    if not insights:
        return "No specific insights detected."

    lines = []
    for ins in insights:
        text = ins.get(lang, ins.get('en', ''))
        source = ins.get('source', 'system').upper()
        if text:
            lines.append(f"- [{source}] {text}")
    return "\n".join(lines) if lines else "No specific insights detected."


def _format_longitudinal_for_prompt(longitudinal, risk, lang):
    """Format the longitudinal delta as EXACT text for Gemini."""
    if not longitudinal:
        return "This is the patient's first screening — no history available."

    previous_pd = longitudinal.get('previous_pd_probability')
    previous_date = longitudinal.get('previous_screening_date')
    delta_pct = longitudinal.get('delta_percentage')
    delta_dir = longitudinal.get('delta_direction')
    current_pd = risk.get('final_pd_probability', 0)

    if previous_pd is None:
        return "This is the patient's first screening — no history available."

    if lang == 'ur':
        lines = [
            f"- پچھلی اسکریننگ ({previous_date}): پی ڈی اسکور {previous_pd:.3f}",
            f"- موجودہ اسکریننگ: پی ڈی اسکور {current_pd:.3f}",
        ]
        if delta_pct is not None:
            if delta_dir == 'improving':
                lines.append(f"- **فرق: {abs(delta_pct):.1f}% کمی** (بہتری)")
            elif delta_dir == 'worsening':
                lines.append(f"- **فرق: {abs(delta_pct):.1f}% اضافہ** (بگاڑ)")
            else:
                lines.append(f"- **فرق: مستحکم**")
        return "\n".join(lines)
    else:
        lines = [
            f"- Previous screening ({previous_date}): PD Score {previous_pd:.3f}",
            f"- Current screening: PD Score {current_pd:.3f}",
        ]
        if delta_pct is not None:
            if delta_dir == 'improving':
                lines.append(f"- **DELTA: Decreased by {abs(delta_pct):.1f}%** (improvement)")
            elif delta_dir == 'worsening':
                lines.append(f"- **DELTA: Increased by {abs(delta_pct):.1f}%** (worsening)")
            else:
                lines.append(f"- **DELTA: Stable** (no significant change)")
        return "\n".join(lines)


# ================================================================
# PROMPT BUILDER
# ================================================================
def build_doctor_prompt(intelligence, lang='en'):
    """Build a doctor-style prompt with EXACT numbers AND substantive per-section content."""

    fusion = intelligence.get('fusion', {})
    speech = intelligence.get('speech') or {}
    questionnaire = intelligence.get('questionnaire') or {}
    longitudinal = intelligence.get('longitudinal') or {}
    risk = intelligence.get('risk_summary', {})
    insights = intelligence.get('insights', [])

    pattern_label = intelligence.get('primary_pattern_label', '')
    pattern_desc = intelligence.get('primary_pattern_description', '')
    dominant_module = fusion.get('dominant_module', 'N/A')
    dominant_share = fusion.get('dominant_share_pct', 0)
    discordance = fusion.get('discordance_level', 'N/A')
    certainty = fusion.get('certainty_level', 'N/A')

    speech_consistency = speech.get('consistency', 'N/A')
    speech_proximity = speech.get('threshold_proximity', 'N/A')

    motor_severity = questionnaire.get('motor_severity', 'N/A')
    support_level = questionnaire.get('support_level', 'N/A')
    reliability = questionnaire.get('reliability', 'N/A')

    overall_trend = longitudinal.get('overall_trend', 'N/A')

    risk_level = risk.get('risk_level', 'Unknown')
    pd_score = risk.get('final_pd_probability', 0)
    confidence = risk.get('fusion_confidence', 0)

    insights_text = _format_insights_for_prompt(insights, lang)
    longitudinal_text = _format_longitudinal_for_prompt(longitudinal, risk, lang)

    if lang == 'ur':
        prompt = f"""آپ ایک تجربہ کار نیورولوجسٹ ہیں جو اپنے مریض سے اس کے اسکریننگ نتائج پر بات کر رہے ہیں۔

⚠️ اہم قواعد:
1. صرف وہی نمبر استعمال کریں جو نیچے دیے گئے ہیں۔ کوئی نیا نمبر ایجاد نہ کریں۔
2. صرف دی گئی بصیرتوں کا استعمال کریں۔ کوئی عمومی طبی مشورہ نہ دیں۔
3. ہر سیکشن (1-6) میں 2-4 جملے لکھیں۔ چھوٹا جواب قبول نہیں۔
4. "کوئی مخصوص سفارش نہیں" جیسی کوئی بات نہ لکھیں۔
5. ایسے لکھیں جیسے کلینک میں مریض سے بات کر رہے ہیں۔ گرمجوش، ہمدرد۔
6. ایک ہی پیراگراف یا "Quick Note" کو مختلف سیکشنز میں دہرائیں نہیں۔ ہر سیکشن میں منفرد مواد ہونا چاہیے۔
7. ڈیش کریکٹر (—) یا (--) استعمال نہ کریں۔ اس کی جگہ کوما، پوائنٹ، یا عام ہائفن (-) استعمال کریں۔
8. "آپ کی آخری 10 اسکریننگز" والا نمبر کہیں ذکر نہ کریں۔ صرف پچھلی اسکریننگ سے موازنہ کریں۔

📊 مریض کا ڈیٹا:

**رسک:**
- مجموعی رسک: {risk_level}
- پی ڈی رسک اسکور: {pd_score:.3f}
- اعتماد: {confidence:.1%}

**پیٹرن:** {pattern_label}
{pattern_desc}

**فیوژن:**
- غالب ماڈیول: {dominant_module} ({dominant_share:.1f}%)
- اختلاف: {discordance}
- یقین: {certainty}

**آواز:** مستقل مزاجی = {speech_consistency}, حد سے قربت = {speech_proximity}

**سوالنامہ:** موٹر شدت = {motor_severity}, سپورٹ = {support_level}, وشوسنییتا = {reliability}

**پچھلی اسکریننگ سے موازنہ:**
{longitudinal_text}

**کلیدی بصیرتیں:**
{insights_text}

✍️ مریض سے بات کریں۔ ہر سیکشن لازمی لکھیں — چھوٹا نہیں۔

## خلاصہ
(ایک پیراگراف — صرف اوپر دیے گئے نمبر استعمال کریں)

## آپ کے نتائج کا مطلب
(ہر بصیرت کی وضاحت کریں)

## 1. طبی مشاورت
(اگر رسک کم ہے تو "کوئی فوری ڈاکٹر سے ملنے کی ضرورت نہیں، لیکن اپنے اگلے معمول کے چیک اپ میں یہ رپورٹ دکھائیں" جیسی بات لکھیں۔ اگر درمیانہ/زیادہ ہے تو مخصوص وقت بتائیں۔ کم از کم 2-3 جملے لکھیں۔)

## 2. فزیکل تھراپی اور ورزش
(غالب ماڈیول کے مطابق مخصوص مشقیں دیں۔ کم از کم 2-3 جملے + 3 مخصوص مشقیں لکھیں۔)

## 3. اسپیچ اور آواز کی تھراپی
(اگر آواز معمول پر ہے تو "آپ کی آواز ٹھیک ہے، لیکن روزانہ بلند آواز میں پڑھیں تاکہ یہ برقرار رہے" لکھیں۔ اگر سرحدی ہے تو مشورہ دیں۔ کم از کم 2-3 جملے لکھیں۔)

## 4. طرز زندگی اور صحت
(خوراک، نیند، تناؤ، سماجی تعلق — سب پر لکھیں۔ کم از کم 3-4 جملے + 3 مخصوص مشورے لکھیں۔)

## 5. علامات کی نگرانی
(کیا ٹریک کریں، کیسے ٹریک کریں، کب ڈاکٹر سے ملیں۔ کم از کم 2-3 جملے لکھیں۔)

## 6. فالو اپ شیڈول
(کب دوبارہ اسکریننگ، کب ڈاکٹر سے ملیں۔ کم از کم 2-3 جملے لکھیں۔)

---
*یہ سفارشات نیوروسینس اے آئی کے تجزیے پر مبنی ہیں۔ کسی بھی طبی فیصلے سے پہلے مستند ڈاکٹر سے مشورہ کریں۔*"""

    else:
        prompt = f"""You are an experienced neurologist talking to your patient about their screening results.

⚠️ CRITICAL RULES:
1. Use ONLY the exact numbers provided below. Do NOT invent any numbers.
2. Use ONLY the insights listed. Do not add generic medical advice.
3. EVERY section (1-6) MUST have at least 3-5 sentences. Short answers are NOT acceptable.
4. NEVER write "no specific recommendations" or "not applicable". Always provide substance.
5. Write like you're sitting with the patient in a clinic. Be warm, empathetic, conversational.
6. NEVER repeat the same paragraph or "Quick Note" across multiple sections. Each section must contain unique content.
7. NEVER use the em dash character (—) or en dash character (–). Use commas, periods, or regular hyphens (-) instead.
8. Do NOT mention the "across your last 10 screenings" number anywhere. Only mention the comparison with the immediately previous screening.

📊 PATIENT DATA:

**Risk:**
- Overall Risk: {risk_level}
- PD Risk Score: {pd_score:.3f}
- Confidence: {confidence:.1%}

**Pattern Detected:** {pattern_label}
{pattern_desc}

**Fusion Analysis:**
- Dominant Module: {dominant_module} ({dominant_share:.1f}%)
- Discordance: {discordance}
- Certainty: {certainty}

**Speech:** Consistency = {speech_consistency}, Threshold Proximity = {speech_proximity}

**Questionnaire:** Motor Severity = {motor_severity}, Support = {support_level}, Reliability = {reliability}

**Comparison with Previous Screening:**
{longitudinal_text}

**Key Insights:**
{insights_text}

✍️ Write the conversation now. EVERY section must be substantive — no one-liners.

## Your Summary
(One paragraph — use only the exact numbers above)

## What Your Results Mean
(Walk through each insight. Use the EXACT delta percentage when comparing to previous screening.)

## 1. Medical Consultation
(If risk is Low: explain that no urgent visit is needed but suggest sharing this report with the primary care physician at the next routine visit. If Medium/High: give specific timeline. Write at least 3-4 sentences.)

## 2. Physical Therapy & Exercise
(Give SPECIFIC exercises based on the dominant module — gait, speech, or general. Include frequency and duration. Write at least 3-4 sentences + 2-3 specific exercises.)

## 3. Speech & Voice Therapy
(If speech is normal: reassure them and give maintenance exercises like reading aloud. If borderline: recommend monitoring. If PD: recommend SLP consultation. Write at least 3-4 sentences.)

## 4. Lifestyle & Wellness
(Cover diet, sleep, stress, social engagement — each with specific advice. Write at least 4-5 sentences + 3 specific tips.)

## 5. Symptom Monitoring
(What to track, how to track, what to look for. Even if symptoms are minimal, tell them what to watch. Write at least 3-4 sentences.)

## 6. Follow-up Schedule
(When to repeat screening, when to see a doctor, what triggers an earlier visit. Write at least 3-4 sentences.)

---
*These recommendations are based on your NeuroSense AI screening. Consult a qualified healthcare provider before making any medical decisions.*"""

    return prompt


# ================================================================
# MAIN ENTRY POINT
# ================================================================
def generate_ai_recommendations(
    user,
    fusion_result,
    questionnaire_result=None,
    questionnaire_result_id=None,
    screening_id=None,
    lang='en'
):
    """Generate doctor-style personalized recommendations."""

    # ============================================================
    # 1. SYNTHESIZE INTELLIGENCE
    # ============================================================
    try:
        intelligence = synthesize_intelligence(
            user=user,
            fusion_result=fusion_result,
            questionnaire_result=questionnaire_result,
            questionnaire_result_id=questionnaire_result_id,
            screening_id=screening_id,
            lang=lang,
        )
        logger.info(
            f"Intelligence synthesized | pattern={intelligence.get('primary_pattern')} | "
            f"insights={len(intelligence.get('insights', []))}"
        )
    except Exception as e:
        logger.error(f"Intelligence synthesis failed: {str(e)}")
        raise Exception(f"Failed to synthesize intelligence: {str(e)}")

    # ============================================================
    # 2. BUILD PROMPT
    # ============================================================
    prompt = build_doctor_prompt(intelligence, lang=lang)
    logger.info(f"Doctor prompt built ({len(prompt)} chars)")

    # ============================================================
    # 3. CALL GEMINI (new google-genai package)
    # ============================================================
    client = get_gemini_client()
    if not client:
        logger.warning("Gemini client not available. Using fallback.")
        return get_fallback_recommendations(intelligence, lang=lang)

    try:
        from google.genai import types

        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=4096,
            ),
        )

        result = response.text
        logger.info(f"Gemini returned {len(result)} characters")

        if not result or len(result) < 400:
            logger.warning(f"Gemini response too short ({len(result)} chars). Using fallback.")
            return get_fallback_recommendations(intelligence, lang=lang)

        return result

    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        logger.info("Falling back to rule-based recommendations.")
        return get_fallback_recommendations(intelligence, lang=lang)


# ================================================================
# FALLBACK
# ================================================================
def get_fallback_recommendations(intelligence, lang='en'):
    """Fallback if Gemini fails. Uses intelligence directly."""

    risk = intelligence.get('risk_summary', {})
    pattern_label = intelligence.get('primary_pattern_label', '')
    insights = intelligence.get('insights', [])
    fusion = intelligence.get('fusion', {})
    speech = intelligence.get('speech') or {}
    questionnaire = intelligence.get('questionnaire') or {}
    longitudinal = intelligence.get('longitudinal') or {}

    risk_level = risk.get('risk_level', 'Unknown')
    pd_score = risk.get('final_pd_probability', 0)
    confidence = risk.get('fusion_confidence', 0)
    dominant_module = fusion.get('dominant_module', 'N/A')
    speech_proximity = speech.get('threshold_proximity', 'N/A')
    motor_severity = questionnaire.get('motor_severity', 'N/A')
    overall_trend = longitudinal.get('overall_trend', 'N/A')
    delta_pct = longitudinal.get('delta_percentage')
    delta_dir = longitudinal.get('delta_direction')

    if lang == 'ur':
        if delta_pct is not None:
            if delta_dir == 'improving':
                delta_line = f"آپ کی پچھلی اسکریننگ کے مقابلے میں مجموعی رسک **{abs(delta_pct):.1f}% کم** ہوا ہے۔"
            elif delta_dir == 'worsening':
                delta_line = f"آپ کی پچھلی اسکریننگ کے مقابلے میں مجموعی رسک **{abs(delta_pct):.1f}% بڑھا** ہے۔"
            else:
                delta_line = "آپ کا مجموعی رسک پچھلی اسکریننگ کے مقابلے میں **مستحکم** ہے۔"
        else:
            delta_line = "یہ آپ کی پہلی اسکریننگ ہے۔"

        lines = []
        lines.append("## خلاصہ\n")
        lines.append(f"آپ کی اسکریننگ کا مجموعی رسک **{risk_level}** ہے (اسکور: {pd_score:.3f}، اعتماد: {confidence:.1%})۔")
        if pattern_label:
            lines.append(f"ہمارے سسٹم نے **{pattern_label}** کا پیٹرن پہچانا ہے۔")
        lines.append(f"{delta_line}\n")

        if insights:
            lines.append("## آپ کے نتائج کا مطلب\n")
            for ins in insights[:5]:
                text = ins.get('ur', ins.get('en', ''))
                if text:
                    lines.append(f"{text}\n")
            lines.append("")

        lines.append("## 1. طبی مشاورت\n")
        if risk_level == 'High':
            lines.append("میں آپ کو مشورہ دوں گا کہ **2 ہفتوں کے اندر نیورولوجسٹ سے ملیں**۔")
        elif risk_level == 'Medium':
            lines.append("میں آپ کو مشورہ دوں گا کہ **3-4 ہفتوں کے اندر نیورولوجسٹ سے ملیں**۔")
        else:
            lines.append("ابھی فوری طور پر ماہر سے ملنے کی ضرورت نہیں۔ معمول کی صحت کی جانچ جاری رکھیں۔")
        lines.append("")

        lines.append("## 2. فزیکل تھراپی اور ورزش\n")
        if dominant_module == 'gait':
            lines.append("چونکہ آپ کی چال میں تبدیلیاں نظر آئی ہیں:")
            lines.append("- **بڑے قدموں کے ساتھ چلنا**: 10 منٹ، دن میں 2 بار")
            lines.append("- **بازوؤں کی حرکت**: چلتے ہوئے قدرتی طور پر بازو ہلائیں")
            lines.append("- **توازن کی مشقیں**: ٹینڈم اسٹینس، ایک ٹانگ پر کھڑے ہوں")
        else:
            lines.append("- ہفتے میں 3-4 بار 30 منٹ کی ورزش کریں")
            lines.append("- ایروبک، طاقت، اور توازن کی مشقیں ملائیں")
        lines.append("")

        lines.append("## 3. اسپیچ اور آواز کی تھراپی\n")
        if speech_proximity == 'above_parkinson':
            lines.append("آپ کی آواز کے نتائج کی بنیاد پر، **اسپیچ لینگویج پیتھولوجسٹ سے مشورہ** کریں۔")
        elif speech_proximity == 'borderline_zone':
            lines.append("آپ کی آواز سرحدی زون میں ہے — ماہانہ نگرانی کریں۔")
        else:
            lines.append("ابھی اسپیچ تھراپی کی ضرورت نہیں۔ روزانہ 10 منٹ بلند آواز میں پڑھیں۔")
        lines.append("")

        lines.append("## 4. طرز زندگی اور صحت\n")
        lines.append("- **خوراک**: میڈیٹرینین طرز کی خوراک")
        lines.append("- **نیند**: 7-8 گھنٹے کی معیاری نیند")
        lines.append("- **تناؤ کا انتظام**: مراقبہ، گہرے سانس")
        lines.append("")

        lines.append("## 5. علامات کی نگرانی\n")
        if motor_severity in ['high', 'moderate']:
            lines.append("روزانہ علامات کی ڈائری رکھیں۔")
        else:
            lines.append("کسی بھی نئی علامت پر نظر رکھیں۔")
        lines.append("")

        lines.append("## 6. فالو اپ شیڈول\n")
        lines.append("- **3 ماہ بعد دوبارہ اسکریننگ** کریں")
        lines.append("")

        lines.append("---")
        lines.append("*یہ سفارشات نیوروسینس اے آئی کے تجزیے پر مبنی ہیں۔ کسی بھی طبی فیصلے سے پہلے مستند ڈاکٹر سے مشورہ کریں۔*")

    else:
        if delta_pct is not None:
            if delta_dir == 'improving':
                delta_line = f"Compared to your previous screening, your overall risk **decreased by {abs(delta_pct):.1f}%**."
            elif delta_dir == 'worsening':
                delta_line = f"Compared to your previous screening, your overall risk **increased by {abs(delta_pct):.1f}%**."
            else:
                delta_line = "Your overall risk remains **stable** compared to your previous screening."
        else:
            delta_line = "This is your first screening."

        lines = []
        lines.append("## Your Summary\n")
        lines.append(f"Your screening shows an overall risk of **{risk_level}** (score: {pd_score:.3f}, confidence: {confidence:.1%}).")
        if pattern_label:
            lines.append(f"Our system detected a **{pattern_label}** pattern.")
        lines.append(f"{delta_line}\n")

        if insights:
            lines.append("## What Your Results Mean\n")
            for ins in insights[:5]:
                text = ins.get('en', '')
                if text:
                    lines.append(f"{text}\n")
            lines.append("")

        lines.append("## 1. Medical Consultation\n")
        if risk_level == 'High':
            lines.append("I would recommend you **see a neurologist within 2 weeks**.")
        elif risk_level == 'Medium':
            lines.append("I would recommend you **see a neurologist within 3-4 weeks**.")
        else:
            lines.append("No urgent specialist visit needed. Continue routine checkups.")
        lines.append("")

        lines.append("## 2. Physical Therapy & Exercise\n")
        if dominant_module == 'gait':
            lines.append("- **Big-step walking**: 10 minutes, twice daily")
            lines.append("- **Arm swing training**: swing arms naturally")
            lines.append("- **Balance drills**: single-leg stands, tandem stance")
        else:
            lines.append("- Exercise 30 minutes, 3-4 times per week")
        lines.append("")

        lines.append("## 3. Speech & Voice Therapy\n")
        if speech_proximity == 'above_parkinson':
            lines.append("**Consult a Speech-Language Pathologist.**")
        elif speech_proximity == 'borderline_zone':
            lines.append("Monitor voice monthly. Read aloud daily.")
        else:
            lines.append("No speech therapy needed now. Read aloud 10 minutes daily.")
        lines.append("")

        lines.append("## 4. Lifestyle & Wellness\n")
        lines.append("- **Diet**: Mediterranean-style")
        lines.append("- **Sleep**: 7-8 hours")
        lines.append("- **Stress**: meditation, deep breathing")
        lines.append("")

        lines.append("## 5. Symptom Monitoring\n")
        if motor_severity in ['high', 'moderate']:
            lines.append("Keep a daily symptom diary.")
        else:
            lines.append("Monitor for any new symptoms.")
        lines.append("")

        lines.append("## 6. Follow-up Schedule\n")
        lines.append("- **Repeat NeuroSense screening in 3 months**")
        if overall_trend in ['worsening', 'significantly_worsening']:
            lines.append("- Given the worsening trend, schedule a **follow-up sooner**")
        lines.append("")

        lines.append("---")
        lines.append("*These recommendations are based on your NeuroSense AI screening. Consult a qualified healthcare provider before making any medical decisions.*")

    return "\n".join(lines)