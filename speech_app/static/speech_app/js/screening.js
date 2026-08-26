(() => {
    "use strict";

    // ==========================
    // DOM REFERENCES
    // ==========================

    const form =
        document.getElementById("screening-form");

    if (!form) return;

    const submitButton =
        document.getElementById("submit-screening");

    const completedCount =
        document.getElementById("completed-count");

    const completionBadge =
        document.getElementById("completion-badge");

    const processingSection =
        document.getElementById("processing-section");

    const resultsSection =
        document.getElementById("results-section");

    const errorSection =
        document.getElementById("error-section");

    const toastRegion =
        document.getElementById("toast-region");


    const MAX_SECONDS =
        Number(
            form.dataset.maxSeconds || 6
        );

    const REQUIRED_RECORDINGS =
        Number(
            form.dataset.requiredRecordings || 3
        );

    const MAX_FILE_SIZE =
        25 * 1024 * 1024;

    const ALLOWED_EXTENSIONS = [
        "wav",
        "ogg",
        "webm",
        "mp4"
    ];


    // ======================================================
    // LANGUAGE / TRANSLATION SYSTEM
    // ======================================================

    const translations = {

        en: {

            hero_badge:
                "Audio Screening · Step 2 of 4",

            hero_title:
                "Audio Screening",

            hero_description:
                "Pronounce and sustain the /aaaa/ sound for a few seconds so NeuroSense AI can analyze key voice characteristics as part of your guided screening journey.",

            overall_progress:
                "Overall Progress",

            progress_completed:
                "You have completed 1 of 4 modules",

            progress_remaining:
                "Estimated time remaining: ~15 mins",

            sample_title:
                "Listen to a Sample",

            sample_recording:
                "Sample Voice Recording",

            sample_description:
                "Listen to the sample to understand how to sustain the “Aaaa” sound before recording your own voice.",

            sample_note:
                "Match the clarity and steady tone — not the exact voice.",

            estimated_time:
                "Estimated Time",

            estimated_duration:
                "5–7 minutes",

            steps_title:
                "Audio Screening Steps",

            first_recording:
                "First Voice Recording",

            first_recording_help:
                "Sustain the “Aaaa” sound clearly for 4–6 seconds.",

            second_recording:
                "Second Voice Recording",

            second_recording_help:
                "Repeat the “Aaaa” sound for another 4–6 seconds.",

            third_recording:
                "Third Voice Recording",

            third_recording_help:
                "Complete one final “Aaaa” recording for 4–6 seconds.",

            quality_assessment:
                "Voice Quality Assessment",

            automatic_screening:
                "Automatic AI screening",

            required:
                "Required",

            pending:
                "Pending",

            tips_title:
                "Tips for best results",

            quiet_environment:
                "Find a quiet environment",

            speak_clearly:
                "Speak clearly and naturally",

            microphone_distance:
                "Keep the microphone about 15–20 cm away",

            different_samples:
                "Record three different voice samples",

            view_guidance:
                "View Recording Guidance",

            audio_recording:
                "Audio Recording",

            add_recordings:
                "Add your voice recordings",

            add_recordings_help:
                "Record directly using your microphone or upload an existing audio file.",

            checking_service:
                "Checking service",

            ready_suffix:
                "/3 ready",

            voice_sample:
                "Voice Sample",

            recording_label:
                "Recording",

            not_recorded:
                "Not recorded",

            maximum:
                "Maximum",

            seconds:
                "seconds",

            start_recording:
                "Start recording",

            record_again:
                "Record again",

            stop:
                "Stop",

            or_upload:
                "or upload a file",

            upload_file:
                "Upload audio file",

            upload_formats:
                "WAV, MP4, OGG or WebM",

            ready_analysis:
                "Ready for analysis",

            all_recordings_required:
                "All three recordings must be added before screening can begin.",

            analyze_recordings:
                "Analyze recordings",

            audio_analysis:
                "Audio Analysis",

            analyzing_recordings:
                "Analyzing your recordings",

            preparing_quality:
                "Preparing audio for quality assessment.",

            audio_module_complete:
                "Audio Module Complete",

            voice_screening_result:
                "Voice screening result",

            start_again:
                "Start again",

            screening_category:
                "Screening category",

            decision_score:
                "Decision score",

            confidence:
                "Confidence",

            consistency:
                "Consistency",

            agreement:
                "Agreement",

            recording_quality:
                "Recording quality",

            recording_quality_help:
                "Individual recording quality and prediction details.",

            result_disclaimer:
                "This result is intended for screening purposes only and is not a medical diagnosis.",

            action_required:
                "Action Required",

            screening_failed:
                "Screening could not be completed",

            review_recordings:
                "Review recordings",

            secure_data:
                "Your data is secure and encrypted.",

            privacy:
                "We prioritize your privacy and data protection.",

            learn_security:
                "Learn more about data security",

            ready:
                "Ready",

            recording_ready:
                "Recording ready.",

            invalid:
                "Invalid",

            invalid_file:
                "Invalid File",

            removed:
                "Removed",

            unsupported:
                "Unsupported",

            microphone_unavailable:
                "Microphone unavailable",

            recording_unavailable:
                "Recording unavailable",

            recording_active:
                "Recording active",

            stop_current_recording:
                "Stop the current recording before starting another.",

            recording_failed:
                "Recording failed",

            microphone_recording_failed:
                "The microphone recording could not be completed. Please try again.",

            recording:
                "Recording",

            microphone_error:
                "Microphone error",

            empty_recording:
                "Empty recording",

            no_audio_captured:
                "No audio was captured. Please record again.",

            no_audio_from_microphone:
                "No audio was captured from the microphone.",

            invalid_recording:
                "Invalid recording",

            preparation_failed:
                "Preparation failed",

            recording_preparation_failed:
                "The recording could not be prepared for upload.",

            recording_error:
                "Recording error",

            upload_audio_instead:
                "Upload an audio file instead.",

            recording_complete:
                "Recording complete",

            recording_is_ready:
                "Recording {id} is ready.",

            recording_removed:
                "Recording {id} removed.",

            no_file_selected:
                "No file selected.",

            unsupported_format:
                "Unsupported format.",

            max_upload_size:
                "Maximum upload size is 25 MB.",

            empty_audio_file:
                "Empty audio file.",

            microphone_not_supported:
                "Microphone access is not supported by this browser.",

            audio_recording_not_supported:
                "Audio recording is not supported by this browser.",

            microphone_access_failed:
                "Microphone access failed.",

            microphone_permission_denied:
                "Microphone permission was denied. Allow access or upload an audio file.",

            no_microphone_detected:
                "No microphone was detected on this device.",

            microphone_in_use:
                "The microphone is already in use by another application.",

            microphone_https_required:
                "Microphone access requires a secure HTTPS connection.",

            preparing_uploaded_audio:
                "Preparing uploaded audio.",

            assessing_recording_quality:
                "Assessing recording quality.",

            checking_duplicate_recordings:
                "Checking for duplicate recordings.",

            extracting_voice_features:
                "Extracting voice features.",

            running_screening_model:
                "Running the screening model.",

            combining_predictions:
                "Combining recording predictions.",

            preparing_audio_analysis:
                "Preparing audio for analysis.",

            screening_completed:
                "Audio screening completed successfully.",

            recording_successful:
                "Recording successful",

            screening_completed_message:
                "Your audio screening has been completed.",

            one_recording_attention:
                "One recording needs your attention.",

            multiple_recordings_attention:
                "{count} recordings need your attention.",

            recording_needs_attention:
                "Recording needs attention",

            review_highlighted_recording:
                "Please review the highlighted recording.",

            review_highlighted_recordings:
                "Please review the highlighted recordings.",

            unable_complete_screening:
                "Unable to complete screening",

            screening_could_not_complete:
                "Audio screening could not be completed.",

            add_recording_before_continue:
                "Add a recording before continuing.",

            recordings_required:
                "Recordings required",

            add_three_valid_recordings:
                "Add three valid recordings before continuing.",

            server_error:
                "The server returned an error ({status}).",

            server_unreachable:
                "The server could not be reached. Check your connection and try again.",

            analysis_complete:
                "Analysis complete.",

            service_ready:
                "Service ready",

            service_unavailable:
                "Service unavailable"
        },


        ur: {

            hero_badge:
                "آڈیو اسکریننگ · مرحلہ 2 از 4",

            hero_title:
                "آڈیو اسکریننگ",

            hero_description:
                "چند سیکنڈ تک مسلسل “آآآآ” کی آواز نکالیں تاکہ نیورو سینس اے آئی آپ کی آواز کی اہم خصوصیات کا تجزیہ کر سکے۔",

            overall_progress:
                "مجموعی پیش رفت",

            progress_completed:
                "آپ 4 میں سے 1 ماڈیول مکمل کر چکے ہیں",

            progress_remaining:
                "متوقع باقی وقت: تقریباً 15 منٹ",

            sample_title:
                "نمونہ ریکارڈنگ سنیں",

            sample_recording:
                "نمونہ آواز کی ریکارڈنگ",

            sample_description:
                "اپنی آواز ریکارڈ کرنے سے پہلے نمونہ سنیں تاکہ آپ سمجھ سکیں کہ “آآآآ” کی آواز کو کس طرح مسلسل ادا کرنا ہے۔",

            sample_note:
                "وضاحت اور مستقل آواز پر توجہ دیں، عین اسی آواز کی نقل ضروری نہیں۔",

            estimated_time:
                "متوقع وقت",

            estimated_duration:
                "5–7 منٹ",

            steps_title:
                "آڈیو اسکریننگ کے مراحل",

            first_recording:
                "پہلی آواز کی ریکارڈنگ",

            first_recording_help:
                "واضح انداز میں 4–6 سیکنڈ تک مسلسل “آآآآ” کی آواز نکالیں۔",

            second_recording:
                "دوسری آواز کی ریکارڈنگ",

            second_recording_help:
                "دوبارہ 4–6 سیکنڈ تک “آآآآ” کی آواز ریکارڈ کریں۔",

            third_recording:
                "تیسری آواز کی ریکارڈنگ",

            third_recording_help:
                "4–6 سیکنڈ کی آخری “آآآآ” ریکارڈنگ مکمل کریں۔",

            quality_assessment:
                "آواز کے معیار کی جانچ",

            automatic_screening:
                "خودکار اے آئی اسکریننگ",

            required:
                "ضروری",

            pending:
                "زیرِ انتظار",

            tips_title:
                "بہتر نتائج کے لیے ہدایات",

            quiet_environment:
                "پرسکون جگہ کا انتخاب کریں",

            speak_clearly:
                "واضح اور قدرتی انداز میں آواز دیں",

            microphone_distance:
                "مائیکروفون کو تقریباً 15–20 سینٹی میٹر دور رکھیں",

            different_samples:
                "آواز کے تین الگ نمونے ریکارڈ کریں",

            view_guidance:
                "ریکارڈنگ ہدایات دیکھیں",

            audio_recording:
                "آڈیو ریکارڈنگ",

            add_recordings:
                "اپنی آواز کی ریکارڈنگ شامل کریں",

            add_recordings_help:
                "مائیکروفون سے براہِ راست ریکارڈ کریں یا موجودہ آڈیو فائل اپ لوڈ کریں۔",

            checking_service:
                "سروس چیک کی جا رہی ہے",

            ready_suffix:
                "/3 تیار",

            voice_sample:
                "آواز کا نمونہ",

            recording_label:
                "ریکارڈنگ",

            not_recorded:
                "ریکارڈ نہیں ہوئی",

            maximum:
                "زیادہ سے زیادہ",

            seconds:
                "سیکنڈ",

            start_recording:
                "ریکارڈنگ شروع کریں",

            record_again:
                "دوبارہ ریکارڈ کریں",

            stop:
                "روکیں",

            or_upload:
                "یا فائل اپ لوڈ کریں",

            upload_file:
                "آڈیو فائل اپ لوڈ کریں",

            upload_formats:
                "WAV، MP4، OGG یا WebM",

            ready_analysis:
                "تجزیے کے لیے تیار",

            all_recordings_required:
                "اسکریننگ شروع کرنے سے پہلے تینوں ریکارڈنگز شامل کرنا ضروری ہیں۔",

            analyze_recordings:
                "ریکارڈنگز کا تجزیہ کریں",

            audio_analysis:
                "آڈیو تجزیہ",

            analyzing_recordings:
                "آپ کی ریکارڈنگز کا تجزیہ کیا جا رہا ہے",

            preparing_quality:
                "آواز کے معیار کی جانچ کے لیے آڈیو تیار کی جا رہی ہے۔",

            audio_module_complete:
                "آڈیو ماڈیول مکمل",

            voice_screening_result:
                "آواز کی اسکریننگ کا نتیجہ",

            start_again:
                "دوبارہ شروع کریں",

            screening_category:
                "اسکریننگ کی درجہ بندی",

            decision_score:
                "فیصلہ اسکور",

            confidence:
                "اعتماد",

            consistency:
                "مطابقت",

            agreement:
                "اتفاق",

            recording_quality:
                "ریکارڈنگ کا معیار",

            recording_quality_help:
                "ہر ریکارڈنگ کے معیار اور پیش گوئی کی تفصیلات۔",

            result_disclaimer:
                "یہ نتیجہ صرف اسکریننگ کے لیے ہے اور طبی تشخیص نہیں ہے۔",

            action_required:
                "کارروائی درکار ہے",

            screening_failed:
                "اسکریننگ مکمل نہیں ہو سکی",

            review_recordings:
                "ریکارڈنگز کا جائزہ لیں",

            secure_data:
                "آپ کا ڈیٹا محفوظ اور خفیہ ہے۔",

            privacy:
                "ہم آپ کی رازداری اور ڈیٹا کے تحفظ کو ترجیح دیتے ہیں۔",

            learn_security:
                "ڈیٹا سیکیورٹی کے بارے میں مزید جانیں",

            ready:
                "تیار",

            recording_ready:
                "ریکارڈنگ تیار ہے۔",

            invalid:
                "غلط",

            invalid_file:
                "غلط فائل",

            removed:
                "ہٹا دی گئی",

            unsupported:
                "غیر معاون",

            microphone_unavailable:
                "مائیکروفون دستیاب نہیں",

            recording_unavailable:
                "ریکارڈنگ دستیاب نہیں",

            recording_active:
                "ریکارڈنگ جاری ہے",

            stop_current_recording:
                "نئی ریکارڈنگ شروع کرنے سے پہلے موجودہ ریکارڈنگ روکیں۔",

            recording_failed:
                "ریکارڈنگ ناکام",

            microphone_recording_failed:
                "مائیکروفون ریکارڈنگ مکمل نہیں ہو سکی۔ براہِ کرم دوبارہ کوشش کریں۔",

            recording:
                "ریکارڈنگ جاری ہے",

            microphone_error:
                "مائیکروفون میں خرابی",

            empty_recording:
                "خالی ریکارڈنگ",

            no_audio_captured:
                "کوئی آواز ریکارڈ نہیں ہوئی۔ براہِ کرم دوبارہ ریکارڈ کریں۔",

            no_audio_from_microphone:
                "مائیکروفون سے کوئی آواز ریکارڈ نہیں ہوئی۔",

            invalid_recording:
                "غلط ریکارڈنگ",

            preparation_failed:
                "تیاری ناکام",

            recording_preparation_failed:
                "ریکارڈنگ اپ لوڈ کے لیے تیار نہیں ہو سکی۔",

            recording_error:
                "ریکارڈنگ میں خرابی",

            upload_audio_instead:
                "اس کے بجائے آڈیو فائل اپ لوڈ کریں۔",

            recording_complete:
                "ریکارڈنگ مکمل",

            recording_is_ready:
                "ریکارڈنگ {id} تیار ہے۔",

            recording_removed:
                "ریکارڈنگ {id} ہٹا دی گئی۔",

            no_file_selected:
                "کوئی فائل منتخب نہیں کی گئی۔",

            unsupported_format:
                "فائل فارمیٹ قابلِ قبول نہیں۔",

            max_upload_size:
                "زیادہ سے زیادہ اپ لوڈ سائز 25 MB ہے۔",

            empty_audio_file:
                "آڈیو فائل خالی ہے۔",

            microphone_not_supported:
                "یہ براؤزر مائیکروفون رسائی کو سپورٹ نہیں کرتا۔",

            audio_recording_not_supported:
                "یہ براؤزر آڈیو ریکارڈنگ کو سپورٹ نہیں کرتا۔",

            microphone_access_failed:
                "مائیکروفون تک رسائی ناکام ہو گئی۔",

            microphone_permission_denied:
                "مائیکروفون کی اجازت مسترد کر دی گئی۔ اجازت دیں یا آڈیو فائل اپ لوڈ کریں۔",

            no_microphone_detected:
                "اس ڈیوائس پر کوئی مائیکروفون نہیں ملا۔",

            microphone_in_use:
                "مائیکروفون کسی دوسری ایپلی کیشن کے استعمال میں ہے۔",

            microphone_https_required:
                "مائیکروفون تک رسائی کے لیے محفوظ HTTPS کنکشن ضروری ہے۔",

            preparing_uploaded_audio:
                "اپ لوڈ شدہ آڈیو تیار کی جا رہی ہے۔",

            assessing_recording_quality:
                "ریکارڈنگ کے معیار کی جانچ کی جا رہی ہے۔",

            checking_duplicate_recordings:
                "ایک جیسی ریکارڈنگز چیک کی جا رہی ہیں۔",

            extracting_voice_features:
                "آواز کی خصوصیات نکالی جا رہی ہیں۔",

            running_screening_model:
                "اسکریننگ ماڈل چلایا جا رہا ہے۔",

            combining_predictions:
                "ریکارڈنگز کی پیش گوئیاں یکجا کی جا رہی ہیں۔",

            preparing_audio_analysis:
                "تجزیے کے لیے آڈیو تیار کی جا رہی ہے۔",

            screening_completed:
                "آڈیو اسکریننگ کامیابی سے مکمل ہو گئی۔",

            recording_successful:
                "ریکارڈنگ کامیاب",

            screening_completed_message:
                "آپ کی آڈیو اسکریننگ مکمل ہو گئی ہے۔",

            one_recording_attention:
                "ایک ریکارڈنگ پر توجہ درکار ہے۔",

            multiple_recordings_attention:
                "{count} ریکارڈنگز پر توجہ درکار ہے۔",

            recording_needs_attention:
                "ریکارڈنگ پر توجہ درکار ہے",

            review_highlighted_recording:
                "براہِ کرم نمایاں ریکارڈنگ کا جائزہ لیں۔",

            review_highlighted_recordings:
                "براہِ کرم نمایاں ریکارڈنگز کا جائزہ لیں۔",

            unable_complete_screening:
                "اسکریننگ مکمل نہیں ہو سکی",

            screening_could_not_complete:
                "آڈیو اسکریننگ مکمل نہیں ہو سکی۔",

            add_recording_before_continue:
                "آگے بڑھنے سے پہلے ریکارڈنگ شامل کریں۔",

            recordings_required:
                "ریکارڈنگز ضروری ہیں",

            add_three_valid_recordings:
                "آگے بڑھنے سے پہلے تین درست ریکارڈنگز شامل کریں۔",

            server_error:
                "سرور نے خرابی واپس کی ({status})۔",

            server_unreachable:
                "سرور سے رابطہ نہیں ہو سکا۔ اپنا کنکشن چیک کریں اور دوبارہ کوشش کریں۔",

            analysis_complete:
                "تجزیہ مکمل",

            service_ready:
                "سروس تیار ہے",

            service_unavailable:
                "سروس دستیاب نہیں"
        }
    };


    // ==========================
    // LANGUAGE HELPERS
    // ==========================

    let currentLanguage =
        localStorage.getItem(
            "neurosense-language"
        ) || "en";


    function t(
        key,
        values = {}
    ) {

        let value =
            translations[currentLanguage]?.[key] ||
            translations.en[key] ||
            key;

        Object.entries(
            values
        ).forEach(
            ([name, replacement]) => {

                value =
                    value.replace(
                        `{${name}}`,
                        String(replacement)
                    );

            }
        );

        return value;
    }


  function translateQualityMessage(message) {

    if (!message) {
        return message;
    }

    const messagePairs = [
        {
            en: "This recording appears to be the same as another recording. Please provide a new and different voice sample.",
            ur: "یہ ریکارڈنگ کسی دوسری ریکارڈنگ جیسی معلوم ہو رہی ہے۔ براہِ کرم آواز کا نیا اور مختلف نمونہ فراہم کریں۔"
        },
        {
            en: "Too much background noise was detected. Please move to a quieter place and record again.",
            ur: "پس منظر میں بہت زیادہ شور موجود ہے۔ براہِ کرم کسی پرسکون جگہ پر جا کر دوبارہ ریکارڈ کریں۔"
        },
        {
            en: "The recording is too short. Please sustain 'Aaaa' for at least 3 seconds.",
            ur: "ریکارڈنگ بہت مختصر ہے۔ براہِ کرم کم از کم 3 سیکنڈ تک مسلسل “آآآآ” کی آواز نکالیں۔"
        },
        {
            en: "Your voice is too quiet. Please speak a little louder or move closer to the microphone.",
            ur: "آپ کی آواز بہت ہلکی ہے۔ براہِ کرم تھوڑی بلند آواز میں آواز دیں یا مائیکروفون کے قریب آئیں۔"
        },
        {
            en: "Your voice is too loud or too close to the microphone. Please move slightly away and record again.",
            ur: "آپ کی آواز بہت بلند ہے یا مائیکروفون بہت قریب ہے۔ براہِ کرم تھوڑا دور ہو کر دوبارہ ریکارڈ کریں۔"
        },
        {
            en: "Too much silence was detected. Please sustain 'Aaaa' continuously while recording.",
            ur: "ریکارڈنگ میں بہت زیادہ خاموشی ہے۔ براہِ کرم ریکارڈنگ کے دوران مسلسل “آآآآ” کی آواز نکالیں۔"
        },
        {
            en: "We couldn't assess the recording quality. Please record again.",
            ur: "ریکارڈنگ کے معیار کی درست جانچ نہیں ہو سکی۔ براہِ کرم دوبارہ ریکارڈ کریں۔"
        },
        {
            en: "This audio format could not be processed correctly. Please record again.",
            ur: "اس آڈیو فارمیٹ کو درست طور پر پراسیس نہیں کیا جا سکا۔ براہِ کرم دوبارہ ریکارڈ کریں۔"
        },
        {
            en: "This audio format is not supported correctly. Please record again.",
            ur: "یہ آڈیو فارمیٹ درست طور پر سپورٹ نہیں ہو رہا۔ براہِ کرم دوبارہ ریکارڈ کریں۔"
        },
        {
            en: "We couldn't process this recording. Please record or upload it again.",
            ur: "اس ریکارڈنگ کو پراسیس نہیں کیا جا سکا۔ براہِ کرم دوبارہ ریکارڈ یا اپ لوڈ کریں۔"
        }
    ];

    const pair = messagePairs.find(
        item =>
            item.en === message ||
            item.ur === message
    );

    if (!pair) {
        return message;
    }

    return currentLanguage === "ur"
        ? pair.ur
        : pair.en;
}

    // ==========================
    // RECORDING SLOTS
    // ==========================

    const slots = {};

    let activeRecorder = null;
    let screeningId = null;
    let lastErrorData = null;


    for (
        let i = 1;
        i <= REQUIRED_RECORDINGS;
        i++
    ) {

        slots[i] = {

            id: i,

            card:
                document.querySelector(
                    `[data-recording-slot="${i}"]`
                ),

            fileInput:
                document.querySelector(
                    `input[name="recording_${i}"]`
                ),

            recordButton:
                document.querySelector(
                    `[data-recording-slot="${i}"] .record-button`
                ),

            stopButton:
                document.querySelector(
                    `[data-recording-slot="${i}"] .stop-button`
                ),

            removeButton:
                document.querySelector(
                    `[data-recording-slot="${i}"] .remove-recording-button`
                ),

            timer:
                document.getElementById(
                    `timer-${i}`
                ),

            preview:
                document.getElementById(
                    `audio-preview-${i}`
                ),

            filename:
                document.getElementById(
                    `file-name-${i}`
                ),

            state:
                document.getElementById(
                    `state-text-${i}`
                ),

            message:
                document.getElementById(
                    `field-message-${i}`
                ),

            recorder: null,
            stream: null,
            chunks: [],
            blob: null,
            file: null,
            objectURL: null,
            timerInterval: null,
            startedAt: null
        };
    }


    // ==========================
    // LANGUAGE APPLICATION
    // ==========================

    function applyLanguage(
        language
    ) {

        if (
            !translations[language]
        ) {
            language = "en";
        }

        currentLanguage =
            language;

        localStorage.setItem(
            "neurosense-language",
            language
        );


        const page =
            document.querySelector(
                ".audio-screening-page"
            );

        if (page) {

            page.setAttribute(
                "lang",
                language
            );

            page.setAttribute(
                "dir",
                language === "ur"
                    ? "rtl"
                    : "ltr"
            );

            page.classList.toggle(
                "is-urdu",
                language === "ur"
            );
        }


        document
            .querySelectorAll(
                "[data-i18n]"
            )
            .forEach(
                element => {

                    const key =
                        element.dataset.i18n;

                    if (
                        translations[language]?.[key]
                    ) {

                        element.textContent =
                            translations[
                                language
                            ][key];
                    }
                }
            );


        document
            .querySelectorAll(
                ".language-button"
            )
            .forEach(
                button => {

                    button.classList.toggle(
                        "is-active",
                        button.dataset.language ===
                            language
                    );
                }
            );


        Object
            .values(slots)
            .forEach(
                slot => {

                    if (
                        !slot ||
                        !slot.card
                    ) {
                        return;
                    }


                    if (
                        slot.card.classList.contains(
                            "is-recording"
                        )
                    ) {

                        slot.state.textContent =
                            t("recording");

                    } else if (
                        slot.card.classList.contains(
                            "is-ready"
                        )
                    ) {

                        slot.state.textContent =
                            t("ready");

                    // Translate "Recording ready."
                        if (slot.message) {
                            slot.message.textContent =
                                t("recording_ready");
                        }

                    } else if (
                        slot.card.classList.contains(
                            "has-error"
                        )
                    ) {

                    // Translate error card status
                        slot.state.textContent =
                            t("record_again");

                    } else {

                        slot.state.textContent =
                            t("not_recorded");
                    }


                    const recordLabel =
                        slot.recordButton
                            ?.querySelector(
                                "span"
                            );

                    if (recordLabel) {

                        recordLabel.textContent =
                            slot.fileInput
                                ?.files
                                ?.length > 0
                                ? t(
                                    "record_again"
                                )
                                : t(
                                    "start_recording"
                                );
                    }
                }
            );
            
            refreshVisibleErrorLanguage();

    }


    document
        .querySelectorAll(
            ".language-button"
        )
        .forEach(
            button => {

                button.addEventListener(
                    "click",
                    () => {

                        applyLanguage(
                            button.dataset.language
                        );
                    }
                );
            }
        );


    // ==========================
    // UTILITIES
    // ==========================

    function escapeHTML(
        value
    ) {

        return String(value)
            .replaceAll(
                "&",
                "&amp;"
            )
            .replaceAll(
                "<",
                "&lt;"
            )
            .replaceAll(
                ">",
                "&gt;"
            );
    }


    function toast(
        title,
        message,
        type = "info"
    ) {

        const item =
            document.createElement(
                "div"
            );

        item.className =
            `toast is-${type}`;

        item.innerHTML = `
            <div class="toast-icon">
                ${
                    type === "success"
                        ? "✓"
                        : type === "error"
                        ? "!"
                        : "i"
                }
            </div>

            <div class="toast-content">
                <strong>
                    ${escapeHTML(title)}
                </strong>

                <p>
                    ${escapeHTML(message)}
                </p>
            </div>
        `;

        toastRegion.appendChild(
            item
        );

        setTimeout(
            () => {

                item.classList.add(
                    "is-leaving"
                );

                setTimeout(
                    () => {
                        item.remove();
                    },
                    250
                );
            },
            4000
        );
    }


    function formatTime(
        ms
    ) {

        const sec =
            Math.ceil(
                ms / 1000
            );

        const min =
            Math.floor(
                sec / 60
            );

        const remain =
            sec % 60;

        return (
            `${String(min).padStart(2, "0")}:` +
            `${String(remain).padStart(2, "0")}`
        );
    }


    function updateCompletion() {

        let completed = 0;

        Object
            .values(slots)
            .forEach(
                slot => {

                    if (
                        slot.fileInput
                            .files
                            .length > 0
                    ) {
                        completed++;
                    }
                }
            );

        completedCount.textContent =
            completed;

        completionBadge
            .classList
            .toggle(
                "is-complete",
                completed ===
                    REQUIRED_RECORDINGS
            );

        submitButton.disabled =
            completed !==
            REQUIRED_RECORDINGS;
    }


    function clearMessage(
        slot
    ) {

        slot.message.textContent =
            "";

        slot.message.className =
            "field-message";
    }


    function setMessage(
        slot,
        text,
        type = "info"
    ) {

        slot.message.textContent =
            text;

        slot.message.className =
            `field-message is-${type}`;
    }


    function setState(
        slot,
        css,
        label
    ) {

        slot.card.classList.remove(
            "is-ready",
            "is-recording",
            "has-error"
        );

        if (css) {
            slot.card.classList.add(
                css
            );
        }

        slot.state.textContent =
            label;
    }


    function revokeURL(
        slot
    ) {

        if (
            slot.objectURL
        ) {

            URL.revokeObjectURL(
                slot.objectURL
            );

            slot.objectURL =
                null;
        }
    }


    function stopStream(
        slot
    ) {

        if (
            !slot.stream
        ) {
            return;
        }

        slot.stream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        slot.stream =
            null;
    }


    // ==========================
    // FILE PREVIEW
    // ==========================

    function showPreview(
        slot,
        file
    ) {

        revokeURL(slot);

        slot.objectURL =
            URL.createObjectURL(
                file
            );

        slot.preview.src =
            slot.objectURL;

        slot.preview.hidden =
            false;

        slot.filename.hidden =
            false;

        slot.filename.textContent =
            file.name;

        slot.removeButton.hidden =
            false;

        setState(
            slot,
            "is-ready",
            t("ready")
        );

        setMessage(
            slot,
            t("recording_ready"),
            "success"
        );

        updateCompletion();
    }


    // ==========================
    // RESET SLOT
    // ==========================

    function resetSlot(
        slot
    ) {

        revokeURL(slot);

        stopStream(slot);

        if (
            slot.timerInterval
        ) {

            clearInterval(
                slot.timerInterval
            );
        }

        slot.preview.pause();

        slot.preview.removeAttribute(
            "src"
        );

        slot.preview.load();

        slot.preview.hidden =
            true;

        slot.filename.hidden =
            true;

        slot.removeButton.hidden =
            true;

        slot.fileInput.value =
            "";

        slot.file =
            null;

        slot.blob =
            null;

        slot.chunks =
            [];

        slot.timer.textContent =
            "00:00";

        slot.recordButton.disabled =
            false;

        slot.stopButton.disabled =
            true;

        clearMessage(
            slot
        );

        setState(
            slot,
            "",
            t("not_recorded")
        );

        const recordLabel =
            slot.recordButton
                .querySelector(
                    "span"
                );

        if (recordLabel) {

            recordLabel.textContent =
                t("start_recording");
        }

        updateCompletion();
    }


    // ==========================
    // FILE VALIDATION
    // ==========================

    function validate(
        file
    ) {

        if (!file) {
            return t(
                "no_file_selected"
            );
        }

        const extension =
            file.name
                .split(".")
                .pop()
                .toLowerCase();

        if (
            !ALLOWED_EXTENSIONS
                .includes(
                    extension
                )
        ) {

            return t(
                "unsupported_format"
            );
        }

        if (
            file.size >
            MAX_FILE_SIZE
        ) {

            return t(
                "max_upload_size"
            );
        }

        if (
            file.size === 0
        ) {

            return t(
                "empty_audio_file"
            );
        }

        return null;
    }


    // ==========================
    // FILE INPUT EVENTS
    // ==========================

    Object
        .values(slots)
        .forEach(
            slot => {

                slot.fileInput
                    .addEventListener(
                        "change",
                        () => {

                            const file =
                                slot
                                    .fileInput
                                    .files[0];

                            const error =
                                validate(
                                    file
                                );

                            if (error) {

                                resetSlot(
                                    slot
                                );

                                setState(
                                    slot,
                                    "has-error",
                                    t(
                                        "invalid"
                                    )
                                );

                                setMessage(
                                    slot,
                                    error,
                                    "error"
                                );

                                toast(
                                    t(
                                        "invalid_file"
                                    ),
                                    error,
                                    "error"
                                );

                                return;
                            }

                            slot.file =
                                file;

                            showPreview(
                                slot,
                                file
                            );
                        }
                    );


                slot.removeButton
                    .addEventListener(
                        "click",
                        () => {

                            resetSlot(
                                slot
                            );

                            toast(
                                t(
                                    "removed"
                                ),
                                t(
                                    "recording_removed",
                                    {
                                        id:
                                            slot.id
                                    }
                                ),
                                "info"
                            );
                        }
                    );
            }
        );


    // ==========================
    // MEDIA RECORDER HELPERS
    // ==========================

    function getSupportedMimeType() {

        if (
            !window.MediaRecorder
        ) {
            return "";
        }

        const mimeTypes = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/ogg"
        ];

        for (
            const type
            of mimeTypes
        ) {

            if (
                MediaRecorder
                    .isTypeSupported(
                        type
                    )
            ) {

                return type;
            }
        }

        return "";
    }


    function createRecordedFile(
        slot,
        blob,
        mimeType
    ) {

        const extension =
            mimeType.includes(
                "ogg"
            )
                ? "ogg"
                : "webm";

        return new File(
            [blob],
            `recording_${slot.id}.${extension}`,
            {
                type:
                    mimeType,

                lastModified:
                    Date.now()
            }
        );
    }


    function assignFileToInput(
        slot,
        file
    ) {

        try {

            const transfer =
                new DataTransfer();

            transfer.items.add(
                file
            );

            slot.fileInput.files =
                transfer.files;

            return true;

        } catch (error) {

            console.error(
                "Unable to assign recorded file:",
                error
            );

            return false;
        }
    }
    // ==========================
    // START RECORDING
    // ==========================

    async function startRecording(
        slot
    ) {

        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices
                .getUserMedia
        ) {

            const message =
                t(
                    "microphone_not_supported"
                );

            setState(
                slot,
                "has-error",
                t("unsupported")
            );

            setMessage(
                slot,
                message,
                "error"
            );

            toast(
                t(
                    "microphone_unavailable"
                ),
                message,
                "error"
            );

            return;
        }


        if (
            !window.MediaRecorder
        ) {

            const message =
                t(
                    "audio_recording_not_supported"
                );

            setState(
                slot,
                "has-error",
                t("unsupported")
            );

            setMessage(
                slot,
                message,
                "error"
            );

            toast(
                t(
                    "recording_unavailable"
                ),
                message,
                "error"
            );

            return;
        }


        if (
            activeRecorder &&
            activeRecorder !== slot
        ) {

            toast(
                t(
                    "recording_active"
                ),
                t(
                    "stop_current_recording"
                ),
                "error"
            );

            return;
        }


        resetSlot(
            slot
        );


        try {

            const stream =
                await navigator
                    .mediaDevices
                    .getUserMedia({
                        audio: {
                            echoCancellation:
                                false,

                            noiseSuppression:
                                false,

                            autoGainControl:
                                false,

                            channelCount:
                                1
                        },

                        video:
                            false
                    });


            const audioTrack =
                stream
                    .getAudioTracks()[0];

            if (
                audioTrack
            ) {

                console.log(
                    "Requested microphone constraints:",
                    audioTrack
                        .getConstraints()
                );

                console.log(
                    "Applied microphone settings:",
                    audioTrack
                        .getSettings()
                );
            }


            const mimeType =
                getSupportedMimeType();


            const recorderOptions =
                mimeType
                    ? {
                        mimeType
                    }
                    : undefined;


            const recorder =
                new MediaRecorder(
                    stream,
                    recorderOptions
                );


            slot.stream =
                stream;

            slot.recorder =
                recorder;

            slot.chunks =
                [];

            slot.startedAt =
                Date.now();

            activeRecorder =
                slot;


            recorder
                .addEventListener(
                    "dataavailable",
                    event => {

                        if (
                            event.data &&
                            event.data
                                .size > 0
                        ) {

                            slot
                                .chunks
                                .push(
                                    event.data
                                );
                        }
                    }
                );


            recorder
                .addEventListener(
                    "stop",
                    () => {

                        finishRecording(
                            slot,
                            mimeType ||
                            recorder.mimeType ||
                            "audio/webm"
                        );
                    }
                );


            recorder
                .addEventListener(
                    "error",
                    event => {

                        console.error(
                            "MediaRecorder error:",
                            event.error
                        );


                        stopStream(
                            slot
                        );


                        if (
                            slot
                                .timerInterval
                        ) {

                            clearInterval(
                                slot
                                    .timerInterval
                            );

                            slot.timerInterval =
                                null;
                        }


                        slot.recordButton
                            .disabled =
                            false;

                        slot.stopButton
                            .disabled =
                            true;

                        activeRecorder =
                            null;


                        setState(
                            slot,
                            "has-error",
                            t(
                                "recording_failed"
                            )
                        );


                        setMessage(
                            slot,
                            t(
                                "microphone_recording_failed"
                            ),
                            "error"
                        );


                        toast(
                            t(
                                "recording_failed"
                            ),
                            t(
                                "microphone_recording_failed"
                            ),
                            "error"
                        );


                        updateCompletion();
                    }
                );


            recorder.start(
                200
            );


            setState(
                slot,
                "is-recording",
                t("recording")
            );


            clearMessage(
                slot
            );


            slot.recordButton
                .disabled =
                true;

            slot.stopButton
                .disabled =
                false;


            const recordLabel =
                slot
                    .recordButton
                    .querySelector(
                        "span"
                    );


            if (
                recordLabel
            ) {

                recordLabel
                    .textContent =
                    t(
                        "recording"
                    );
            }


            slot.timer
                .textContent =
                "00:00";


            slot.timerInterval =
                setInterval(
                    () => {

                        const elapsed =
                            Date.now() -
                            slot
                                .startedAt;


                        slot.timer
                            .textContent =
                            formatTime(
                                elapsed
                            );


                        if (
                            elapsed >=
                            MAX_SECONDS *
                            1000
                        ) {

                            stopRecording(
                                slot
                            );
                        }
                    },
                    100
                );


            updateCompletion();

        } catch (error) {

            console.error(
                "Microphone access error:",
                error
            );


            let message =
                t(
                    "microphone_access_failed"
                );


            if (
                error.name ===
                "NotAllowedError"
            ) {

                message =
                    t(
                        "microphone_permission_denied"
                    );

            } else if (
                error.name ===
                "NotFoundError"
            ) {

                message =
                    t(
                        "no_microphone_detected"
                    );

            } else if (
                error.name ===
                "NotReadableError"
            ) {

                message =
                    t(
                        "microphone_in_use"
                    );

            } else if (
                error.name ===
                "SecurityError"
            ) {

                message =
                    t(
                        "microphone_https_required"
                    );
            }


            setState(
                slot,
                "has-error",
                t(
                    "microphone_error"
                )
            );


            setMessage(
                slot,
                message,
                "error"
            );


            toast(
                t(
                    "microphone_error"
                ),
                message,
                "error"
            );
        }
    }


    // ==========================
    // STOP RECORDING
    // ==========================

    function stopRecording(
        slot
    ) {

        if (
            !slot.recorder ||
            slot.recorder
                .state ===
                "inactive"
        ) {
            return;
        }


        slot.stopButton
            .disabled =
            true;


        if (
            slot.timerInterval
        ) {

            clearInterval(
                slot
                    .timerInterval
            );

            slot.timerInterval =
                null;
        }


        slot.recorder
            .stop();
    }


    // ==========================
    // FINISH RECORDING
    // ==========================

    function finishRecording(
        slot,
        mimeType
    ) {

        stopStream(
            slot
        );


        if (
            slot.timerInterval
        ) {

            clearInterval(
                slot
                    .timerInterval
            );

            slot.timerInterval =
                null;
        }


        const blob =
            new Blob(
                slot.chunks,
                {
                    type:
                        mimeType
                }
            );


        slot.blob =
            blob;

        slot.chunks =
            [];

        slot.recorder =
            null;

        activeRecorder =
            null;


        slot.recordButton
            .disabled =
            false;

        slot.stopButton
            .disabled =
            true;


        const recordLabel =
            slot
                .recordButton
                .querySelector(
                    "span"
                );


        if (
            recordLabel
        ) {

            recordLabel
                .textContent =
                t(
                    "record_again"
                );
        }


        if (
            blob.size === 0
        ) {

            setState(
                slot,
                "has-error",
                t(
                    "empty_recording"
                )
            );


            setMessage(
                slot,
                t(
                    "no_audio_captured"
                ),
                "error"
            );


            toast(
                t(
                    "empty_recording"
                ),
                t(
                    "no_audio_from_microphone"
                ),
                "error"
            );


            updateCompletion();

            return;
        }


        const recordedFile =
            createRecordedFile(
                slot,
                blob,
                mimeType
            );


        const validationError =
            validate(
                recordedFile
            );


        if (
            validationError
        ) {

            setState(
                slot,
                "has-error",
                t(
                    "invalid_recording"
                )
            );


            setMessage(
                slot,
                validationError,
                "error"
            );


            toast(
                t(
                    "invalid_recording"
                ),
                validationError,
                "error"
            );


            updateCompletion();

            return;
        }


        const assigned =
            assignFileToInput(
                slot,
                recordedFile
            );


        if (
            !assigned
        ) {

            setState(
                slot,
                "has-error",
                t(
                    "preparation_failed"
                )
            );


            setMessage(
                slot,
                t(
                    "recording_preparation_failed"
                ),
                "error"
            );


            toast(
                t(
                    "recording_error"
                ),
                t(
                    "upload_audio_instead"
                ),
                "error"
            );


            updateCompletion();

            return;
        }


        slot.file =
            recordedFile;


        showPreview(
            slot,
            recordedFile
        );


        toast(
            t(
                "recording_complete"
            ),
            t(
                "recording_is_ready",
                {
                    id:
                        slot.id
                }
            ),
            "success"
        );
    }


    // ==========================
    // RECORD BUTTON EVENTS
    // ==========================

    Object
        .values(slots)
        .forEach(
            slot => {

                slot
                    .recordButton
                    .addEventListener(
                        "click",
                        () => {

                            startRecording(
                                slot
                            );
                        }
                    );


                slot
                    .stopButton
                    .addEventListener(
                        "click",
                        () => {

                            stopRecording(
                                slot
                            );
                        }
                    );
            }
        );


    // ==========================
    // PAGE VISIBILITY SAFETY
    // ==========================

    document
        .addEventListener(
            "visibilitychange",
            () => {

                if (
                    document.hidden &&
                    activeRecorder
                ) {

                    stopRecording(
                        activeRecorder
                    );
                }
            }
        );


    // ==========================
    // PAGE CLEANUP
    // ==========================

    window
        .addEventListener(
            "beforeunload",
            () => {

                Object
                    .values(slots)
                    .forEach(
                        slot => {

                            if (
                                slot
                                    .timerInterval
                            ) {

                                clearInterval(
                                    slot
                                        .timerInterval
                                );
                            }


                            stopStream(
                                slot
                            );


                            revokeURL(
                                slot
                            );
                        }
                    );
            }
        );


    // ==========================
    // RESULT DOM REFERENCES
    // ==========================

    const processingMessage =
        document.getElementById(
            "processing-message"
        );


    const progressElement =
        document.getElementById(
            "analysis-progress"
        );


    const progressBar =
        document.getElementById(
            "analysis-progress-bar"
        );


    const errorMessage =
        document.getElementById(
            "error-message"
        );


    const replacementList =
        document.getElementById(
            "replacement-list"
        );


    const restartButton =
        document.getElementById(
            "restart-screening"
        );


    const dismissErrorButton =
        document.getElementById(
            "dismiss-error"
        );


    const serviceStatus =
        document.getElementById(
            "service-status"
        );


    const helpButton =
        document.getElementById(
            "help-button"
        );


    const guidancePanel =
        document.getElementById(
            "guidance-panel"
        );


    const resultStatusCard =
        document.getElementById(
            "result-status-card"
        );


    const resultCategory =
        document.getElementById(
            "result-category"
        );


    const resultMessage =
        document.getElementById(
            "result-message"
        );


    const resultScore =
        document.getElementById(
            "result-score"
        );


    const resultConfidence =
        document.getElementById(
            "result-confidence"
        );


    const resultConsistency =
        document.getElementById(
            "result-consistency"
        );


    const resultAgreement =
        document.getElementById(
            "result-agreement"
        );


    const qualityResultsGrid =
        document.getElementById(
            "quality-results-grid"
        );


    const resultDisclaimer =
        document.getElementById(
            "result-disclaimer"
        );


    let progressTimer =
        null;


    // ==========================
    // PROGRESS HELPERS
    // ==========================

    function setProgress(
        percentage,
        message
    ) {

        const safeValue =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(
                        percentage
                    ) || 0
                )
            );


        if (
            progressElement
        ) {

            progressElement
                .setAttribute(
                    "aria-valuenow",
                    String(
                        safeValue
                    )
                );
        }


        if (
            progressBar
        ) {

            progressBar
                .style
                .width =
                `${safeValue}%`;
        }


        if (
            message &&
            processingMessage
        ) {

            processingMessage
                .textContent =
                message;
        }
    }


    function startProgressSimulation() {

        stopProgressSimulation();


        const stages = [
            {
                value:
                    10,

                message:
                    t(
                        "preparing_uploaded_audio"
                    )
            },

            {
                value:
                    25,

                message:
                    t(
                        "assessing_recording_quality"
                    )
            },

            {
                value:
                    42,

                message:
                    t(
                        "checking_duplicate_recordings"
                    )
            },

            {
                value:
                    60,

                message:
                    t(
                        "extracting_voice_features"
                    )
            },

            {
                value:
                    76,

                message:
                    t(
                        "running_screening_model"
                    )
            },

            {
                value:
                    90,

                message:
                    t(
                        "combining_predictions"
                    )
            }
        ];


        let stageIndex =
            0;


        setProgress(
            5,
            t(
                "preparing_audio_analysis"
            )
        );


        progressTimer =
            setInterval(
                () => {

                    if (
                        stageIndex >=
                        stages.length
                    ) {

                        stopProgressSimulation();

                        return;
                    }


                    const stage =
                        stages[
                            stageIndex
                        ];


                    setProgress(
                        stage.value,
                        stage.message
                    );


                    stageIndex++;

                },
                700
            );
    }


    function stopProgressSimulation() {

        if (
            progressTimer
        ) {

            clearInterval(
                progressTimer
            );

            progressTimer =
                null;
        }
    }


    // ==========================
    // PAGE SCROLL
    // ==========================

    function scrollToElement(
        element
    ) {

        if (
            !element
        ) {
            return;
        }


        element
            .scrollIntoView({
                behavior:
                    "smooth",

                block:
                    "start"
            });
    }


    // ==========================
    // RESPONSE EXTRACTION
    // ==========================

    function extractReplacementSlots(
        data
    ) {

        const possibleSlots = [
            data.replacement_slots,
            data.slots_to_replace,
            data.invalid_slots,
            data.duplicate_slots,
            data.details
                ?.replacement_slots
        ];


        for (
            const slotList
            of possibleSlots
        ) {

            if (
                Array.isArray(
                    slotList
                )
            ) {

                return slotList;
            }
        }


        return [];
    }

    // ==========================
    // RENDER RESULTS
    // ==========================

    function renderResults(
        data
    ) {
        
        lastErrorData = null;
        stopProgressSimulation();

        setProgress(
            100,
            t(
                "screening_completed"
            )
        );

        processingSection.hidden =
            true;

        errorSection.hidden =
            true;

        document.body.classList.remove(
            "is-processing"
        );

        toast(
            t(
                "recording_successful"
            ),
            t(
                "screening_completed_message"
            ),
            "success"
        );

        const nextURL =
            data.next_url ||
            form.dataset.nextUrl;

        if (
            nextURL
        ) {

            window.setTimeout(
                () => {

                    window.location.href =
                        nextURL;
                },
                1500
            );

            return;
        }

        resultsSection.hidden =
            false;

        resultsSection.innerHTML = `
            <section class="screening-success-panel">

                <div
                    class="success-icon"
                    aria-hidden="true"
                >
                    ✓
                </div>

                <p class="eyebrow">
                    ${
                        currentLanguage === "ur"
                            ? "آڈیو ماڈیول مکمل"
                            : "AUDIO MODULE COMPLETE"
                    }
                </p>

                <h2>
                    ${
                        currentLanguage === "ur"
                            ? "ریکارڈنگ کامیابی سے مکمل ہو گئی"
                            : "Recording completed successfully"
                    }
                </h2>

                <p>
                    ${
                        currentLanguage === "ur"
                            ? "آپ کی آڈیو ریکارڈنگز قبول کر لی گئی ہیں۔ اب آپ ماڈیول 3 پر جا سکتے ہیں۔"
                            : "Your audio recordings were accepted. You may now continue to Module 3."
                    }
                </p>

                <button
                    type="button"
                    id="continue-module-three"
                    class="button button-primary"
                >
                    ${
                        currentLanguage === "ur"
                            ? "ماڈیول 3 پر جائیں"
                            : "Continue to Module 3"
                    }
                </button>

            </section>
        `;


        document
            .getElementById(
                "continue-module-three"
            )
            ?.addEventListener(
                "click",
                () => {

                    if (
                        nextURL
                    ) {

                        window.location.href =
                            nextURL;
                    }
                }
            );


        scrollToElement(
            resultsSection
        );
    }


    // ==========================
    // ERROR INTERFACE
    // ==========================
    function refreshVisibleErrorLanguage() {

    if (
        !lastErrorData ||
        !errorSection ||
        errorSection.hidden
    ) {
        return;
    }

    const replacementSlots =
        extractReplacementSlots(
            lastErrorData
        );

    const qualityFeedback =
        Array.isArray(
            lastErrorData.quality_feedback
        )
            ? lastErrorData.quality_feedback
            : [];


    if (
        replacementSlots.length === 0
    ) {
        return;
    }


    errorMessage.textContent =
        replacementSlots.length === 1
            ? t("one_recording_attention")
            : t(
                "multiple_recordings_attention",
                {
                    count:
                        replacementSlots.length
                }
            );


    replacementList.innerHTML = "";


    replacementSlots.forEach(
        rawSlotNumber => {

            const slotNumber =
                Number(rawSlotNumber);

            const slot =
                slots[slotNumber];

            if (!slot) return;


            const feedback =
                qualityFeedback.find(
                    item =>
                        Number(item.slot) ===
                        slotNumber
                );


            const messages =
                feedback &&
                Array.isArray(
                    feedback.messages
                )
                    ? feedback.messages
                    : [];


            const friendlyMessage =
                messages.length > 0
                    ? messages
                        .map(
                            translateQualityMessage
                        )
                        .join(" ")
                    : (
                        currentLanguage === "ur"
                            ? "براہِ کرم یہ آواز دوبارہ ریکارڈ کریں۔"
                            : "Please record this sample again."
                    );


            /* Update message under recording card */
            setMessage(
                slot,
                friendlyMessage,
                "error"
            );


            /* Rebuild error summary item */
            const item =
                document.createElement(
                    "div"
                );

            item.className =
                "replacement-chip";


            const title =
                document.createElement(
                    "strong"
                );

            title.textContent =
                `${t("recording_label")} ${slotNumber}`;


            const message =
                document.createElement(
                    "span"
                );

            message.textContent =
                friendlyMessage;


            item.appendChild(title);
            item.appendChild(message);


            item.addEventListener(
                "click",
                () => {

                    scrollToElement(
                        slot.card
                    );

                    slot.recordButton.focus();
                }
            );


            replacementList.appendChild(
                item
            );
        }
    );
}
    function showScreeningError(
        data = {},
        fallbackMessage = ""
    ) {
        lastErrorData = data;
        const replacementSlots =
            extractReplacementSlots(
                data
            );

        const qualityFeedback =
            Array.isArray(
                data.quality_feedback
            )
                ? data.quality_feedback
                : [];


        if (
            data.screening_id
        ) {

            screeningId =
                data.screening_id;
        }


        processingSection.hidden =
            true;

        resultsSection.hidden =
            true;

        document.body.classList.remove(
            "is-processing"
        );


        if (
            replacementSlots.length >
            0
        ) {

            errorSection.hidden =
                false;


            errorMessage.textContent =
                replacementSlots.length ===
                1

                    ? t(
                        "one_recording_attention"
                    )

                    : t(
                        "multiple_recordings_attention",
                        {
                            count:
                                replacementSlots.length
                        }
                    );


            replacementList.hidden =
                false;

            replacementList.innerHTML =
                "";


            replacementSlots.forEach(
                rawSlotNumber => {

                    const slotNumber =
                        Number(
                            rawSlotNumber
                        );

                    const slot =
                        slots[
                            slotNumber
                        ];

                    if (
                        !slot
                    ) {
                        return;
                    }


                    const feedback =
                        qualityFeedback.find(
                            item =>
                                Number(
                                    item.slot
                                ) ===
                                slotNumber
                        );


                    const messages =
                        feedback &&
                        Array.isArray(
                            feedback.messages
                        )

                            ? feedback.messages
                            : [];


                    const friendlyMessage =
                        messages.length > 0

                            ? messages
                                .map(
                                    translateQualityMessage
                                )
                                .join(" ")

                            : (
                                currentLanguage ===
                                "ur"

                                    ? "براہِ کرم یہ آواز دوبارہ ریکارڈ کریں۔"

                                    : "Please record this sample again."
                            );


                    resetSlot(
                        slot
                    );


                    setState(
                        slot,
                        "has-error",
                        t(
                            "record_again"
                        )
                    );


                    setMessage(
                        slot,
                        friendlyMessage,
                        "error"
                    );


                    const item =
                        document.createElement(
                            "div"
                        );

                    item.className =
                        "replacement-chip";


                    const title =
                        document.createElement(
                            "strong"
                        );

                    title.textContent =
                        `${t("recording_label")} ${slotNumber}`;


                    const message =
                        document.createElement(
                            "span"
                        );

                    message.textContent =
                        friendlyMessage;


                    item.appendChild(
                        title
                    );

                    item.appendChild(
                        message
                    );


                    item.addEventListener(
                        "click",
                        () => {

                            scrollToElement(
                                slot.card
                            );

                            slot.recordButton
                                .focus();
                        }
                    );


                    replacementList
                        .appendChild(
                            item
                        );
                }
            );


            updateCompletion();


            toast(
                t(
                    "recording_needs_attention"
                ),

                replacementSlots.length ===
                1

                    ? t(
                        "review_highlighted_recording"
                    )

                    : t(
                        "review_highlighted_recordings"
                    ),

                "error"
            );


            scrollToElement(
                errorSection
            );

            return;
        }


        errorMessage.textContent =
            data.message ||
            data.error ||
            fallbackMessage ||
            t(
                "screening_could_not_complete"
            );


        replacementList.hidden =
            true;

        replacementList.innerHTML =
            "";

        errorSection.hidden =
            false;


        toast(
            t(
                "unable_complete_screening"
            ),
            errorMessage.textContent,
            "error"
        );


        scrollToElement(
            errorSection
        );
    }


    // ==========================
    // FORM VALIDATION
    // ==========================

    function validateAllRecordings() {

        let valid =
            true;


        Object
            .values(slots)
            .forEach(
                slot => {

                    const file =
                        slot
                            .fileInput
                            .files[0];

                    const error =
                        validate(
                            file
                        );


                    if (
                        error
                    ) {

                        valid =
                            false;


                        setState(
                            slot,
                            "has-error",
                            t(
                                "required"
                            )
                        );


                        setMessage(
                            slot,

                            file
                                ? error
                                : t(
                                    "add_recording_before_continue"
                                ),

                            "error"
                        );
                    }
                }
            );


        return valid;
    }


    // ==========================
    // DJANGO FORM SUBMISSION
    // ==========================

    async function submitScreening(
        event
    ) {

        event.preventDefault();


        if (
            !validateAllRecordings()
        ) {

            toast(
                t(
                    "recordings_required"
                ),
                t(
                    "add_three_valid_recordings"
                ),
                "error"
            );

            return;
        }


        submitButton.disabled =
            true;

        resultsSection.hidden =
            true;

        errorSection.hidden =
            true;

        processingSection.hidden =
            false;


        document.body.classList.add(
            "is-processing"
        );


        scrollToElement(
            processingSection
        );


        startProgressSimulation();


        try {

            const formData =
                new FormData(
                    form
                );


            if (
                screeningId
            ) {

                formData.append(
                    "screening_id",
                    screeningId
                );
            }


            const response =
                await fetch(
                    form.action,
                    {
                        method:
                            "POST",

                        body:
                            formData,

                        credentials:
                            "same-origin",

                        headers: {
                            "X-Requested-With":
                                "XMLHttpRequest"
                        }
                    }
                );


            const contentType =
                response.headers.get(
                    "content-type"
                ) || "";


            let data;


            if (
                contentType.includes(
                    "application/json"
                )
            ) {

                data =
                    await response.json();

            } else {

                data = {
                    error:
                        await response.text()
                };
            }


            stopProgressSimulation();


            if (
                !response.ok ||
                data.success === false ||
                data.ok === false
            ) {

                showScreeningError(
                    data,
                    t(
                        "server_error",
                        {
                            status:
                                response.status
                        }
                    )
                );

                return;
            }


            setProgress(
                100,
                t(
                    "analysis_complete"
                )
            );


            setTimeout(
                () => {

                    renderResults(
                        data
                    );
                },
                250
            );

        } catch (error) {

            console.error(
                "Screening request failed:",
                error
            );


            stopProgressSimulation();


            showScreeningError(
                {},
                t(
                    "server_unreachable"
                )
            );

        } finally {

            document.body.classList.remove(
                "is-processing"
            );

            updateCompletion();
        }
    }


    // ==========================
    // HEALTH CHECK
    // ==========================

    async function checkServiceHealth() {

        const healthURL =
            form.dataset.healthUrl;


        if (
            !healthURL ||
            !serviceStatus
        ) {
            return;
        }


        try {

            const response =
                await fetch(
                    healthURL,
                    {
                        method:
                            "GET",

                        credentials:
                            "same-origin",

                        headers: {
                            "X-Requested-With":
                                "XMLHttpRequest"
                        }
                    }
                );


            const data =
                await response
                    .json()
                    .catch(
                        () => ({})
                    );


            const healthy =
                response.ok &&
                data.status !== "error" &&
                data.healthy !== false &&
                data.model_loaded !== false;


            if (
                !healthy
            ) {

                throw new Error(
                    data.error ||
                    "Service unavailable"
                );
            }


            serviceStatus
                .classList
                .remove(
                    "is-error"
                );


            serviceStatus
                .classList
                .add(
                    "is-ready"
                );


            const statusText =
                serviceStatus
                    .querySelector(
                        "[data-status-text]"
                    );


            if (
                statusText
            ) {

                statusText.textContent =
                    t(
                        "service_ready"
                    );
            }

        } catch (error) {

            console.error(
                "Health check failed:",
                error
            );


            serviceStatus
                .classList
                .remove(
                    "is-ready"
                );


            serviceStatus
                .classList
                .add(
                    "is-error"
                );


            const statusText =
                serviceStatus
                    .querySelector(
                        "[data-status-text]"
                    );


            if (
                statusText
            ) {

                statusText.textContent =
                    t(
                        "service_unavailable"
                    );
            }
        }
    }


    // ==========================
    // RESET SCREENING
    // ==========================

    function resetScreening() {

        stopProgressSimulation();

        screeningId = null;
        lastErrorData = null;


        Object
            .values(slots)
            .forEach(
                slot => {

                    resetSlot(
                        slot
                    );
                }
            );


        form.reset();


        processingSection.hidden =
            true;

        resultsSection.hidden =
            true;

        errorSection.hidden =
            true;

        replacementList.hidden =
            true;

        replacementList.innerHTML =
            "";


        if (
            qualityResultsGrid
        ) {

            qualityResultsGrid.innerHTML =
                "";
        }


        setProgress(
            0,
            t(
                "preparing_audio_analysis"
            )
        );


        document.body.classList.remove(
            "is-processing"
        );


        updateCompletion();


        applyLanguage(
            currentLanguage
        );


        scrollToElement(
            form
        );

    }


    // ==========================
    // FORM EVENTS
    // ==========================

    form.addEventListener(
        "submit",
        submitScreening
    );


    if (
        restartButton
    ) {

        restartButton
            .addEventListener(
                "click",
                resetScreening
            );
    }


    if (
        dismissErrorButton
    ) {

        dismissErrorButton
            .addEventListener(
                "click",
                () => {

                    errorSection.hidden =
                        true;

                    scrollToElement(
                        form
                    );
                }
            );
    }


    if (
        helpButton
    ) {

        helpButton
            .addEventListener(
                "click",
                () => {

                    const expanded =
                        helpButton
                            .getAttribute(
                                "aria-expanded"
                            ) === "true";


                    helpButton
                        .setAttribute(
                            "aria-expanded",
                            String(
                                !expanded
                            )
                        );


                    if (
                        guidancePanel
                    ) {

                        guidancePanel
                            .classList
                            .toggle(
                                "is-highlighted",
                                !expanded
                            );


                        guidancePanel
                            .scrollIntoView({
                                behavior:
                                    "smooth",

                                block:
                                    "center"
                            });
                    }
                }
            );
    }


    // ==========================
    // INITIALIZATION
    // ==========================

    updateCompletion();


    setProgress(
        0,
        t(
            "preparing_audio_analysis"
        )
    );


    applyLanguage(
        currentLanguage
    );


    checkServiceHealth();


})();