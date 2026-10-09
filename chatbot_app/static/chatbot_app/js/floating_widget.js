/* ==========================================================
   NEUROSENSE FLOATING CHATBOT WIDGET — WITH VOICE CHAT
   ----------------------------------------------------------
   Persistent chat widget on every page (except /chatbot/).
   Voice features: mic (speech-to-text) + speaker (text-to-speech).

   Auto-send behavior:
     • Primary: fires on `onend` (Chrome finishing listening).
     • Speed-up: fires ~400ms after a final result arrives, so
       the user doesn't wait the full ~5s Chrome silence window.
     • Manual: clicking mic while listening = "send now".
   No idle timers on interim text. No race conditions.
   ========================================================== */

(function () {
    'use strict';

    // ==========================================================
    // CONFIGURATION
    // ==========================================================
    const API_URL = '/chatbot/api/chat/';
    const FINAL_RESULT_SETTLE_MS = 400;   // after isFinal, wait this long then send

    let launcher, panel, messagesArea, inputEl, sendBtn, closeBtn, clearBtn,
        typingEl, suggestionsWrap, welcomeEl, micBtn, speakerBtn, listeningBanner;

    let currentLanguage = 'en';
    let isSending = false;
    let initialized = false;

    // Voice state
    let recognition = null;
    let isListening = false;
    let voiceOutputEnabled = false;
    let currentSpeech = null;
    let pendingSpeechText = "";       // holds the transcript waiting for send
    let finalResultTimer = null;      // short settle timer after isFinal

    // ==========================================================
    // CSRF TOKEN
    // ==========================================================
    function getCSRFToken() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith('csrftoken=')) {
                return decodeURIComponent(cookie.substring('csrftoken='.length));
            }
        }
        return '';
    }

    // ==========================================================
    // SCROLL / RESIZE
    // ==========================================================
    function scrollToBottom() {
        if (!messagesArea) return;
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }

    function autoResizeInput() {
        if (!inputEl) return;
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + 'px';
    }

    // ==========================================================
    // OPEN / CLOSE
    // ==========================================================
    function openWidget() {
        if (!panel || !launcher) return;
        panel.classList.add('nsw-open');
        launcher.classList.add('nsw-hidden');
        if (window.innerWidth > 480 && inputEl) {
            setTimeout(() => inputEl.focus(), 250);
        }
        try { sessionStorage.setItem('nsw_open', '1'); } catch (e) {}
    }

    function closeWidget() {
        if (!panel || !launcher) return;

        // Cancel any pending auto-send before stopping recognition
        if (isListening) {
            pendingSpeechText = "";
            clearFinalResultTimer();
            stopListening();
        }
        if (currentSpeech) {
            window.speechSynthesis.cancel();
            currentSpeech = null;
        }

        panel.classList.remove('nsw-open');
        launcher.classList.remove('nsw-hidden');
        try { sessionStorage.removeItem('nsw_open'); } catch (e) {}
    }

    // ==========================================================
    // LANGUAGE
    // ==========================================================
    function getText(key) {
        const dict = {
            en: {
                title: 'NeuroSense Assistant',
                subtitle: 'AI-powered support',
                placeholder: 'Ask NeuroSense anything...',
                welcome_title: 'Hello!',
                welcome_body: 'Ask me anything about Parkinson\u2019s, your screening results, or NeuroSense.',
                error: "Sorry, I couldn't get a response just now. Please try again.",
                clear: 'New chat',
                close: 'Close',
                mic_start: 'Start voice input',
                mic_stop: 'Stop listening and send',
                mic_unsupported: 'Voice input is not supported in this browser',
                speaker_on: 'Turn off voice output',
                speaker_off: 'Turn on voice output',
                listening: 'Listening... speak now',
            },
            ur: {
                title: 'نیوروسینس اسسٹنٹ',
                subtitle: 'اے آئی سے چلنے والی معاونت',
                placeholder: 'NeuroSense سے کچھ پوچھیں...',
                welcome_title: 'السلام علیکم!',
                welcome_body: 'پارکنسن، اپنے اسکریننگ نتائج، یا نیوروسینس کے بارے میں کچھ بھی پوچھیں۔',
                error: 'معذرت، ابھی جواب حاصل نہیں ہو سکا۔ براہِ کرم دوبارہ کوشش کریں۔',
                clear: 'نئی گفتگو',
                close: 'بند کریں',
                mic_start: 'آواز سے لکھیں',
                mic_stop: 'سننا بند کریں اور بھیجیں',
                mic_unsupported: 'اس براؤزر میں آواز کی سہولت دستیاب نہیں',
                speaker_on: 'آواز بند کریں',
                speaker_off: 'آواز آن کریں',
                listening: 'سن رہے ہیں... بولیں',
            },
        };
        return (dict[currentLanguage] || dict.en)[key];
    }

    function applyLanguage(lang) {
        currentLanguage = (lang === 'ur') ? 'ur' : 'en';
        if (!panel) return;
        panel.setAttribute('dir', currentLanguage === 'ur' ? 'rtl' : 'ltr');
        if (inputEl) {
            inputEl.placeholder = getText('placeholder');
            inputEl.setAttribute('dir', currentLanguage === 'ur' ? 'rtl' : 'ltr');
        }
        const titleEl = panel.querySelector('.nsw-header-text h3');
        const subtitleEl = panel.querySelector('.nsw-header-text p');
        if (titleEl) titleEl.textContent = getText('title');
        if (subtitleEl) subtitleEl.textContent = getText('subtitle');
        if (welcomeEl) {
            const wTitle = welcomeEl.querySelector('h4');
            const wBody = welcomeEl.querySelector('p');
            if (wTitle) wTitle.textContent = getText('welcome_title');
            if (wBody) wBody.textContent = getText('welcome_body');
        }
        if (clearBtn) clearBtn.setAttribute('title', getText('clear'));
        if (closeBtn) closeBtn.setAttribute('title', getText('close'));
        if (micBtn) micBtn.setAttribute('title', isListening ? getText('mic_stop') : getText('mic_start'));
        if (speakerBtn) speakerBtn.setAttribute('title', voiceOutputEnabled ? getText('speaker_on') : getText('speaker_off'));
        const bannerText = panel.querySelector('.nsw-listening-banner span.nsw-listening-text');
        if (bannerText) bannerText.textContent = getText('listening');
    }

    // ==========================================================
    // MESSAGE FORMATTING
    // ==========================================================
    function formatMessage(text) {
        if (!text) return '';
        const escaped = String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        return escaped
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/__(.*?)__/g, '<strong>$1</strong>')
            .replace(/^\s*\*\s+(.+)$/gm, '• $1')
            .replace(/^\s*-\s+(.+)$/gm, '• $1')
            .replace(/\n/g, '<br>');
    }

    function hideWelcome() {
        if (welcomeEl) welcomeEl.style.display = 'none';
        if (suggestionsWrap) suggestionsWrap.style.display = 'none';
    }
    function showWelcome() {
        if (welcomeEl) welcomeEl.style.display = '';
        if (suggestionsWrap) suggestionsWrap.style.display = '';
    }

    function buildMessage(role, text, emergency) {
        const row = document.createElement('div');
        row.className = 'nsw-message-row nsw-' + role;
        if (emergency) row.classList.add('nsw-emergency');
        const bubble = document.createElement('div');
        bubble.className = 'nsw-message-bubble';
        bubble.innerHTML = formatMessage(text);
        row.appendChild(bubble);
        return row;
    }

    function addMessage(role, text, emergency) {
        hideWelcome();
        if (!messagesArea) return;
        messagesArea.appendChild(buildMessage(role, text, emergency));
        scrollToBottom();
    }

    function setTyping(show) {
        if (!typingEl) return;
        typingEl.classList.toggle('nsw-show', !!show);
        scrollToBottom();
    }

    function setSending(sending) {
        isSending = sending;
        if (sendBtn) sendBtn.disabled = sending;
        if (inputEl) inputEl.disabled = sending;
    }

    // ==========================================================
    // SEND MESSAGE
    // ==========================================================
    async function sendMessage(customText) {
        if (isSending) return;
        const message = (customText !== undefined && customText !== null)
            ? String(customText).trim()
            : (inputEl ? inputEl.value.trim() : '');
        if (!message) return;

        addMessage('user', message);

        // Only clear the input if the user typed it there (no explicit customText).
        // Voice auto-send clears the input itself in onend / final-result timer.
        if (inputEl && customText === undefined) {
            inputEl.value = '';
            autoResizeInput();
        }

        setSending(true);
        setTyping(true);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCSRFToken(),
                },
                body: JSON.stringify({
                    message: message,
                    language: currentLanguage,
                }),
            });
            const data = await response.json();
            setTyping(false);

            if (!response.ok || !data.success) {
                addMessage('assistant', data.error || getText('error'));
                return;
            }
            addMessage('assistant', data.reply, !!data.emergency);

            if (voiceOutputEnabled) speakText(data.reply);

        } catch (err) {
            console.error('Floating widget error:', err);
            setTyping(false);
            addMessage('assistant', getText('error'));
        } finally {
            setSending(false);
        }
    }

    function clearConversation() {
        if (!messagesArea) return;

        // Cancel any pending auto-send before clearing
        if (isListening) {
            pendingSpeechText = "";
            clearFinalResultTimer();
            stopListening();
        }

        messagesArea.querySelectorAll('.nsw-message-row').forEach(el => el.remove());
        if (inputEl) { inputEl.value = ''; autoResizeInput(); }
        setTyping(false);
        showWelcome();

        if (currentSpeech) {
            window.speechSynthesis.cancel();
            currentSpeech = null;
        }
    }

    // ==========================================================
    // SPEECH-TO-TEXT
    // ==========================================================
    function getSpeechRecognitionCtor() {
        return window.SpeechRecognition || window.webkitSpeechRecognition || null;
    }
    function getRecognitionLang() {
        return currentLanguage === 'ur' ? 'ur-PK' : 'en-US';
    }

    function clearFinalResultTimer() {
        if (finalResultTimer) {
            clearTimeout(finalResultTimer);
            finalResultTimer = null;
        }
    }

    // Clears the input box (used by all voice auto-send paths)
    function clearInputBox() {
        if (inputEl) {
            inputEl.value = '';
            autoResizeInput();
        }
    }

    function initRecognition() {
        const Ctor = getSpeechRecognitionCtor();
        if (!Ctor) return null;

        const rec = new Ctor();
        rec.continuous = false;
        rec.interimResults = true;
        rec.maxAlternatives = 1;

        rec.onstart = function () {
            isListening = true;
            pendingSpeechText = "";
            clearFinalResultTimer();

            if (micBtn) {
                micBtn.classList.add('nsw-mic-listening');
                micBtn.setAttribute('title', getText('mic_stop'));
            }
            if (listeningBanner) listeningBanner.classList.add('nsw-show');
        };

        rec.onresult = function (event) {
            let finalText = '';
            let interimText = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) finalText += transcript;
                else interimText += transcript;
            }

            const displayText = (finalText || interimText).trim();

            if (inputEl && displayText) {
                inputEl.value = displayText;
                autoResizeInput();
            }

            if (finalText) {
                pendingSpeechText = finalText.trim();

                // SPEED-UP: final result arrived — send after a short settle window.
                // This usually fires ~1–2s after the user stops talking, much
                // faster than Chrome's ~5s silence-driven `onend`.
                clearFinalResultTimer();
                finalResultTimer = setTimeout(() => {
                    finalResultTimer = null;

                    const text = pendingSpeechText;
                    pendingSpeechText = '';

                    // Clear input now that we've captured the text
                    clearInputBox();

                    try { recognition.stop(); } catch (e) {}

                    if (text && !isSending) {
                        sendMessage(text);
                    }
                }, FINAL_RESULT_SETTLE_MS);

            } else if (interimText && !pendingSpeechText) {
                pendingSpeechText = interimText.trim();
            }
        };

        rec.onerror = function (event) {
            console.warn('[NeuroSense] Speech recognition error:', event.error);
            clearFinalResultTimer();

            if (event.error === 'not-allowed') {
                addMessage('assistant', currentLanguage === 'ur'
                    ? 'مائیکروفون کی اجازت نہیں دی گئی۔'
                    : 'Microphone permission was denied.');
            }

            isListening = false;
            pendingSpeechText = "";

            if (micBtn) {
                micBtn.classList.remove('nsw-mic-listening');
                micBtn.setAttribute('title', getText('mic_start'));
            }
            if (listeningBanner) listeningBanner.classList.remove('nsw-show');
        };

        // ============================================================
        // Fallback auto-send on `onend`.
        // If the isFinal timer already sent, pendingSpeechText is empty
        // and this is a no-op. Otherwise it ensures nothing is lost when
        // Chrome never fires an early `isFinal`.
        // ============================================================
        rec.onend = function () {
            isListening = false;

            if (micBtn) {
                micBtn.classList.remove('nsw-mic-listening');
                micBtn.setAttribute('title', getText('mic_start'));
            }
            if (listeningBanner) listeningBanner.classList.remove('nsw-show');

            // If the isFinal timer is about to fire, let it handle the send.
            if (finalResultTimer) {
                return;
            }

            const textToSend = pendingSpeechText || (inputEl ? inputEl.value.trim() : '');
            pendingSpeechText = '';

            // Clear input now that we've captured the text
            clearInputBox();

            if (textToSend && !isSending) {
                sendMessage(textToSend);
            }
        };

        return rec;
    }

    function startListening() {
        if (isSending) return;
        if (!recognition) recognition = initRecognition();
        if (!recognition) {
            addMessage('assistant', getText('mic_unsupported'));
            return;
        }
        if (currentSpeech) {
            window.speechSynthesis.cancel();
            currentSpeech = null;
        }

        pendingSpeechText = "";
        clearFinalResultTimer();

        try {
            recognition.lang = getRecognitionLang();
            recognition.start();
        } catch (err) {
            console.warn('[NeuroSense] Recognition start failed:', err);
        }
    }

    function stopListening() {
        // Only user-initiated stop (mic click, Send, Enter, close, clear).
        // This triggers `onend`, which handles the auto-send.
        if (!recognition) return;
        try { recognition.stop(); } catch (e) {}
    }

    function toggleListening() {
        if (isListening) {
            // User clicked mic while listening = "I'm done, send now".
            // Do NOT clear pendingSpeechText — let onend send it.
            stopListening();
        } else {
            startListening();
        }
    }

    // ==========================================================
    // TEXT-TO-SPEECH
    // ==========================================================
    function pickVoiceForLanguage(lang) {
        const voices = window.speechSynthesis.getVoices() || [];
        if (!voices.length) return null;
        const preferred = lang === 'ur'
            ? ['ur', 'hi', 'en-IN', 'en-GB', 'en-US']
            : ['en-US', 'en-GB', 'en-IN', 'en'];
        for (const prefix of preferred) {
            const match = voices.find(v => v.lang && v.lang.toLowerCase().startsWith(prefix.toLowerCase()));
            if (match) return match;
        }
        return voices[0];
    }

    function speakText(text) {
        if (!('speechSynthesis' in window)) return;
        if (!text) return;
        const clean = String(text)
            .replace(/\*\*/g, '')
            .replace(/__/g, '')
            .replace(/^\s*[•\-*]\s+/gm, '')
            .replace(/<br\s*\/?>/gi, ' ')
            .replace(/<[^>]+>/g, '')
            .trim();
        if (!clean) return;

        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(clean);
        utter.lang = currentLanguage === 'ur' ? 'ur-PK' : 'en-US';
        utter.rate = 1.0;
        utter.pitch = 1.0;
        utter.volume = 1.0;
        const voice = pickVoiceForLanguage(currentLanguage);
        if (voice) utter.voice = voice;

        utter.onstart = function () {
            if (speakerBtn) speakerBtn.classList.add('nsw-speaker-speaking');
        };
        utter.onend = function () {
            if (speakerBtn) speakerBtn.classList.remove('nsw-speaker-speaking');
            currentSpeech = null;
        };
        utter.onerror = function () {
            if (speakerBtn) speakerBtn.classList.remove('nsw-speaker-speaking');
            currentSpeech = null;
        };

        currentSpeech = utter;
        window.speechSynthesis.speak(utter);
    }

    function toggleVoiceOutput() {
        voiceOutputEnabled = !voiceOutputEnabled;
        if (speakerBtn) {
            speakerBtn.classList.toggle('nsw-speaker-active', voiceOutputEnabled);
            speakerBtn.setAttribute('title', voiceOutputEnabled ? getText('speaker_on') : getText('speaker_off'));
            const icon = speakerBtn.querySelector('i');
            if (icon) {
                icon.className = voiceOutputEnabled
                    ? 'fa-solid fa-volume-high'
                    : 'fa-solid fa-volume-xmark';
            }
        }
        if (!voiceOutputEnabled && currentSpeech) {
            window.speechSynthesis.cancel();
            currentSpeech = null;
            if (speakerBtn) speakerBtn.classList.remove('nsw-speaker-speaking');
        }
    }

    // ==========================================================
    // EVENTS
    // ==========================================================
    function bindEvents() {
        launcher.addEventListener('click', openWidget);
        closeBtn.addEventListener('click', closeWidget);
        clearBtn.addEventListener('click', clearConversation);

        sendBtn.addEventListener('click', () => {
            // If listening, cancel pending auto-send and stop recognition.
            // The user's typed text (already in the input) will be sent below.
            if (isListening) {
                pendingSpeechText = "";
                clearFinalResultTimer();
                stopListening();
            }
            sendMessage();
        });

        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (isListening) {
                    pendingSpeechText = "";
                    clearFinalResultTimer();
                    stopListening();
                }
                sendMessage();
            }
        });

        inputEl.addEventListener('input', autoResizeInput);

        panel.querySelectorAll('.nsw-suggestion').forEach(chip => {
            chip.addEventListener('click', () => {
                const en = chip.dataset.messageEn || '';
                const ur = chip.dataset.messageUr || en;
                const text = currentLanguage === 'ur' ? ur : en;
                sendMessage(text);
            });
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && panel.classList.contains('nsw-open')) {
                closeWidget();
            }
        });

        if (micBtn) micBtn.addEventListener('click', toggleListening);
        if (speakerBtn) speakerBtn.addEventListener('click', toggleVoiceOutput);
    }

    function initVoiceUI() {
        const hasSTT = !!getSpeechRecognitionCtor();
        if (micBtn && !hasSTT) {
            micBtn.classList.add('nsw-voice-disabled');
            micBtn.setAttribute('title', getText('mic_unsupported'));
        }
        const hasTTS = 'speechSynthesis' in window;
        if (speakerBtn && !hasTTS) {
            speakerBtn.classList.add('nsw-voice-disabled');
            speakerBtn.disabled = true;
        }
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = function () {};
        }
    }

    function init() {
        if (initialized) return;
        initialized = true;

        launcher = document.getElementById('nswLauncher');
        panel = document.getElementById('nswPanel');
        messagesArea = document.getElementById('nswMessages');
        inputEl = document.getElementById('nswInput');
        sendBtn = document.getElementById('nswSend');
        closeBtn = document.getElementById('nswClose');
        clearBtn = document.getElementById('nswClear');
        typingEl = document.getElementById('nswTyping');
        suggestionsWrap = document.getElementById('nswSuggestions');
        welcomeEl = document.getElementById('nswWelcome');
        micBtn = document.getElementById('nswMic');
        speakerBtn = document.getElementById('nswSpeaker');
        listeningBanner = document.getElementById('nswListeningBanner');

        if (!launcher || !panel || !messagesArea || !inputEl || !sendBtn) {
            console.warn('[NeuroSense] Floating chatbot: required DOM nodes not found.');
            return;
        }

        const lang = launcher.dataset.lang || 'en';
        applyLanguage(lang);
        bindEvents();
        initVoiceUI();

        try {
            if (sessionStorage.getItem('nsw_open') === '1') openWidget();
        } catch (e) {}

        autoResizeInput();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();