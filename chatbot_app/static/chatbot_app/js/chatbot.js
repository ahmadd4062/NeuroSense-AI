// ==========================================
// NEUROSENSE AI CHATBOT — WITH VOICE SUPPORT
// Auto-send fires on `onend` (when Chrome stops listening).
// No timers, no race conditions. Reliable on Chrome.
// ==========================================

const chatMessages = document.getElementById("chatMessages");
const messageInput = document.getElementById("messageInput");
const sendMessageBtn = document.getElementById("sendMessageBtn");
const typingIndicator = document.getElementById("typingIndicator");
const clearChatBtn = document.getElementById("clearChatBtn");
const newChatBtn = document.getElementById("newChatBtn");
const suggestionCards = document.querySelectorAll(".suggestion-card");

// Voice elements
const micBtn = document.getElementById("micBtn");
const speakerBtn = document.getElementById("speakerBtn");
const listeningBanner = document.getElementById("listeningBanner");


// ==========================================
// LANGUAGE
// ==========================================

let currentLanguage =
    typeof CURRENT_LANG !== "undefined"
        ? CURRENT_LANG
        : "en";

let isSending = false;


// ==========================================
// VOICE STATE
// ==========================================

let recognition = null;
let isListening = false;
let voiceOutputEnabled = false;
let currentSpeech = null;
let pendingSpeechText = "";   // holds the transcript waiting for onend


// ==========================================
// CSRF TOKEN
// ==========================================

function getCSRFToken() {

    const cookies = document.cookie.split(";");

    for (let cookie of cookies) {

        cookie = cookie.trim();

        if (cookie.startsWith("csrftoken=")) {

            return decodeURIComponent(
                cookie.substring("csrftoken=".length)
            );
        }
    }

    return "";
}


// ==========================================
// SCROLL / RESIZE
// ==========================================

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height =
        `${Math.min(messageInput.scrollHeight, 120)}px`;
}


// ==========================================
// WELCOME SECTION
// ==========================================

function hideWelcomeSection() {
    const welcomeSection =
        document.querySelector(".welcome-section");
    if (welcomeSection) {
        welcomeSection.style.display = "none";
    }
}

function showWelcomeSection() {
    const welcomeSection =
        document.querySelector(".welcome-section");
    if (welcomeSection) {
        welcomeSection.style.display = "";
    }
}


// ==========================================
// TYPING / SENDING STATE
// ==========================================

function setTypingState(show) {
    if (show) {
        typingIndicator.classList.add("show");
    } else {
        typingIndicator.classList.remove("show");
    }
    scrollToBottom();
}

function setSendingState(sending) {
    isSending = sending;
    sendMessageBtn.disabled = sending;
    messageInput.disabled = sending;
}


// ==========================================
// FORMAT MESSAGE
// ==========================================

function formatMessage(text) {

    if (!text) {
        return "";
    }

    const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    return escaped
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/__(.*?)__/g, "<strong>$1</strong>")
        .replace(/^\s*\*\s+(.+)$/gm, "• $1")
        .replace(/^\s*-\s+(.+)$/gm, "• $1")
        .replace(/^\s*\*\s*$/gm, "")
        .replace(/\n/g, "<br>");
}


// ==========================================
// CREATE MESSAGE
// ==========================================

function createMessageElement(role, text, emergency = false) {

    const row = document.createElement("div");
    row.classList.add("message-row", role);
    if (emergency) {
        row.classList.add("emergency");
    }

    const content = document.createElement("div");
    content.classList.add("message-content");

    const avatar = document.createElement("div");
    avatar.classList.add("message-avatar");

    const avatarIcon = document.createElement("i");

    if (role === "user") {
        avatarIcon.className = "fa-solid fa-user";
    } else {
        avatarIcon.className = "fa-solid fa-robot";
    }

    avatar.appendChild(avatarIcon);

    const bubble = document.createElement("div");
    bubble.classList.add("message-bubble");
    bubble.innerHTML = formatMessage(text);

    content.appendChild(avatar);
    content.appendChild(bubble);
    row.appendChild(content);

    return row;
}


// ==========================================
// ADD MESSAGE
// ==========================================

function addMessage(role, text, emergency = false) {
    hideWelcomeSection();
    const messageElement = createMessageElement(role, text, emergency);
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}


// ==========================================
// ERROR MESSAGE
// ==========================================

function getErrorMessage() {
    if (currentLanguage === "ur") {
        return "معذرت، ابھی جواب حاصل نہیں ہو سکا۔ براہِ کرم دوبارہ کوشش کریں۔";
    }
    return "Sorry, I couldn't get a response just now. Please try again.";
}


// ==========================================
// TEXT-TO-SPEECH
// ==========================================

function pickVoiceForLanguage(lang) {

    const voices = window.speechSynthesis.getVoices() || [];
    if (!voices.length) {
        return null;
    }

    const preferred = lang === "ur"
        ? ["ur", "hi", "en-IN", "en-GB", "en-US"]
        : ["en-US", "en-GB", "en-IN", "en"];

    for (const prefix of preferred) {
        const match = voices.find(v =>
            v.lang && v.lang.toLowerCase().startsWith(prefix.toLowerCase())
        );
        if (match) {
            return match;
        }
    }

    return voices[0];
}


function speakText(text) {

    if (!("speechSynthesis" in window)) {
        return;
    }
    if (!text) {
        return;
    }

    const clean = String(text)
        .replace(/\*\*/g, "")
        .replace(/__/g, "")
        .replace(/^\s*[•\-*]\s+/gm, "")
        .replace(/<br\s*\/?>/gi, " ")
        .replace(/<[^>]+>/g, "")
        .trim();

    if (!clean) {
        return;
    }

    window.speechSynthesis.cancel();

    const utter = new SpeechSynthesisUtterance(clean);
    utter.lang = currentLanguage === "ur" ? "ur-PK" : "en-US";
    utter.rate = 1.0;
    utter.pitch = 1.0;
    utter.volume = 1.0;

    const voice = pickVoiceForLanguage(currentLanguage);
    if (voice) {
        utter.voice = voice;
    }

    utter.onstart = function () {
        if (speakerBtn) {
            speakerBtn.classList.add("speaker-speaking");
        }
    };

    utter.onend = function () {
        if (speakerBtn) {
            speakerBtn.classList.remove("speaker-speaking");
        }
        currentSpeech = null;
    };

    utter.onerror = function () {
        if (speakerBtn) {
            speakerBtn.classList.remove("speaker-speaking");
        }
        currentSpeech = null;
    };

    currentSpeech = utter;
    window.speechSynthesis.speak(utter);
}


function toggleVoiceOutput() {

    voiceOutputEnabled = !voiceOutputEnabled;

    if (speakerBtn) {
        speakerBtn.classList.toggle("speaker-active", voiceOutputEnabled);
        const icon = speakerBtn.querySelector("i");
        if (icon) {
            icon.className = voiceOutputEnabled
                ? "fa-solid fa-volume-high"
                : "fa-solid fa-volume-xmark";
        }
    }

    if (!voiceOutputEnabled && currentSpeech) {
        window.speechSynthesis.cancel();
        currentSpeech = null;
        if (speakerBtn) {
            speakerBtn.classList.remove("speaker-speaking");
        }
    }
}


// ==========================================
// SPEECH-TO-TEXT
// ==========================================

function getSpeechRecognitionCtor() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function getRecognitionLang() {
    return currentLanguage === "ur" ? "ur-PK" : "en-US";
}


function initRecognition() {

    const Ctor = getSpeechRecognitionCtor();

    if (!Ctor) {
        return null;
    }

    const rec = new Ctor();
    rec.continuous = false;
    rec.interimResults = true;
    rec.maxAlternatives = 1;

    rec.onstart = function () {
        isListening = true;
        pendingSpeechText = "";

        if (micBtn) {
            micBtn.classList.add("mic-listening");
        }
        if (listeningBanner) {
            listeningBanner.classList.add("show");
        }
    };

    rec.onresult = function (event) {

        let finalText = "";
        let interimText = "";

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalText += transcript;
            } else {
                interimText += transcript;
            }
        }

        const displayText = (finalText || interimText).trim();

        if (messageInput && displayText) {
            messageInput.value = displayText;
            autoResizeTextarea();
        }

        // Remember the best transcript we've seen so far.
        // Prefer final text over interim text if both arrive.
        if (finalText) {
            pendingSpeechText = finalText.trim();
        } else if (interimText && !pendingSpeechText) {
            pendingSpeechText = interimText.trim();
        }
    };

    rec.onerror = function (event) {

        console.warn("Speech recognition error:", event.error);

        if (event.error === "not-allowed") {
            addMessage(
                "assistant",
                currentLanguage === "ur"
                    ? "مائیکروفون کی اجازت نہیں دی گئی۔"
                    : "Microphone permission was denied."
            );
        }

        // Reset state on error
        isListening = false;
        pendingSpeechText = "";

        if (micBtn) {
            micBtn.classList.remove("mic-listening");
        }
        if (listeningBanner) {
            listeningBanner.classList.remove("show");
        }
    };

    // ============================================================
    // FIX (Option B): Auto-send on `onend`
    // When Chrome stops listening, immediately send whatever
    // transcript we captured. No timers, no race conditions.
    // ============================================================
    rec.onend = function () {

        isListening = false;

        if (micBtn) {
            micBtn.classList.remove("mic-listening");
        }
        if (listeningBanner) {
            listeningBanner.classList.remove("show");
        }

        const textToSend = pendingSpeechText || (messageInput ? messageInput.value.trim() : "");

        pendingSpeechText = "";

        if (textToSend && !isSending) {
            sendMessage(textToSend);
        }
    };

    return rec;
}


function startListening() {

    if (isSending) {
        return;
    }

    if (!recognition) {
        recognition = initRecognition();
    }

    if (!recognition) {
        addMessage(
            "assistant",
            currentLanguage === "ur"
                ? "اس براؤزر میں آواز کی سہولت دستیاب نہیں۔"
                : "Voice input is not supported in this browser."
        );
        return;
    }

    if (currentSpeech) {
        window.speechSynthesis.cancel();
        currentSpeech = null;
    }

    pendingSpeechText = "";

    try {
        recognition.lang = getRecognitionLang();
        recognition.start();
    } catch (err) {
        console.warn("Recognition start failed:", err);
    }
}


function stopListening() {

    // Only user-initiated stop (clicking mic again, or pressing Send).
    // This will trigger `onend` which handles the auto-send.
    if (!recognition) {
        return;
    }

    try { recognition.stop(); } catch (e) {}
}


function toggleListening() {

    if (isListening) {
        // Manual stop — do NOT auto-send, just cancel.
        pendingSpeechText = "";
        stopListening();
    } else {
        startListening();
    }
}


// ==========================================
// SEND MESSAGE
// ==========================================

async function sendMessage(customMessage = null) {

    if (isSending) {
        return;
    }

    const message =
        customMessage !== null
            ? customMessage.trim()
            : messageInput.value.trim();

    if (!message) {
        return;
    }

    addMessage("user", message);

    messageInput.value = "";
    autoResizeTextarea();

    setSendingState(true);
    setTypingState(true);

    try {

        const response = await fetch(
            CHAT_API_URL,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": getCSRFToken()
                },
                body: JSON.stringify({
                    message: message,
                    language: currentLanguage
                })
            }
        );

        const data = await response.json();

        setTypingState(false);

        if (!response.ok || !data.success) {
            addMessage(
                "assistant",
                data.error || getErrorMessage()
            );
            return;
        }

        addMessage(
            "assistant",
            data.reply,
            Boolean(data.emergency)
        );

        if (voiceOutputEnabled) {
            speakText(data.reply);
        }

    } catch (error) {

        console.error("Chatbot request failed:", error);
        setTypingState(false);
        addMessage("assistant", getErrorMessage());

    } finally {
        setSendingState(false);
    }
}


// ==========================================
// CLEAR CONVERSATION
// ==========================================

function clearConversationUI() {

    const existingMessages =
        chatMessages.querySelectorAll(".message-row");

    existingMessages.forEach(message => {
        message.remove();
    });

    showWelcomeSection();

    messageInput.value = "";
    autoResizeTextarea();
    setTypingState(false);

    pendingSpeechText = "";

    if (currentSpeech) {
        window.speechSynthesis.cancel();
        currentSpeech = null;
    }
}


// ==========================================
// LANGUAGE UPDATE
// ==========================================

function updateLanguage(language) {

    currentLanguage = language;

    document.body.classList.toggle(
        "urdu-mode",
        language === "ur"
    );

    document
        .querySelectorAll("[data-en][data-ur]")
        .forEach(element => {
            element.textContent = element.dataset[language];
        });

    if (language === "ur") {
        messageInput.placeholder =
            "NeuroSense سے کچھ بھی پوچھیں...";
        messageInput.setAttribute("dir", "rtl");
    } else {
        messageInput.placeholder =
            "Ask NeuroSense anything...";
        messageInput.setAttribute("dir", "ltr");
    }
}


// ==========================================
// VOICE UI INIT
// ==========================================

function initVoiceUI() {

    const hasSTT = !!getSpeechRecognitionCtor();

    if (micBtn && !hasSTT) {
        micBtn.classList.add("voice-disabled");
    }

    const hasTTS = "speechSynthesis" in window;

    if (speakerBtn && !hasTTS) {
        speakerBtn.classList.add("voice-disabled");
        speakerBtn.disabled = true;
    }

    if ("speechSynthesis" in window) {
        window.speechSynthesis.onvoiceschanged = function () {};
    }
}


// ==========================================
// EVENTS
// ==========================================

sendMessageBtn.addEventListener("click", () => {
    // If listening, cancel pending auto-send
    if (isListening) {
        pendingSpeechText = "";
        stopListening();
    }
    sendMessage();
});

messageInput.addEventListener("input", () => {
    autoResizeTextarea();
});

messageInput.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (isListening) {
            pendingSpeechText = "";
            stopListening();
        }
        sendMessage();
    }
});

if (micBtn) {
    micBtn.addEventListener("click", toggleListening);
}

if (speakerBtn) {
    speakerBtn.addEventListener("click", toggleVoiceOutput);
}

suggestionCards.forEach(card => {
    card.addEventListener("click", () => {
        const message =
            currentLanguage === "ur"
                ? card.dataset.messageUr
                : card.dataset.messageEn;
        sendMessage(message);
    });
});

if (clearChatBtn) {
    clearChatBtn.addEventListener("click", () => {
        clearConversationUI();
    });
}

if (newChatBtn) {
    newChatBtn.addEventListener("click", () => {
        clearConversationUI();
    });
}


// ==========================================
// INIT
// ==========================================

window.addEventListener("load", () => {
    updateLanguage(currentLanguage);
    autoResizeTextarea();
    initVoiceUI();
});