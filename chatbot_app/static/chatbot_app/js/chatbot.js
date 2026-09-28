// ==========================================
// NEUROSENSE AI CHATBOT
// ==========================================

const chatMessages =
    document.getElementById("chatMessages");

const messageInput =
    document.getElementById("messageInput");

const sendMessageBtn =
    document.getElementById("sendMessageBtn");

const typingIndicator =
    document.getElementById("typingIndicator");

const clearChatBtn =
    document.getElementById("clearChatBtn");

const newChatBtn =
    document.getElementById("newChatBtn");

const suggestionCards =
    document.querySelectorAll(".suggestion-card");


// ==========================================
// LANGUAGE
// ==========================================

let currentLanguage =
    typeof CURRENT_LANG !== "undefined"
        ? CURRENT_LANG
        : "en";

let isSending = false;


// ==========================================
// CSRF TOKEN
// ==========================================

function getCSRFToken() {

    const cookies =
        document.cookie.split(";");

    for (let cookie of cookies) {

        cookie = cookie.trim();

        if (cookie.startsWith("csrftoken=")) {

            return decodeURIComponent(
                cookie.substring(
                    "csrftoken=".length
                )
            );
        }
    }

    return "";
}


// ==========================================
// SCROLL
// ==========================================

function scrollToBottom() {

    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


// ==========================================
// TEXTAREA RESIZE
// ==========================================

function autoResizeTextarea() {

    messageInput.style.height = "auto";

    messageInput.style.height =
        `${Math.min(
            messageInput.scrollHeight,
            120
        )}px`;
}


// ==========================================
// WELCOME SECTION
// ==========================================

function hideWelcomeSection() {

    const welcomeSection =
        document.querySelector(
            ".welcome-section"
        );

    if (welcomeSection) {

        welcomeSection.style.display =
            "none";
    }
}


function showWelcomeSection() {

    const welcomeSection =
        document.querySelector(
            ".welcome-section"
        );

    if (welcomeSection) {

        welcomeSection.style.display =
            "";
    }
}


// ==========================================
// TYPING STATE
// ==========================================

function setTypingState(show) {

    if (show) {

        typingIndicator.classList.add(
            "show"
        );

    } else {

        typingIndicator.classList.remove(
            "show"
        );
    }

    scrollToBottom();
}


// ==========================================
// SENDING STATE
// ==========================================

function setSendingState(sending) {

    isSending = sending;

    sendMessageBtn.disabled =
        sending;

    messageInput.disabled =
        sending;
}


// ==========================================
// FORMAT MESSAGE
// ==========================================

function formatMessage(text) {

    if (!text) {
        return "";
    }

    const escaped =
        text
            .replace(
                /&/g,
                "&amp;"
            )
            .replace(
                /</g,
                "&lt;"
            )
            .replace(
                />/g,
                "&gt;"
            );

    return escaped

        .replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        )

        .replace(
            /__(.*?)__/g,
            "<strong>$1</strong>"
        )

        .replace(
            /^\s*\*\s+(.+)$/gm,
            "• $1"
        )

        .replace(
            /^\s*-\s+(.+)$/gm,
            "• $1"
        )

        .replace(
            /^\s*\*\s*$/gm,
            ""
        )

        .replace(
            /\n/g,
            "<br>"
        );
}


// ==========================================
// CREATE MESSAGE
// ==========================================

function createMessageElement(
    role,
    text,
    emergency = false
) {

    const row =
        document.createElement(
            "div"
        );

    row.classList.add(
        "message-row",
        role
    );

    if (emergency) {

        row.classList.add(
            "emergency"
        );
    }


    const content =
        document.createElement(
            "div"
        );

    content.classList.add(
        "message-content"
    );


    const avatar =
        document.createElement(
            "div"
        );

    avatar.classList.add(
        "message-avatar"
    );


    const avatarIcon =
        document.createElement(
            "i"
        );


    if (role === "user") {

        avatarIcon.className =
            "fa-solid fa-user";

    } else {

        avatarIcon.className =
            "fa-solid fa-robot";
    }


    avatar.appendChild(
        avatarIcon
    );


    const bubble =
        document.createElement(
            "div"
        );

    bubble.classList.add(
        "message-bubble"
    );

    bubble.innerHTML =
        formatMessage(text);


    content.appendChild(
        avatar
    );

    content.appendChild(
        bubble
    );


    row.appendChild(
        content
    );


    return row;
}


// ==========================================
// ADD MESSAGE
// ==========================================

function addMessage(
    role,
    text,
    emergency = false
) {

    hideWelcomeSection();


    const messageElement =
        createMessageElement(
            role,
            text,
            emergency
        );


    chatMessages.appendChild(
        messageElement
    );


    scrollToBottom();
}


// ==========================================
// ERROR MESSAGE
// ==========================================

function getErrorMessage() {

    if (
        currentLanguage === "ur"
    ) {

        return "معذرت، ابھی جواب حاصل نہیں ہو سکا۔ براہِ کرم دوبارہ کوشش کریں۔";
    }


    return "Sorry, I couldn't get a response just now. Please try again.";
}


// ==========================================
// SEND MESSAGE
// ==========================================

async function sendMessage(
    customMessage = null
) {

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


    // Add user message
    addMessage(
        "user",
        message
    );


    // Clear input
    messageInput.value = "";

    autoResizeTextarea();


    setSendingState(true);

    setTypingState(true);


    try {

        const response =
            await fetch(
                CHAT_API_URL,
                {
                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "X-CSRFToken":
                            getCSRFToken()
                    },

                    body:
                        JSON.stringify({
                            message:
                                message,

                            language:
                                currentLanguage
                        })
                }
            );


        const data =
            await response.json();


        setTypingState(false);


        if (
            !response.ok ||
            !data.success
        ) {

            addMessage(
                "assistant",
                data.error ||
                    getErrorMessage()
            );

            return;
        }


        addMessage(
            "assistant",
            data.reply,
            Boolean(
                data.emergency
            )
        );


    } catch (error) {

        console.error(
            "Chatbot request failed:",
            error
        );


        setTypingState(false);


        addMessage(
            "assistant",
            getErrorMessage()
        );


    } finally {

        setSendingState(false);

        // IMPORTANT:
        // Do NOT automatically focus the textarea.
        // This prevents the cursor from appearing
        // unexpectedly after page reload/send.
    }
}


// ==========================================
// CLEAR CONVERSATION
// ==========================================

function clearConversationUI() {

    const existingMessages =
        chatMessages.querySelectorAll(
            ".message-row"
        );


    existingMessages.forEach(
        message => {

            message.remove();
        }
    );


    showWelcomeSection();


    messageInput.value = "";

    autoResizeTextarea();

    setTypingState(false);

    // Intentionally no automatic focus.
}


// ==========================================
// LANGUAGE UPDATE
// ==========================================

function updateLanguage(
    language
) {

    currentLanguage =
        language;


    document.body.classList.toggle(
        "urdu-mode",
        language === "ur"
    );


    // Update all bilingual elements

    document
        .querySelectorAll(
            "[data-en][data-ur]"
        )
        .forEach(
            element => {

                element.textContent =
                    element.dataset[
                        language
                    ];
            }
        );


    // Update placeholder

    if (
        language === "ur"
    ) {

        messageInput.placeholder =
            "NeuroSense سے کچھ بھی پوچھیں...";


        messageInput.setAttribute(
            "dir",
            "rtl"
        );


    } else {

        messageInput.placeholder =
            "Ask NeuroSense anything...";


        messageInput.setAttribute(
            "dir",
            "ltr"
        );
    }
}


// ==========================================
// SEND BUTTON
// ==========================================

sendMessageBtn.addEventListener(
    "click",
    () => {

        sendMessage();
    }
);


// ==========================================
// TEXTAREA INPUT
// ==========================================

messageInput.addEventListener(
    "input",
    () => {

        autoResizeTextarea();
    }
);


// ==========================================
// ENTER TO SEND
// ==========================================

messageInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }
    }
);


// ==========================================
// SUGGESTION CARDS
// ==========================================

suggestionCards.forEach(
    card => {

        card.addEventListener(
            "click",
            () => {

                const message =
                    currentLanguage === "ur"
                        ? card.dataset
                            .messageUr
                        : card.dataset
                            .messageEn;


                sendMessage(
                    message
                );
            }
        );
    }
);


// ==========================================
// CLEAR CHAT BUTTON
// ==========================================

if (clearChatBtn) {

    clearChatBtn.addEventListener(
        "click",
        () => {

            clearConversationUI();
        }
    );
}


// ==========================================
// NEW CHAT BUTTON
// ==========================================

if (newChatBtn) {

    newChatBtn.addEventListener(
        "click",
        () => {

            clearConversationUI();
        }
    );
}


// ==========================================
// INITIALIZATION
// ==========================================

window.addEventListener(
    "load",
    () => {

        // Apply current language
        updateLanguage(
            currentLanguage
        );


        // Set textarea height
        autoResizeTextarea();


        // IMPORTANT:
        // No messageInput.focus() here.
        //
        // This prevents the cursor from automatically
        // appearing inside the textbox after reload.
    }
);