const chatMessages = document.getElementById("chatMessages");
const messageInput = document.getElementById("messageInput");
const sendMessageBtn = document.getElementById("sendMessageBtn");
const typingIndicator = document.getElementById("typingIndicator");
const clearChatBtn = document.getElementById("clearChatBtn");
const newChatBtn = document.getElementById("newChatBtn");

const languageButtons = document.querySelectorAll(".language-btn");
const suggestionCards = document.querySelectorAll(".suggestion-card");

let currentLanguage = "en";
let isSending = false;

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

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height =
        `${Math.min(messageInput.scrollHeight, 120)}px`;
}

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

function formatMessage(text) {
    if (!text) {
        return "";
    }

    const escaped = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

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

function createMessageElement(
    role,
    text,
    emergency = false
) {
    const row = document.createElement("div");

    row.classList.add(
        "message-row",
        role
    );

    if (emergency) {
        row.classList.add("emergency");
    }

    const content =
        document.createElement("div");

    content.classList.add(
        "message-content"
    );

    const avatar =
        document.createElement("div");

    avatar.classList.add(
        "message-avatar"
    );

    const avatarIcon =
        document.createElement("i");

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
        document.createElement("div");

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

function getErrorMessage() {
    if (currentLanguage === "ur") {
        return "معذرت، ابھی جواب حاصل نہیں ہو سکا۔ براہِ کرم دوبارہ کوشش کریں۔";
    }

    return "Sorry, I couldn't get a response just now. Please try again.";
}

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

    addMessage(
        "user",
        message
    );

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

                    body: JSON.stringify({
                        message: message
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

        messageInput.focus();
    }
}

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

    messageInput.focus();
}

function updateLanguage(language) {
    currentLanguage =
        language;

    document.body.classList.toggle(
        "urdu-mode",
        language === "ur"
    );

    languageButtons.forEach(
        button => {
            button.classList.toggle(
                "active",
                button.dataset.language ===
                    language
            );
        }
    );

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

    if (language === "ur") {
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

sendMessageBtn.addEventListener(
    "click",
    () => {
        sendMessage();
    }
);

messageInput.addEventListener(
    "input",
    () => {
        autoResizeTextarea();
    }
);

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

suggestionCards.forEach(
    card => {
        card.addEventListener(
            "click",
            () => {
                const message =
                    currentLanguage ===
                    "ur"
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

languageButtons.forEach(
    button => {
        button.addEventListener(
            "click",
            () => {
                updateLanguage(
                    button.dataset
                        .language
                );
            }
        );
    }
);

clearChatBtn.addEventListener(
    "click",
    () => {
        clearConversationUI();
    }
);

newChatBtn.addEventListener(
    "click",
    () => {
        clearConversationUI();
    }
);

window.addEventListener(
    "load",
    () => {
        updateLanguage("en");

        autoResizeTextarea();

        messageInput.focus();
    }
);