(() => {
    "use strict";

    // ==========================
    // DOM REFERENCES
    // ==========================

    const form = document.getElementById("screening-form");

    if (!form) return;

    const submitButton = document.getElementById("submit-screening");
    const completedCount = document.getElementById("completed-count");
    const completionBadge = document.getElementById("completion-badge");

    const processingSection =
        document.getElementById("processing-section");

    const resultsSection =
        document.getElementById("results-section");

    const errorSection =
        document.getElementById("error-section");

    const toastRegion =
        document.getElementById("toast-region");

    const MAX_SECONDS =
        Number(form.dataset.maxSeconds || 6);

    const REQUIRED_RECORDINGS =
        Number(form.dataset.requiredRecordings || 3);

    const MAX_FILE_SIZE =
        25 * 1024 * 1024;

    const ALLOWED_EXTENSIONS = [
        "wav",
        "ogg",
        "webm",
        "mp4"
    ];

    // ==========================
    // RECORDING SLOTS
    // ==========================

    const slots = {};

    let activeRecorder = null;
    let screeningId = null;
    for (let i = 1; i <= REQUIRED_RECORDINGS; i++) {

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
                document.getElementById(`timer-${i}`),

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
    // UTILITIES
    // ==========================

    function escapeHTML(value) {

        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;");

    }

    function toast(title, message, type = "info") {

        const item = document.createElement("div");

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
                <strong>${escapeHTML(title)}</strong>
                <p>${escapeHTML(message)}</p>
            </div>
        `;

        toastRegion.appendChild(item);

        setTimeout(() => {

            item.classList.add("is-leaving");

            setTimeout(() => {

                item.remove();

            }, 250);

        }, 4000);

    }

    function formatTime(ms) {

        const sec =
            Math.ceil(ms / 1000);

        const min =
            Math.floor(sec / 60);

        const remain =
            sec % 60;

        return `${String(min).padStart(2, "0")}:${String(remain).padStart(2, "0")}`;

    }

    function updateCompletion() {

        let completed = 0;

        Object.values(slots).forEach(slot => {

            if (
                slot.fileInput.files.length > 0
            ) {
                completed++;
            }

        });

        completedCount.textContent =
            completed;

        completionBadge.classList.toggle(
            "is-complete",
            completed === REQUIRED_RECORDINGS
        );

        submitButton.disabled =
            completed !== REQUIRED_RECORDINGS;

    }

    function clearMessage(slot) {

        slot.message.textContent = "";

        slot.message.className =
            "field-message";

    }

    function setMessage(
        slot,
        text,
        type = "info"
    ) {

        slot.message.textContent = text;

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

        if (css)
            slot.card.classList.add(css);

        slot.state.textContent =
            label;

    }

    function revokeURL(slot) {

        if (slot.objectURL) {

            URL.revokeObjectURL(
                slot.objectURL
            );

            slot.objectURL = null;

        }

    }

    function stopStream(slot) {

        if (!slot.stream)
            return;

        slot.stream
            .getTracks()
            .forEach(track =>
                track.stop()
            );

        slot.stream = null;

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
            URL.createObjectURL(file);

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
            "Ready"
        );

        setMessage(
            slot,
            "Recording ready.",
            "success"
        );

        updateCompletion();

    }

    // ==========================
    // RESET SLOT
    // ==========================

    function resetSlot(slot) {

        revokeURL(slot);

        stopStream(slot);

        if (slot.timerInterval)
            clearInterval(slot.timerInterval);

        slot.preview.pause();

        slot.preview.removeAttribute(
            "src"
        );

        slot.preview.load();

        slot.preview.hidden = true;

        slot.filename.hidden = true;

        slot.removeButton.hidden = true;

        slot.fileInput.value = "";

        slot.file = null;

        slot.blob = null;

        slot.chunks = [];

        slot.timer.textContent =
            "00:00";

        slot.recordButton.disabled =
            false;

        slot.stopButton.disabled =
            true;

        clearMessage(slot);

        setState(
            slot,
            "",
            "Not recorded"
        );

        updateCompletion();

    }

    // ==========================
    // VALIDATION
    // ==========================

    function validate(file) {

        if (!file)
            return "No file selected.";

        const extension =
            file.name
                .split(".")
                .pop()
                .toLowerCase();

        if (
            !ALLOWED_EXTENSIONS.includes(
                extension
            )
        ) {

            return "Unsupported format.";

        }

        if (
            file.size > MAX_FILE_SIZE
        ) {

            return "Maximum upload size is 25 MB.";

        }

        if (
            file.size === 0
        ) {

            return "Empty audio file.";

        }

        return null;

    }

    // ==========================
    // FILE INPUT
    // ==========================

    Object.values(slots).forEach(slot => {

        slot.fileInput.addEventListener(
            "change",
            () => {

                const file =
                    slot.fileInput.files[0];

                const error =
                    validate(file);

                if (error) {

                    resetSlot(slot);

                    setState(
                        slot,
                        "has-error",
                        "Invalid"
                    );

                    setMessage(
                        slot,
                        error,
                        "error"
                    );

                    toast(
                        "Invalid File",
                        error,
                        "error"
                    );

                    return;

                }

                slot.file = file;

                showPreview(
                    slot,
                    file
                );

            }
        );

        slot.removeButton.addEventListener(
            "click",
            () => {

                resetSlot(slot);

                toast(
                    "Removed",
                    `Recording ${slot.id} removed.`,
                    "info"
                );

            }
        );

    });

    // ==========================
    // MEDIA RECORDER HELPERS
    // ==========================

    function getSupportedMimeType() {

        if (!window.MediaRecorder) {
            return "";
        }

        const mimeTypes = [
            "audio/webm;codecs=opus",
            "audio/webm",
            "audio/ogg;codecs=opus",
            "audio/ogg"
        ];

        for (const type of mimeTypes) {

            if (
                MediaRecorder.isTypeSupported(type)
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
            mimeType.includes("ogg")
                ? "ogg"
                : "webm";

        return new File(
            [blob],
            `recording_${slot.id}.${extension}`,
            {
                type: mimeType,
                lastModified: Date.now()
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

            transfer.items.add(file);

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

    async function startRecording(slot) {

        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {

            const message =
                "Microphone access is not supported by this browser.";

            setState(
                slot,
                "has-error",
                "Unsupported"
            );

            setMessage(
                slot,
                message,
                "error"
            );

            toast(
                "Microphone unavailable",
                message,
                "error"
            );

            return;

        }

        if (!window.MediaRecorder) {

            const message =
                "Audio recording is not supported by this browser.";

            setState(
                slot,
                "has-error",
                "Unsupported"
            );

            setMessage(
                slot,
                message,
                "error"
            );

            toast(
                "Recording unavailable",
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
                "Recording active",
                "Stop the current recording before starting another.",
                "error"
            );

            return;

        }

        resetSlot(slot);

        try {

            const stream =
                await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                        channelCount: 1
                    },
                    video: false
                });

            const audioTrack = stream.getAudioTracks()[0];

            if (audioTrack) {
                console.log(
                    "Requested microphone constraints:",
                    audioTrack.getConstraints()
                );

                console.log(
                    "Applied microphone settings:",
                    audioTrack.getSettings()
                );
            }

            const mimeType =
                getSupportedMimeType();

            const recorderOptions =
                mimeType
                    ? { mimeType }
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

            recorder.addEventListener(
                "dataavailable",
                event => {

                    if (
                        event.data &&
                        event.data.size > 0
                    ) {

                        slot.chunks.push(
                            event.data
                        );

                    }

                }
            );

            recorder.addEventListener(
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

            recorder.addEventListener(
                "error",
                event => {

                    console.error(
                        "MediaRecorder error:",
                        event.error
                    );

                    stopStream(slot);

                    if (
                        slot.timerInterval
                    ) {

                        clearInterval(
                            slot.timerInterval
                        );

                        slot.timerInterval =
                            null;

                    }

                    slot.recordButton.disabled =
                        false;

                    slot.stopButton.disabled =
                        true;

                    activeRecorder =
                        null;

                    setState(
                        slot,
                        "has-error",
                        "Recording failed"
                    );

                    setMessage(
                        slot,
                        "Recording failed. Please try again.",
                        "error"
                    );

                    toast(
                        "Recording failed",
                        "The microphone recording could not be completed.",
                        "error"
                    );

                    updateCompletion();

                }
            );

            recorder.start(200);

            setState(
                slot,
                "is-recording",
                "Recording"
            );

            clearMessage(slot);

            slot.recordButton.disabled =
                true;

            slot.stopButton.disabled =
                false;

            const recordLabel =
                slot.recordButton.querySelector(
                    "span"
                );

            if (recordLabel) {

                recordLabel.textContent =
                    "Recording";

            }

            slot.timer.textContent =
                "00:00";

            slot.timerInterval =
                setInterval(() => {

                    const elapsed =
                        Date.now() -
                        slot.startedAt;

                    slot.timer.textContent =
                        formatTime(elapsed);

                    if (
                        elapsed >=
                        MAX_SECONDS * 1000
                    ) {

                        stopRecording(slot);

                    }

                }, 100);

            updateCompletion();

        } catch (error) {

            console.error(
                "Microphone access error:",
                error
            );

            let message =
                "Microphone access failed.";

            if (
                error.name ===
                "NotAllowedError"
            ) {

                message =
                    "Microphone permission was denied. Allow access or upload an audio file.";

            } else if (
                error.name ===
                "NotFoundError"
            ) {

                message =
                    "No microphone was detected on this device.";

            } else if (
                error.name ===
                "NotReadableError"
            ) {

                message =
                    "The microphone is already in use by another application.";

            } else if (
                error.name ===
                "SecurityError"
            ) {

                message =
                    "Microphone access requires a secure HTTPS connection.";

            }

            setState(
                slot,
                "has-error",
                "Microphone error"
            );

            setMessage(
                slot,
                message,
                "error"
            );

            toast(
                "Microphone error",
                message,
                "error"
            );

        }

    }

    // ==========================
    // STOP RECORDING
    // ==========================

    function stopRecording(slot) {

        if (
            !slot.recorder ||
            slot.recorder.state ===
            "inactive"
        ) {
            return;
        }

        slot.stopButton.disabled =
            true;

        if (
            slot.timerInterval
        ) {

            clearInterval(
                slot.timerInterval
            );

            slot.timerInterval =
                null;

        }

        slot.recorder.stop();

    }

    // ==========================
    // FINISH RECORDING
    // ==========================

    function finishRecording(
        slot,
        mimeType
    ) {

        stopStream(slot);

        if (
            slot.timerInterval
        ) {

            clearInterval(
                slot.timerInterval
            );

            slot.timerInterval =
                null;

        }

        const blob =
            new Blob(
                slot.chunks,
                {
                    type: mimeType
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

        slot.recordButton.disabled =
            false;

        slot.stopButton.disabled =
            true;

        const recordLabel =
            slot.recordButton.querySelector(
                "span"
            );

        if (recordLabel) {

            recordLabel.textContent =
                "Record again";

        }

        if (
            blob.size === 0
        ) {

            setState(
                slot,
                "has-error",
                "Empty recording"
            );

            setMessage(
                slot,
                "No audio was captured. Please record again.",
                "error"
            );

            toast(
                "Empty recording",
                "No audio was captured from the microphone.",
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
            validate(recordedFile);

        if (validationError) {

            setState(
                slot,
                "has-error",
                "Invalid recording"
            );

            setMessage(
                slot,
                validationError,
                "error"
            );

            toast(
                "Invalid recording",
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

        if (!assigned) {

            setState(
                slot,
                "has-error",
                "Preparation failed"
            );

            setMessage(
                slot,
                "The recording could not be prepared for upload.",
                "error"
            );

            toast(
                "Recording error",
                "Upload an audio file instead.",
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
            "Recording complete",
            `Recording ${slot.id} is ready.`,
            "success"
        );

    }

    // ==========================
    // BUTTON EVENTS
    // ==========================

    Object.values(slots).forEach(
        slot => {

            slot.recordButton.addEventListener(
                "click",
                () => {

                    startRecording(slot);

                }
            );

            slot.stopButton.addEventListener(
                "click",
                () => {

                    stopRecording(slot);

                }
            );

        }
    );

    // ==========================
    // PAGE VISIBILITY SAFETY
    // ==========================

    document.addEventListener(
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

    window.addEventListener(
        "beforeunload",
        () => {

            Object.values(slots).forEach(
                slot => {

                    if (
                        slot.timerInterval
                    ) {

                        clearInterval(
                            slot.timerInterval
                        );

                    }

                    stopStream(slot);

                    revokeURL(slot);

                }
            );

        }
    );

    // ====================================
    // CONTINUES IN PART 3
    // ====================================

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

    let progressTimer = null;

 

    function setProgress(
        percentage,
        message
    ) {

        const safeValue =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(percentage) || 0
                )
            );

        if (progressElement) {

            progressElement.setAttribute(
                "aria-valuenow",
                String(safeValue)
            );

        }

        if (progressBar) {

            progressBar.style.width =
                `${safeValue}%`;

        }

        if (
            message &&
            processingMessage
        ) {

            processingMessage.textContent =
                message;

        }

    }

    function startProgressSimulation() {

        stopProgressSimulation();

        const stages = [
            {
                value: 10,
                message:
                    "Preparing uploaded audio."
            },
            {
                value: 25,
                message:
                    "Assessing recording quality."
            },
            {
                value: 42,
                message:
                    "Checking for duplicate recordings."
            },
            {
                value: 60,
                message:
                    "Extracting voice features."
            },
            {
                value: 76,
                message:
                    "Running the screening model."
            },
            {
                value: 90,
                message:
                    "Combining recording predictions."
            }
        ];

        let stageIndex = 0;

        setProgress(
            5,
            "Preparing audio for analysis."
        );

        progressTimer =
            setInterval(() => {

                if (
                    stageIndex >=
                    stages.length
                ) {

                    stopProgressSimulation();

                    return;

                }

                const stage =
                    stages[stageIndex];

                setProgress(
                    stage.value,
                    stage.message
                );

                stageIndex++;

            }, 700);

    }

    function stopProgressSimulation() {

        if (progressTimer) {

            clearInterval(
                progressTimer
            );

            progressTimer = null;

        }

    }

    // ==========================
    // PAGE SCROLL
    // ==========================

    function scrollToElement(
        element
    ) {

        if (!element) {
            return;
        }

        element.scrollIntoView({
            behavior: "smooth",
            block: "start"
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
            data.details?.replacement_slots
        ];

        for (
            const slotList
            of possibleSlots
        ) {

            if (
                Array.isArray(slotList)
            ) {

                return slotList;

            }

        }

        return [];

    }


  function renderResults(data) {
    stopProgressSimulation();

    setProgress(
        100,
        "Audio screening completed successfully."
    );

    processingSection.hidden = true;
    errorSection.hidden = true;

    document.body.classList.remove(
        "is-processing"
    );

    toast(
        "Recording successful",
        "Your audio screening has been completed.",
        "success"
    );

    const nextURL =
        data.next_url ||
        form.dataset.nextUrl;

    if (nextURL) {
        window.setTimeout(() => {
            window.location.href = nextURL;
        }, 1500);

        return;
    }

    // Temporary behavior until Module 3 is ready.
    resultsSection.hidden = false;

    resultsSection.innerHTML = `
        <section class="screening-success-panel">
            <div class="success-icon" aria-hidden="true">
                ✓
            </div>

            <p class="eyebrow">
                AUDIO MODULE COMPLETE
            </p>

            <h2>
                Recording completed successfully
            </h2>

            <p>
                Your audio recordings were accepted.
                You may now continue to Module 3.
            </p>

            <button
                type="button"
                id="continue-module-three"
                class="button button-primary"
            >
                Continue to Module 3
            </button>
        </section>
    `;

    document
        .getElementById("continue-module-three")
        ?.addEventListener("click", () => {
            if (nextURL) {
                window.location.href = nextURL;
            }
        });

    scrollToElement(resultsSection);
}
    // ==========================
    // ERROR INTERFACE
    // ==========================

function showScreeningError(
    data,
    fallbackMessage
) {
    const replacementSlots =
        extractReplacementSlots(data || {});
    if (data?.screening_id) {

    screeningId =
        data.screening_id;

    }

    processingSection.hidden = true;
    resultsSection.hidden = true;

    document.body.classList.remove(
        "is-processing"
    );

    if (replacementSlots.length > 0) {
        errorSection.hidden = false;

        errorMessage.textContent =
            replacementSlots.length === 1
                ? "One recording needs to be recorded again."
                : `${replacementSlots.length} recordings need to be recorded again.`;

        replacementList.hidden = false;
        replacementList.innerHTML = "";

        replacementSlots.forEach(
            rawSlotNumber => {
                const slotNumber =
                    Number(rawSlotNumber);

                const slot =
                    slots[slotNumber];

                if (!slot) {
                    return;
                }

                // Remove only the rejected recording.
                resetSlot(slot);

                setState(
                    slot,
                    "has-error",
                    "Record again"
                );

                setMessage(
                    slot,
                    "Please replace this recording.",
                    "error"
                );

                const chip =
                    document.createElement("button");

                chip.type = "button";
                chip.className =
                    "replacement-chip";

                chip.textContent =
                    `Recording ${slotNumber}`;

                chip.addEventListener(
                    "click",
                    () => {
                        scrollToElement(
                            slot.card
                        );

                        slot.recordButton.focus();
                    }
                );

                replacementList.appendChild(
                    chip
                );
            }
        );

        updateCompletion();

        toast(
            "Recording replacement required",
            replacementSlots.length === 1
                ? "Please replace the highlighted recording."
                : "Please replace the highlighted recordings.",
            "error"
        );

        scrollToElement(errorSection);
        return;
    }

    errorMessage.textContent =
        data?.message ||
        data?.error ||
        fallbackMessage ||
        "Audio screening could not be completed.";

    replacementList.hidden = true;
    errorSection.hidden = false;

    toast(
        "Unable to complete screening",
        errorMessage.textContent,
        "error"
    );

    scrollToElement(errorSection);
}
    // ==========================
    // FORM VALIDATION
    // ==========================

    function validateAllRecordings() {

        let valid =
            true;

        Object.values(
            slots
        ).forEach(slot => {

            const file =
                slot.fileInput.files[0];

            const error =
                validate(file);

            if (error) {

                valid =
                    false;

                setState(
                    slot,
                    "has-error",
                    "Required"
                );

                setMessage(
                    slot,
                    file
                        ? error
                        : (
                            "Add a recording " +
                            "before continuing."
                        ),
                    "error"
                );

            }

        });

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
                "Recordings required",
                "Add three valid recordings before continuing.",
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
                new FormData(form);
            if (screeningId) {

                formData.append(
                    "screening_id",
                    screeningId
                );

            }

            const response =
                await fetch(
                    form.action,
                    {
                        method: "POST",
                        body: formData,
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
                    (
                        "The server returned " +
                        `an error (${response.status}).`
                    )
                );

                return;

            }

            setProgress(
                100,
                "Analysis complete."
            );

            setTimeout(() => {

                renderResults(data);

            }, 250);

        } catch (error) {

            console.error(
                "Screening request failed:",
                error
            );

            stopProgressSimulation();

            showScreeningError(
                {},
                (
                    "The server could not be reached. " +
                    "Check your connection and try again."
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
                        method: "GET",
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

            if (!healthy) {

                throw new Error(
                    data.error ||
                    "Service unavailable"
                );

            }

            serviceStatus.classList.remove(
                "is-error"
            );

            serviceStatus.classList.add(
                "is-ready"
            );

            const statusText =
                serviceStatus.querySelector(
                    "[data-status-text]"
                );

            if (statusText) {

                statusText.textContent =
                    "Service ready";

            }

        } catch (error) {

            console.error(
                "Health check failed:",
                error
            );

            serviceStatus.classList.remove(
                "is-ready"
            );

            serviceStatus.classList.add(
                "is-error"
            );

            const statusText =
                serviceStatus.querySelector(
                    "[data-status-text]"
                );

            if (statusText) {

                statusText.textContent =
                    "Service unavailable";

            }

        }

    }

    // ==========================
    // RESET SCREENING
    // ==========================

    function resetScreening() {

        stopProgressSimulation();
        screeningId = null;

        Object.values(
            slots
        ).forEach(slot => {

            resetSlot(slot);

        });

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

        qualityResultsGrid.innerHTML =
            "";

        setProgress(
            0,
            "Preparing audio for analysis."
        );

        document.body.classList.remove(
            "is-processing"
        );

        updateCompletion();

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

    if (restartButton) {

        restartButton.addEventListener(
            "click",
            resetScreening
        );

    }

    if (dismissErrorButton) {

        dismissErrorButton.addEventListener(
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

    if (helpButton) {

        helpButton.addEventListener(
            "click",
            () => {

                const expanded =
                    helpButton.getAttribute(
                        "aria-expanded"
                    ) === "true";

                helpButton.setAttribute(
                    "aria-expanded",
                    String(!expanded)
                );

                if (guidancePanel) {

                    guidancePanel.classList.toggle(
                        "is-highlighted",
                        !expanded
                    );

                    guidancePanel.scrollIntoView({
                        behavior: "smooth",
                        block: "center"
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
        "Preparing audio for analysis."
    );

    checkServiceHealth();

})();