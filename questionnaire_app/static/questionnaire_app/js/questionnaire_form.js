document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("questionnaireForm");
    const slides = document.querySelectorAll(".question-slide");

    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const submitBtn = document.getElementById("submitBtn");

    const questionNumber = document.getElementById("questionNumber");
    const totalQuestions = document.getElementById("totalQuestions");
    const progressPercent = document.getElementById("progressPercent");
    const progressFill = document.getElementById("progressFill");

    let currentSlide = 0;
    const totalSlides = slides.length;

    totalQuestions.textContent = totalSlides;

    function showSlide(index) {
        slides.forEach((slide, slideIndex) => {
            slide.classList.toggle("active", slideIndex === index);
        });

        questionNumber.textContent = index + 1;

        const progress = Math.round(
            ((index + 1) / totalSlides) * 100
        );

        progressPercent.textContent = `${progress}%`;
        progressFill.style.width = `${progress}%`;

        prevBtn.disabled = index === 0;

        if (index === totalSlides - 1) {
            nextBtn.style.display = "none";
            submitBtn.style.display = "inline-flex";
        } else {
            nextBtn.style.display = "inline-flex";
            submitBtn.style.display = "none";
        }

        
    }

    function validateCurrentSlide() {
        const current = slides[currentSlide];

        const numberInput = current.querySelector(
            'input[type="number"]'
        );

        if (numberInput) {
            const value = numberInput.value.trim();

            if (value === "") {
                numberInput.reportValidity();
                return false;
            }

            const age = Number(value);
            const min = Number(numberInput.min);
            const max = Number(numberInput.max);

            if (
                Number.isNaN(age) ||
                age < min ||
                age > max
            ) {
                numberInput.reportValidity();
                return false;
            }

            return true;
        }

        const radioInputs = current.querySelectorAll(
            'input[type="radio"]'
        );

        if (radioInputs.length > 0) {
            const selected = current.querySelector(
                'input[type="radio"]:checked'
            );

            if (!selected) {
                showValidationMessage(current);
                return false;
            }
        }

        return true;
    }

    function showValidationMessage(slide) {
        let message = slide.querySelector(
            ".question-validation-message"
        );

        if (!message) {
            message = document.createElement("p");

            message.className =
                "question-validation-message";

            message.textContent =
                "Please select an option before continuing.";

            message.style.color = "#dc2626";
            message.style.fontSize = "14px";
            message.style.fontWeight = "500";
            message.style.marginTop = "15px";

            const options =
                slide.querySelector(".options-grid") ||
                slide.querySelector(".options");

            if (options) {
                options.insertAdjacentElement(
                    "afterend",
                    message
                );
            } else {
                slide.appendChild(message);
            }
        }
    }

    function removeValidationMessage(slide) {
        const message = slide.querySelector(
            ".question-validation-message"
        );

        if (message) {
            message.remove();
        }
    }

    nextBtn.addEventListener("click", function () {
        if (!validateCurrentSlide()) {
            return;
        }

        removeValidationMessage(
            slides[currentSlide]
        );

        if (currentSlide < totalSlides - 1) {
            currentSlide++;
            showSlide(currentSlide);
        }
    });

    prevBtn.addEventListener("click", function () {
        removeValidationMessage(
            slides[currentSlide]
        );

        if (currentSlide > 0) {
            currentSlide--;
            showSlide(currentSlide);
        }
    });

    slides.forEach((slide) => {
        const radioInputs = slide.querySelectorAll(
            'input[type="radio"]'
        );

        radioInputs.forEach((radio) => {
            radio.addEventListener("change", function () {
                removeValidationMessage(slide);

                const optionCards =
                    slide.querySelectorAll(".option-card");

                optionCards.forEach((card) => {
                    card.classList.remove("selected");
                });

                const selectedCard =
                    radio.closest(".option-card");

                if (selectedCard) {
                    selectedCard.classList.add("selected");
                }
            });
        });
    });

    form.addEventListener("submit", function (event) {
        if (!validateCurrentSlide()) {
            event.preventDefault();
            return;
        }

        submitBtn.disabled = true;

        submitBtn.innerHTML = `
            Processing...
            <i class="fa-solid fa-spinner fa-spin"></i>
        `;
    });

    showSlide(currentSlide);
});