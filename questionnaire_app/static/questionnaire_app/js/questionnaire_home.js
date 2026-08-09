document.addEventListener("DOMContentLoaded", function () {

    const animatedElements = document.querySelectorAll(
        ".info-box, .step-card, .notice-card"
    );

    animatedElements.forEach((element) => {
        element.classList.add("hidden");
    });

    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("show");
                    observer.unobserve(entry.target);
                }
            });
        },
        {
            threshold: 0.15
        }
    );

    animatedElements.forEach((element) => {
        observer.observe(element);
    });


    const startButton = document.querySelector(".start-btn");

    if (startButton) {
        startButton.addEventListener("click", function (event) {

            const ripple = document.createElement("span");
            ripple.classList.add("ripple");

            const rect = startButton.getBoundingClientRect();

            const size = Math.max(
                rect.width,
                rect.height
            );

            ripple.style.width = `${size}px`;
            ripple.style.height = `${size}px`;

            ripple.style.left =
                `${event.clientX - rect.left - size / 2}px`;

            ripple.style.top =
                `${event.clientY - rect.top - size / 2}px`;

            startButton.appendChild(ripple);

            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    }

});