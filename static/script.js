/* =========================================================
   WHITE PARTICLE BACKGROUND
   Soft floating circles + rice-grain glyphs, drawn white with
   a light shadow so they stay visible against the light page
   background. Respects prefers-reduced-motion.
========================================================= */

(function particleBackground() {
    const canvas = document.getElementById("particle-canvas");
    const ctx = canvas.getContext("2d");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width, height, particles;

    function resize() {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    }

    function makeParticles() {
        const count = Math.max(18, Math.min(38, Math.floor((width * height) / 45000)));
        particles = Array.from({ length: count }, () => ({
            x: Math.random() * width,
            y: Math.random() * height,
            r: 3 + Math.random() * 7,
            speed: 0.15 + Math.random() * 0.35,
            drift: (Math.random() - 0.5) * 0.25,
            opacity: 0.35 + Math.random() * 0.35,
        }));
    }

    function draw() {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#ffffff";

        for (const p of particles) {
            ctx.save();
            ctx.globalAlpha = p.opacity;
            ctx.shadowColor = "rgba(120, 160, 130, 0.35)";
            ctx.shadowBlur = 6;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();

            p.y -= p.speed;
            p.x += p.drift;

            if (p.y < -10) {
                p.y = height + 10;
                p.x = Math.random() * width;
            }
            if (p.x < -10) p.x = width + 10;
            if (p.x > width + 10) p.x = -10;
        }

        if (!reduceMotion) requestAnimationFrame(draw);
    }

    resize();
    makeParticles();
    draw();

    window.addEventListener("resize", () => {
        resize();
        makeParticles();
        if (reduceMotion) draw();
    });
})();


/* =========================================================
   UPLOAD: drag & drop + click + live preview
========================================================= */

const uploadBox = document.getElementById("upload-box");
const fileInput = document.getElementById("rice_image");
const uploadIcon = document.getElementById("upload-icon");
const previewImg = document.getElementById("preview-img");
const uploadTitle = document.getElementById("upload-title");
const uploadSubtitle = document.getElementById("upload-subtitle");

["dragenter", "dragover"].forEach((evt) =>
    uploadBox.addEventListener(evt, (e) => {
        e.preventDefault();
        uploadBox.classList.add("drag-over");
    })
);

["dragleave", "drop"].forEach((evt) =>
    uploadBox.addEventListener(evt, (e) => {
        e.preventDefault();
        uploadBox.classList.remove("drag-over");
    })
);

uploadBox.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files;
        showPreview(file);
    }
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) showPreview(file);
});

function showPreview(file) {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.classList.remove("hidden");
        uploadIcon.classList.add("hidden");
        uploadBox.classList.add("has-image");
        uploadTitle.textContent = file.name;
        uploadSubtitle.textContent = "Click or drop to choose a different image";
    };
    reader.readAsDataURL(file);
}


/* =========================================================
   SUBMIT: AJAX predict + animated result render
========================================================= */

const form = document.getElementById("upload-form");
const detectButton = document.getElementById("detect-button");
const btnLabel = detectButton.querySelector(".btn-label");
const btnSpinner = detectButton.querySelector(".btn-spinner");
const errorBox = document.getElementById("error-box");
const analysisSection = document.getElementById("analysis-section");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorBox.classList.add("hidden");

    if (!fileInput.files[0]) {
        showError("Please select a rice image first.");
        return;
    }

    setLoading(true);

    try {
        const formData = new FormData();
        formData.append("rice_image", fileInput.files[0]);

        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();

        if (!res.ok) {
            showError(data.error || "Something went wrong — please try again.");
            return;
        }

        renderResult(data);
    } catch (err) {
        showError("Couldn't reach the server — please check your connection and try again.");
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    detectButton.disabled = isLoading;
    btnLabel.classList.toggle("hidden", isLoading);
    btnSpinner.classList.toggle("hidden", !isLoading);
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
}

function renderResult(data) {
    // Re-trigger the section's entrance animation on every new result
    analysisSection.classList.remove("hidden");
    analysisSection.style.animation = "none";
    void analysisSection.offsetWidth;
    analysisSection.style.animation = "";

    document.getElementById("result-image").src = data.image_path;

    const isReal = data.result === "REAL RICE";
    const predictionCard = document.getElementById("prediction-card");
    predictionCard.className = "prediction-card " + (isReal ? "real" : "fake");
    document.getElementById("check-icon").textContent = isReal ? "✓" : "!";
    document.getElementById("prediction-text").textContent = data.result;

    animateCount("confidence-number", data.confidence);
    requestAnimationFrame(() => {
        document.getElementById("progress-fill").style.width = data.confidence + "%";
    });

    fillList("reason-list", data.reasons, (text) => `<li>${text}</li>`);
    fillList(
        "main-characteristics-list",
        data.reasons.slice(0, 3),
        (text) => `<li>${text}</li>`
    );

    ["shape", "texture", "color", "surface", "size"].forEach((key, i) => {
        const row = document.querySelector(`.feature-row[data-feature="${key}"]`);
        const value = data.quality[key];
        row.querySelector(".feature-value").textContent = value + "%";
        requestAnimationFrame(() => {
            setTimeout(() => {
                row.querySelector(".feature-fill").style.width = value + "%";
            }, i * 80);
        });
    });

    document.getElementById("quality-stars").textContent = data.quality.stars;
    document.getElementById("quality-word").textContent = data.quality.quality_label;
    animateCount("quality-score-number", data.quality.quality_score);
    requestAnimationFrame(() => {
        document.getElementById("quality-fill").style.width = data.quality.quality_score + "%";
    });

    fillList(
        "characteristics-list",
        data.quality.characteristics,
        (text) => `<li>${text}</li>`
    );

    analysisSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function fillList(elementId, items, template) {
    const el = document.getElementById(elementId);
    el.innerHTML = items.map(template).join("");
    Array.from(el.children).forEach((li, i) => {
        li.style.animationDelay = `${0.1 + i * 0.08}s`;
    });
}

function animateCount(elementId, target) {
    const el = document.getElementById(elementId);
    const duration = 700;
    const start = performance.now();

    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}
