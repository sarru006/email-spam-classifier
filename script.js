const API_BASE_URL = "http://localhost:8008";

// UI Elements
const emailInput = document.getElementById("email-input");
const analyzeBtn = document.getElementById("analyze-btn");
const btnText = analyzeBtn.querySelector(".btn-text");
const loaderDots = analyzeBtn.querySelector(".loader-dots");
const resultContainer = document.getElementById("result-container");
const resultLabel = document.getElementById("result-label");
const resultIcon = document.getElementById("result-icon");
const confidenceCircle = document.getElementById("confidence-circle");
const confidenceValue = document.getElementById("confidence-value");
const procTimeDisplay = document.getElementById("proc-time");
const tokenCloud = document.getElementById("token-cloud");
const charCount = document.getElementById("current-count");
const apiStatus = document.getElementById("api-status");

// Update character count
emailInput.addEventListener("input", () => {
    charCount.textContent = emailInput.value.length;
});

// Check API Health on load
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            apiStatus.textContent = "System Online";
            apiStatus.classList.add("online");
        } else {
            throw new Error();
        }
    } catch (err) {
        apiStatus.textContent = "System Offline";
        apiStatus.classList.add("offline");
    }
}

// Analyze function
analyzeBtn.addEventListener("click", async () => {
    const text = emailInput.value.trim();
    if (!text) return;

    // UI Loading State
    analyzeBtn.disabled = true;
    btnText.classList.add("hidden");
    loaderDots.classList.remove("hidden");
    resultContainer.classList.add("hidden");

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        displayResult(data);
    } catch (err) {
        console.error(err);
        alert("Failed to analyze. Ensure the backend is running on port 8008.");
    } finally {
        analyzeBtn.disabled = false;
        btnText.classList.remove("hidden");
        loaderDots.classList.add("hidden");
    }
});

function displayResult(data) {
    resultContainer.classList.remove("hidden", "spam", "ham");
    
    const isSpam = data.label === "spam";
    resultContainer.classList.add(isSpam ? "spam" : "ham");
    
    // Set Label and Icon
    resultLabel.textContent = data.label;
    resultIcon.textContent = isSpam ? "🚫" : "✅";

    // Update Confidence Meter
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceCircle.style.strokeDasharray = `${confidencePercent}, 100`;

    // Update Details
    procTimeDisplay.textContent = `${data.processing_time_ms}ms`;
    
    // Clear and fill token cloud
    tokenCloud.innerHTML = "";
    if (data.clean_text) {
        data.clean_text.split(" ").forEach(word => {
            const span = document.createElement("span");
            span.className = "token";
            span.textContent = word;
            tokenCloud.appendChild(span);
        });
    } else {
        tokenCloud.textContent = "No significant words detected.";
    }

    // Scroll to results
    resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Initialize
checkHealth();
setInterval(checkHealth, 30000);
