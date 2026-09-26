// InferenceGuard Frontend Controller

document.addEventListener("DOMContentLoaded", () => {
    const textInput = document.getElementById("text-input");
    const sessionIdInput = document.getElementById("session-id");
    const analyzeBtn = document.getElementById("analyze-btn");
    const sampleBtn = document.getElementById("sample-btn");

    const overallBadge = document.getElementById("overall-badge");
    const overallScore = document.getElementById("overall-score");
    const primaryAttr = document.getElementById("primary-attr");
    const jointEntropy = document.getElementById("joint-entropy");
    const leakageDelta = document.getElementById("leakage-delta");

    const barAge = document.getElementById("bar-age");
    const barLoc = document.getElementById("bar-location");
    const barOcc = document.getElementById("bar-occupation");
    const barEdu = document.getElementById("bar-education");

    const valAge = document.getElementById("val-age");
    const valLoc = document.getElementById("val-location");
    const valOcc = document.getElementById("val-occupation");
    const valEdu = document.getElementById("val-education");

    const originalDisplay = document.getElementById("original-display");
    const rewriteDisplay = document.getElementById("rewrite-display");
    const presidioDisplay = document.getElementById("presidio-display");

    const utilCosine = document.getElementById("util-cosine");
    const utilNliFwd = document.getElementById("util-nli-fwd");
    const utilNliBwd = document.getElementById("util-nli-bwd");
    const utilComposite = document.getElementById("util-composite");

    const benchmarkSample = "I am 26 years old living in Denver, CO and working as a software engineer after graduating from Carnegie Mellon.";

    sampleBtn.addEventListener("click", () => {
        textInput.value = benchmarkSample;
    });

    analyzeBtn.addEventListener("click", async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert("Please enter a text message to analyze.");
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Analyzing...";

        try {
            const payload = {
                text: text,
                session_id: sessionIdInput.value.trim() || "default",
                user_id: "demo_user"
            };

            const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Analysis request failed.");
            }

            const data = await response.json();
            renderResults(text, data);
        } catch (error) {
            console.error("Analysis error:", error);
            alert("Error during analysis: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = "Analyze and Protect";
        }
    });

    function renderResults(rawText, data) {
        const risk = data.risk_summary;
        const turn = data.turn_record || {};
        const utility = data.utility_metrics || {};

        // Overall risk band and badge
        const band = risk.overall_band || "LOW";
        overallBadge.textContent = "BAND: " + band;
        overallBadge.className = "badge";
        if (band === "HIGH") {
            overallBadge.classList.add("badge-high");
        } else if (band === "MEDIUM") {
            overallBadge.classList.add("badge-med");
        } else {
            overallBadge.classList.add("badge-low");
        }

        // Metrics
        overallScore.textContent = Number(risk.overall || 0).toFixed(2);
        primaryAttr.textContent = risk.primary ? risk.primary.toUpperCase() : "None";
        jointEntropy.textContent = turn.joint_entropy !== undefined ? Number(turn.joint_entropy).toFixed(2) + " bits" : "4.00 bits";
        leakageDelta.textContent = turn.leakage_delta !== undefined ? Number(turn.leakage_delta).toFixed(2) : "0.00";

        // Attribute bars
        const scores = risk.scores || {};
        setBar(barAge, valAge, scores.age || 0);
        setBar(barLoc, valLoc, scores.location || 0);
        setBar(barOcc, valOcc, scores.occupation || 0);
        setBar(barEdu, valEdu, scores.education || 0);

        // Highlight Cues in original text
        let highlighted = escapeHtml(rawText);
        const cues = risk.cues || [];
        cues.forEach(cue => {
            if (cue.span && cue.span.length > 2 && !cue.span.startsWith("[")) {
                const regex = new RegExp("(" + escapeRegExp(cue.span) + ")", "gi");
                highlighted = highlighted.replace(regex, '<span class="highlight-cue">$1</span>');
            }
        });
        originalDisplay.innerHTML = highlighted;

        // Rewritten text and Presidio text
        rewriteDisplay.textContent = data.rewritten_text || "No rewrite generated.";
        presidioDisplay.textContent = data.presidio_text || "No baseline generated.";

        // Utility metrics
        utilCosine.textContent = Number(utility.cosine_similarity || 0).toFixed(4);
        utilNliFwd.textContent = Number(utility.nli_forward || 0).toFixed(4);
        utilNliBwd.textContent = Number(utility.nli_backward || 0).toFixed(4);
        utilComposite.textContent = Number(utility.utility_score || 0).toFixed(4);
    }

    function setBar(barEl, valEl, score) {
        const pct = Math.min(100, Math.max(0, Math.round(score * 100)));
        barEl.style.width = pct + "%";
        if (score >= 0.60) {
            barEl.style.backgroundColor = "var(--high-color)";
        } else if (score >= 0.30) {
            barEl.style.backgroundColor = "var(--med-color)";
        } else {
            barEl.style.backgroundColor = "var(--low-color)";
        }
        valEl.textContent = Number(score).toFixed(2);
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
    }

    function escapeRegExp(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
});
