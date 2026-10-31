// InferenceGuard Frontend Controller

document.addEventListener("DOMContentLoaded", () => {
    const textInput = document.getElementById("text-input");
    const sessionIdInput = document.getElementById("session-id");
    const analyzeBtn = document.getElementById("analyze-btn");
    const sampleBtn = document.getElementById("sample-btn");
    const downloadBtn = document.getElementById("download-btn");

    let latestResult = null;

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
    const pseudonymDisplay = document.getElementById("pseudonym-display");
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

            const requestStart = performance.now();

            const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const requestEnd = performance.now();
            const clientLatencySeconds = (requestEnd - requestStart) / 1000;

            if (!response.ok) {
                let errMsg = "Analysis request failed (" + response.status + ").";
                try {
                    const err = await response.json();
                    if (err && err.detail) errMsg = err.detail;
                } catch (_) {}
                throw new Error(errMsg);
            }

            const data = await response.json();

            latestResult = {
                session_id: payload.session_id,
                user_id: payload.user_id,
                input_text: text,
                exported_at: new Date().toISOString(),
                client_latency_seconds: clientLatencySeconds,
                response: data
            };

            downloadBtn.disabled = false;

            renderResults(text, data);
        } catch (error) {
            console.error("Analysis error:", error);
            alert("Error during analysis: " + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = "Analyze and Protect";
        }
    });

    downloadBtn.addEventListener("click", () => {
        if (!latestResult) {
            alert("No analysis result available to download.");
            return;
        }

        const safeSessionId = latestResult.session_id.replace(
            /[^a-zA-Z0-9_-]/g,
            "_"
        );

        const blob = new Blob(
            [JSON.stringify(latestResult, null, 2)],
            { type: "application/json" }
        );

        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = `${safeSessionId}.json`;

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        URL.revokeObjectURL(url);
    });

    function renderResults(rawText, data) {
        const safeData = data || {};
        const risk = safeData.risk_summary || {};
        const turn = safeData.turn_record || {};
        const utility = safeData.utility_metrics || {};

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

        // Metrics: Residual Anonymity Capacity and Leakage Delta
        overallScore.textContent = Number(risk.overall || 0).toFixed(2);
        primaryAttr.textContent = risk.primary ? String(risk.primary).toUpperCase() : "None";
        const racVal = turn.residual_anonymity_capacity !== undefined ? turn.residual_anonymity_capacity : turn.joint_entropy;
        jointEntropy.textContent = racVal !== undefined && racVal !== null ? Number(racVal).toFixed(2) + " bits" : "4.00 bits";
        leakageDelta.textContent = turn.leakage_delta !== undefined && turn.leakage_delta !== null ? Number(turn.leakage_delta).toFixed(2) : "0.00";

        // Attribute bars
        const scores = risk.scores || {};
        setBar(barAge, valAge, scores.age || 0);
        setBar(barLoc, valLoc, scores.location || 0);
        setBar(barOcc, valOcc, scores.occupation || 0);
        setBar(barEdu, valEdu, scores.education || 0);

        // Highlight Cues in original text using character span offsets when available
        const cues = (risk.cues || []).filter(c => c && c.span && c.span.length >= 2 && !c.span.startsWith("["));
        const validOffsetCues = cues.filter(c => typeof c.start === "number" && typeof c.end === "number" && c.start >= 0 && c.end <= rawText.length && c.start < c.end);

        let highlighted = "";
        if (validOffsetCues.length > 0) {
            // Sort non-overlapping cues by start offset ascending
            validOffsetCues.sort((a, b) => a.start - b.start);
            let lastIdx = 0;
            validOffsetCues.forEach(cue => {
                if (cue.start >= lastIdx) {
                    highlighted += escapeHtml(rawText.slice(lastIdx, cue.start));
                    const tip = escapeHtml((cue.attribute || 'risk') + ' (' + Number(cue.importance || 0).toFixed(2) + ')');
                    highlighted += `<span class="highlight-cue" title="${tip}">${escapeHtml(rawText.slice(cue.start, cue.end))}</span>`;
                    lastIdx = cue.end;
                }
            });
            highlighted += escapeHtml(rawText.slice(lastIdx));
        } else {
            highlighted = escapeHtml(rawText);
            cues.forEach(cue => {
                if (cue.span) {
                    const regex = new RegExp("(" + escapeRegExp(cue.span) + ")", "gi");
                    highlighted = highlighted.replace(regex, '<span class="highlight-cue">$1</span>');
                }
            });
        }
        originalDisplay.innerHTML = highlighted || "No input submitted.";

        // Stage 1: Persistent Pseudonymization text
        if (pseudonymDisplay) {
            pseudonymDisplay.textContent = safeData.pseudonymized_text || safeData.presidio_text || "No pseudonymization generated.";
        }

        // Rewritten text and Presidio text
        // Passthrough is decided by the engine, not the band: a LOW turn that
        // adds to what earlier turns revealed is rewritten (decision_basis "cumulative").
        const engineName = safeData.rewrite_engine || data.rewrite_engine;
        if (engineName ? engineName === "passthrough_low_risk" : risk.overall_band === "LOW") {
            rewriteDisplay.innerHTML = "<em>No inferential markers detected. Original text preserved.</em><br/><br/>" + escapeHtml(safeData.original_text || rawText);
        } else if (data.rewrite_changed === false || safeData.rewrite_changed === false) {
            // Session 06: participants read an unchanged message as protected.
            rewriteDisplay.innerHTML = "<strong>No safe rewrite was found — your message is unchanged and is NOT protected.</strong>";
        } else {
            rewriteDisplay.textContent = safeData.rewritten_text || "No rewrite generated.";
        }
        if (data.rewrite_engine || safeData.rewrite_engine) {
            const engineNote = document.createElement("small");
            engineNote.textContent = "Engine: " + (safeData.rewrite_engine || data.rewrite_engine);
            rewriteDisplay.appendChild(document.createElement("br"));
            rewriteDisplay.appendChild(engineNote);
        }
        // Multi-turn: say what the conversation as a whole has already revealed.
        const conv = safeData.conversation || data.conversation;
        if (conv && conv.exposed_attributes && conv.exposed_attributes.length) {
            const convNote = document.createElement("p");
            let msg = "Across this conversation, your earlier messages already point to your " +
                conv.exposed_attributes.join(", ") + ".";
            if (conv.decision_basis === "cumulative") {
                msg += " This message adds to that, so it was checked even though it looks harmless on its own.";
            }
            convNote.textContent = msg;
            rewriteDisplay.appendChild(convNote);
        }
        presidioDisplay.textContent = safeData.presidio_text || data.presidio_text || "No baseline generated.";

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
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function escapeRegExp(str) {
        if (!str) return "";
        return String(str).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
});
