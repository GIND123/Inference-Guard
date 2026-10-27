const igDefinitions = {"Overall Risk": "The highest estimated risk among age, location, occupation, and education for this message. Higher means greater estimated exposure, not a verified probability of identifying someone.", "Primary Risk": "The personal attribute with the highest estimated risk in this message. This indicates potential exposure, not a confirmed fact about the author.", "Residual Anonymity Capacity": "How much anonymity capacity remains across the four tracked attributes in this session. Calculated as the sum of (1 minus cumulative risk) for each attribute. Range: 0–4; higher means less cumulative exposure. This is a heuristic index, not Shannon entropy, despite the current 'bits' label.", "Leakage Delta": "The increase in the session's highest cumulative attribute risk since the previous message. For example, 0.10 means an increase of 0.10 on the 0–1 scale. The first message is compared with zero. Zero does not mean the message contains no private information.", "Cosine Similarity (MiniLM):": "How similar the original and rewritten text are. Higher means greater similarity. With MiniLM loaded, this uses sentence embeddings. The current rule-based demo uses word overlap instead. Similarity alone does not guarantee important details are preserved.", "Forward NLI (DeBERTa):": "Checks whether the original text supports the rewritten text. Higher suggests fewer unsupported claims were added. The current rule-based demo uses a lexical heuristic, not DeBERTa inference.", "Backward NLI:": "Checks whether the rewritten text supports the original text. Higher suggests fewer original details were lost. The current rule-based demo uses a lexical heuristic rather than a language-model entailment assessment.", "Composite Utility:": "A combined preservation score using similarity and both NLI directions, with a penalty for contradiction. Higher means better estimated preservation. Degenerate rewrites receive zero. In this demo, the inputs are lexical heuristics; the score does not guarantee rewrite quality.", "Stage 1: Persistent Pseudonyms": "Replaces identifying information with consistent aliases, such as PERSON_1 or LOCATION_1. Repeated mentions of the same entity keep the same alias within a session, preserving relationships while hiding the original identifiers. In this demo, this panel falls back to the baseline redaction output when no pseudonymized output is provided.", "Presidio Explicit Baseline": "A reference method for detecting and masking explicitly stated personal information, such as names, email addresses, phone numbers, and locations. It is used to compare basic redaction with InferenceGuard's protection against indirect privacy clues. The current demo uses built-in regex redaction because Presidio is not installed."};

(() => {
    const tip = document.createElement("div");
    tip.id = "ig-evaluation-tooltip";
    tip.setAttribute("role", "tooltip");
    tip.hidden = true;
    document.body.append(tip);

    let active = null;
    let timer;

    function hide() {
        clearTimeout(timer);
        if (active) active.removeAttribute("aria-describedby");
        active = null;
        tip.hidden = true;
    }

    function position() {
        if (!active) return;

        const r = active.getBoundingClientRect();
        const t = tip.getBoundingClientRect();

        const left = Math.max(
            12,
            Math.min(r.left, innerWidth - t.width - 12)
        );

        const below = r.bottom + 8;
        const top = below + t.height <= innerHeight - 12
            ? below
            : r.top - t.height - 8;

        tip.style.left = left + "px";
        tip.style.top = Math.max(12, top) + "px";
    }

    function show(button, text) {
        hide();
        active = button;
        tip.textContent = text;
        tip.hidden = false;
        button.setAttribute("aria-describedby", tip.id);
        position();
    }

    function scheduleHide() {
        clearTimeout(timer);
        timer = setTimeout(() => {
            if (
                active &&
                !tip.matches(":hover") &&
                !active.matches(":hover, :focus")
            ) {
                hide();
            }
        }, 180);
    }

    document.querySelectorAll(".metric-label, .util-label, .column-title")
        .forEach(label => {
            const name = label.textContent.trim();
            const text = igDefinitions[name];
            if (!text) return;

            const button = document.createElement("button");
            button.type = "button";
            button.className = "ig-help-button";
            button.textContent = "?";
            button.setAttribute(
                "aria-label",
                "Explain " + name.replace(/:$/, "")
            );

            button.addEventListener(
                "mouseenter", () => show(button, text)
            );
            button.addEventListener(
                "focus", () => show(button, text)
            );
            button.addEventListener(
                "click", () => show(button, text)
            );
            button.addEventListener("mouseleave", scheduleHide);
            button.addEventListener("blur", scheduleHide);

            label.append(button);
        });

    tip.addEventListener(
        "mouseenter", () => clearTimeout(timer)
    );
    tip.addEventListener("mouseleave", scheduleHide);

    document.addEventListener("keydown", event => {
        if (event.key === "Escape") hide();
    });

    document.addEventListener("pointerdown", event => {
        if (
            active &&
            !active.contains(event.target) &&
            !tip.contains(event.target)
        ) {
            hide();
        }
    });

    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
})();
