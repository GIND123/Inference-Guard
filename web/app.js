"use strict";

function getRewriteDiff(original, suggested) {
  const tokenize = text => text.match(/\s+|[\p{L}\p{M}\p{N}_]+(?:['’][\p{L}\p{M}\p{N}_]+)*|[^\s]/gu) || [];
  const before = tokenize(original);
  const after = tokenize(suggested);
  let start = 0;
  while (start < before.length && start < after.length && before[start] === after[start]) {
    start++;
  }
  let beforeEnd = before.length;
  let afterEnd = after.length;
  while (beforeEnd > start && afterEnd > start && before[beforeEnd - 1] === after[afterEnd - 1]) {
    beforeEnd--;
    afterEnd--;
  }

  const rows = beforeEnd - start + 1;
  const columns = afterEnd - start + 1;
  // Bound the work on long messages so the results page remains responsive.
  if (rows * columns > 250_000) return null;

  const matches = new Uint16Array(rows * columns);
  for (let i = rows - 2; i >= 0; i--) {
    for (let j = columns - 2; j >= 0; j--) {
      const cell = i * columns + j;
      matches[cell] = before[start + i] === after[start + j]
        ? 1 + matches[cell + columns + 1]
        : Math.max(matches[cell + columns], matches[cell + 1]);
    }
  }

  const diff = [];
  function append(kind, text) {
    if (!text) return;
    const previous = diff[diff.length - 1];
    if (previous?.kind === kind) previous.text += text;
    else diff.push({kind, text});
  }
  append("same", before.slice(0, start).join(""));
  let i = 0;
  let j = 0;
  while (i < rows - 1 || j < columns - 1) {
    if (i < rows - 1 && j < columns - 1 && before[start + i] === after[start + j]) {
      append("same", before[start + i++]);
      j++;
    } else if (i < rows - 1 && (j === columns - 1 || matches[(i + 1) * columns + j] >= matches[i * columns + j + 1])) {
      append("removed", before[start + i++]);
    } else {
      append("added", after[start + j++]);
    }
  }
  append("same", before.slice(beforeEnd).join(""));
  return diff;
}

document.addEventListener("DOMContentLoaded", () => {
  const $ = id => document.getElementById(id);
  const attributes = ["age", "location", "occupation", "education"];
  const number = (v, digits=2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "Not available";
  const label = value => String(value).charAt(0).toUpperCase()+String(value).slice(1);
  const newId = () => "conversation-" + (globalThis.crypto?.randomUUID?.() || Date.now()+"-"+Math.random().toString(36).slice(2));
  const outcomes = {
    improved: ["Estimated privacy risk decreased.", "Review the suggested text to make sure it still says what you need.", "good"],
    improved_with_cost: ["Risk decreased, but meaning may have changed.", "Check for missing or altered details before using this version.", "caution"],
    low_risk: ["Low estimated risk. Nothing needs changing.", "We did not find much this message reveals about you. Review it as usual before sending.", "good"],
    no_change: ["No meaningful reduction in estimated risk.", "The suggestion did not meaningfully reduce estimated inference risk, even if some wording changed.", "caution"],
    worse: ["Estimated risk increased after rewriting.", "Do not assume this suggestion offers better protection. Review or edit it before use.", "bad"],
    explicit_pii_remains: ["Identifying information remains in the suggestion.", "Review detected identifying details before sharing this text.", "bad"]
  };
  const newSecret = () => {
    const bytes = new Uint8Array(32);
    if (globalThis.crypto?.getRandomValues) {
      globalThis.crypto.getRandomValues(bytes);
    } else {
      for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
    }
    return Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
  };
  let latest = null, busy = false, edited = false, multiTurnIdx = 0;
  let sessionSecret = newSecret();
  const aliasVault = new Map();
  const multiTurnDemo = [
    "I take the Green Line every morning to get to lectures.",
    "My second undergraduate co-op starts next semester.",
    "I live close enough to walk through Back Bay on weekends.",
    "My professor moved our capstone deadline on Canvas."
  ];
  $("session-id").value = newId();
  function updateTurnControls(nextTurnNum) {
    $("turn-badge").textContent = "Conversation Turn #" + nextTurnNum;
    const demoLabelTurn = (multiTurnIdx % multiTurnDemo.length) + 1;
    $("multiturn-btn").textContent = "Try multi-turn example (Turn " + demoLabelTurn + "/" + multiTurnDemo.length + ")";
  }
  updateTurnControls(1);
  function isStale() { return !!latest && (latest.input_text !== $("text-input").value.trim() || latest.session_id !== $("session-id").value.trim()); }
  function updateStale() {
    const stale = isStale(); $("stale").hidden = !stale;
    const priorTurns = latest?.response?.conversation_history?.length || 0;
    $("conversation-note").hidden = !(stale && priorTurns > 0 && $("text-input").value.trim());
    $("conversation-note-new").disabled = busy;
    $("copy-btn").disabled = busy || stale;
    $("edit-btn").disabled = busy || stale;
    $("check-edit-btn").disabled = busy || stale;
  }
  function message(text) { $("error").textContent = text; $("error").hidden = !text; }

  // --- live input length -------------------------------------------------
  // The risk model reads at most 512 tokens (~384 English words). Past that it
  // truncates, and the band, the cues and the rewrite would all describe a
  // prefix while the panel presents them as describing the whole message.
  // 300 is the shown limit, left below the real cap so punctuation and rare
  // words cannot push a message that looked fine over the true boundary.
  const WORD_LIMIT = 300;
  const MODEL_WORD_CAP = 384;

  function countWords(text) {
    const trimmed = text.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
  }

  function updateWordCount() {
    const el = $("word-count");
    if (!el) return;
    const words = countWords($("text-input").value);
    $("word-count-value").textContent = words + " / " + WORD_LIMIT + " words";

    let state = "ok", note = "";
    if (words > MODEL_WORD_CAP) {
      state = "truncated";
      note = "Past this length the checker only reads the first " + MODEL_WORD_CAP +
             " words or so, so the result would not cover the rest of your message.";
    } else if (words > WORD_LIMIT) {
      state = "over";
      note = "Over the limit. Shorten it, or check it in parts \u2014 results get less " +
             "reliable as you approach the checker's reading limit.";
    } else if (words > WORD_LIMIT * 0.8) {
      state = "near";
      note = "Approaching the limit.";
    }
    el.dataset.state = state;
    $("word-count-note").textContent = note;
  }

  $("text-input").addEventListener("input", updateWordCount);
  updateWordCount();

  async function checkConnection() {
    $("connection").textContent = "Checking connection…"; $("connection").dataset.state = "";
    try {
      const response = await fetch("/health", {signal: AbortSignal.timeout(8000)});
      const health = await response.json();
      if (!response.ok || health.status !== "ok") throw new Error();
      $("connection").textContent = "Backend connected"; $("connection").dataset.state = "ok";
    } catch (_) { $("connection").textContent = "Backend unavailable"; $("connection").dataset.state = "down"; }
  }
  function setBusy(value) {
    busy = value;
    for (const id of ["analyze-btn", "sample-btn", "multiturn-btn", "new-session", "session-id", "text-input", "suggested-text", "privacy-mode"]) {
      const el = $(id);
      if (el) el.disabled = value;
    }
    $("analyze-btn").textContent = value ? "Checking your message…" : "Check my message";
    $("results").setAttribute("aria-busy", String(value)); updateStale();
  }
  function renderComparison() {
    const original = latest.input_text;
    const suggested = $("suggested-text").value;
    const beforePane = $("original-display");
    const afterPane = $("suggested-display");
    const description = $("change-description");
    const legend = $("change-legend");

    legend.hidden = true;
    beforePane.textContent = original;
    afterPane.textContent = suggested;
    if (original === suggested) {
      description.textContent = "The suggested text is unchanged.";
      return;
    }

    const diff = getRewriteDiff(original, suggested);
    if (!diff) {
      description.textContent = "The text changed. This comparison is too long to highlight individual changes.";
      return;
    }

    beforePane.replaceChildren();
    afterPane.replaceChildren();
    for (const {kind, text} of diff) {
      if (kind === "same") {
        beforePane.append(text);
        afterPane.append(text);
      } else {
        const removed = kind === "removed";
        const change = document.createElement(removed ? "del" : "ins");
        change.className = removed ? "rewrite-removed" : "rewrite-added";
        change.textContent = text;
        (removed ? beforePane : afterPane).append(change);
      }
    }
    legend.hidden = false;
    description.textContent = "Review the highlighted wording changes and the protection outcome below.";
  }

  function setEditing(active) {
    $("suggested-text").hidden = !active;
    $("suggested-display").hidden = active;
    $("edit-btn").textContent = active ? "Finish editing" : "Edit suggested text";
    if (active) {
      $("original-display").textContent = latest.input_text;
      $("change-legend").hidden = true;
      $("change-description").textContent = "Select Finish editing to compare your changes.";
      $("suggested-text").focus();
    } else {
      renderComparison();
    }
  }

  function render(data) {
    const comparison = data.comparison || {}, risk = data.risk_summary || {}, turn = data.turn_record || {}, utility = data.utility_metrics || {};
    let verdict = data.verdict || comparison.verdict;
    if (verdict === "no_change" && risk.overall_band === "LOW") verdict = "low_risk";
    const outcome = outcomes[verdict] || ["Protection outcome is unavailable.", "Review the text; a changed message alone does not establish improved protection.", "caution"];
    $("outcome-title").textContent = outcome[0]; $("outcome-description").textContent = outcome[1];
    $("outcome-title").closest("section").dataset.tone = outcome[2];
    $("before-score").textContent = number(comparison.inferential_before ?? risk.overall);
    $("after-score").textContent = number(comparison.inferential_after);
    $("exposure-summary").textContent = risk.primary ? "The highest estimated exposure is for " + risk.primary + "." : "No primary attribute was returned. Review the scores below.";
    $("attribute-bars").replaceChildren();
    for (const attr of attributes) {
      const score = risk.scores?.[attr], valid = typeof score === "number" && Number.isFinite(score);
      const band = risk.bands?.[attr] || (valid ? score >= .6 ? "HIGH" : score >= .3 ? "MEDIUM" : "LOW" : "Unavailable");
      const row = document.createElement("div"); row.className = "attr-row";
      const name = document.createElement("span"); name.textContent = label(attr);
      const track = document.createElement("div"); track.className = "bar-track"; track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("div"); fill.className = "bar-fill"; fill.style.width = valid ? Math.max(0, Math.min(100, score*100))+"%" : "0%";
      fill.style.backgroundColor = band === "HIGH" ? "var(--red)" : band === "MEDIUM" ? "var(--amber)" : "var(--green)"; track.append(fill);
      const category = document.createElement("span"); category.className = "attr-band"; category.textContent = label(band.toLowerCase());
      const value = document.createElement("span"); value.className = "attr-number"; value.textContent = valid ? number(score) : "—";
      row.append(name, track, category, value); $("attribute-bars").append(row);
    }
    $("suggested-text").value = data.rewritten_text;
    edited = false; $("edit-note").hidden = true; $("check-edit-btn").hidden = true;
    $("action-status").textContent = "";
    setEditing(false);
    const pseudonyms = data.pseudonyms && typeof data.pseudonyms === "object" ? data.pseudonyms : {};
    for (const [orig, tok] of Object.entries(pseudonyms)) {
      aliasVault.set(orig, tok);
    }
    const aliasEntries = Array.from(aliasVault.entries());
    $("pseudonym-badges").replaceChildren();
    if (aliasEntries.length > 0) {
      for (const [orig, tok] of aliasEntries) {
        const badge = document.createElement("span");
        badge.className = "alias-badge";
        badge.textContent = orig + " → " + tok;
        $("pseudonym-badges").append(badge);
      }
      $("alias-bar").hidden = false;
      $("restore-aliases-btn").hidden = false;
      $("pseudonym-display").textContent = typeof data.pseudonymized_text === "string" ? data.pseudonymized_text : "";
    } else {
      $("alias-bar").hidden = true;
      $("restore-aliases-btn").hidden = true;
      $("pseudonym-display").textContent = "No explicit names, emails, or phone numbers required aliasing in this turn.";
    }
    $("presidio-display").textContent = typeof data.presidio_text === "string" ? data.presidio_text : "Baseline output is unavailable.";
    const capacity = turn.residual_anonymity_capacity ?? turn.joint_entropy;
    $("capacity").textContent = typeof capacity === "number" ? number(capacity)+" / 4" : "Not available";
    $("leakage").textContent = typeof turn.leakage_delta === "number" ? (turn.leakage_delta >= 0 ? "+" : "")+number(turn.leakage_delta) : "Not available";

    const history = Array.isArray(data.conversation_history) ? data.conversation_history : [];
    const lastEntry = history.length ? history[history.length - 1] : null;
    const protTurn = data.protected_turn_record || {};
    const cumBefore = lastEntry ? lastEntry.cumulative_risk_before : turn.overall_cumulative_risk;
    const cumAfter = lastEntry ? lastEntry.cumulative_risk_after : (protTurn.overall_cumulative_risk ?? comparison.inferential_after);
    const dBefore = lastEntry ? lastEntry.leakage_delta_before : turn.leakage_delta;
    const dAfter = lastEntry ? lastEntry.leakage_delta_after : protTurn.leakage_delta;
    $("mt-cum-before").textContent = number(cumBefore);
    $("mt-cum-after").textContent = number(cumAfter);
    $("mt-delta-before").textContent = typeof dBefore === "number" ? "Latest turn added " + (dBefore >= 0 ? "+" : "") + number(dBefore) : "";
    $("mt-delta-after").textContent = typeof dAfter === "number" ? "Latest turn added " + (dAfter >= 0 ? "+" : "") + number(dAfter) : "";

    $("mt-attribute-bars").replaceChildren();
    const cumScoresBefore = (lastEntry && lastEntry.cumulative_scores_before) || turn.cumulative_scores || {};
    const cumScoresAfter = (lastEntry && lastEntry.cumulative_scores_after) || protTurn.cumulative_scores || {};
    for (const attr of attributes) {
      const bVal = cumScoresBefore[attr], aVal = cumScoresAfter[attr];
      const validA = typeof aVal === "number" && Number.isFinite(aVal);
      const validB = typeof bVal === "number" && Number.isFinite(bVal);
      const row = document.createElement("div"); row.className = "attr-row";
      const name = document.createElement("span"); name.textContent = label(attr);
      const track = document.createElement("div"); track.className = "bar-track"; track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("div"); fill.className = "bar-fill";
      const displayScore = validA ? aVal : (validB ? bVal : 0);
      fill.style.width = Math.max(0, Math.min(100, displayScore * 100)) + "%";
      fill.style.backgroundColor = displayScore >= 0.6 ? "var(--red)" : displayScore >= 0.3 ? "var(--amber)" : "var(--green)";
      track.append(fill);
      const bandEl = document.createElement("span"); bandEl.className = "attr-band";
      bandEl.textContent = validB && validA ? number(bVal) + " → " + number(aVal) : "—";
      const valEl = document.createElement("span"); valEl.className = "attr-number";
      valEl.textContent = validA ? number(aVal) : "—";
      row.append(name, track, bandEl, valEl);
      $("mt-attribute-bars").append(row);
    }

    $("mt-history-list").replaceChildren();
    for (const item of history) {
      const row = document.createElement("div"); row.className = "mt-turn-row";
      const head = document.createElement("div"); head.className = "mt-turn-head";
      const tTitle = document.createElement("strong"); tTitle.textContent = "Turn " + item.turn_number + " (" + (item.risk_band_before || "—") + " → " + (item.risk_band_after || "—") + ")";
      const preview = document.createElement("span"); preview.className = "mt-turn-preview"; preview.textContent = item.preview || "";
      head.append(tTitle, preview);
      row.append(head);

      for (const [lbl, val, color] of [
        ["Unprotected", item.cumulative_risk_before, "var(--red)"],
        ["Protected", item.cumulative_risk_after, "var(--green)"]
      ]) {
        const pair = document.createElement("div"); pair.className = "mt-bar-pair";
        const spanLbl = document.createElement("span"); spanLbl.textContent = lbl;
        const track = document.createElement("div"); track.className = "bar-track";
        const fill = document.createElement("div"); fill.className = "bar-fill";
        const pct = typeof val === "number" ? Math.max(0, Math.min(100, val * 100)) : 0;
        fill.style.width = pct + "%"; fill.style.backgroundColor = color;
        track.append(fill);
        const num = document.createElement("span"); num.textContent = number(val);
        pair.append(spanLbl, track, num);
        row.append(pair);
      }
      $("mt-history-list").append(row);
    }
    updateTurnControls((history.length || turn.turn_number || 0) + 1);
    for (const [id, key] of [["util-cosine","cosine_similarity"],["util-forward","nli_forward"],["util-backward","nli_backward"],["util-composite","utility_score"]]) $(id).textContent = number(utility[key],4);
    $("utility-method").textContent = utility.utility_backend === "model" ? "These scores come from models that compare the two versions. Check the wording for important changes." : utility.utility_backend === "fallback" ? "Some scores use simple text checks that can miss changes in meaning. Compare both versions before using the suggestion." : "The scoring method is unavailable. Compare both versions to check for changes in meaning.";
    const engines = {qwen_adapter:"Qwen adapter",heuristic:"rule-based rewriting",presidio_fallback:"baseline masking fallback",passthrough_low_risk:"original text retained"};
    $("rewrite-method").textContent = "Served rewrite method: " + (engines[data.rewrite_engine] || "not reported") + ".";
    $("empty").hidden = true; $("results").hidden = false; updateStale();
  }
  async function analyze() {
    if (busy) return;
    const text = $("text-input").value.trim(), session = $("session-id").value.trim();
    if (!text) { message("Enter a message first, then choose Check my message."); $("text-input").focus(); return; }
    if (!session) { message("Enter a conversation ID or start a new conversation in the settings below."); $("session-id").closest("details").open = true; $("session-id").focus(); return; }
    message(""); setBusy(true);
    try {
      const start = performance.now();
      const response = await fetch("/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-Secret": sessionSecret
        },
        body: JSON.stringify({
          text,
          session_id: session,
          user_id: "demo_user",
          privacy_mode: $("privacy-mode")?.value || "personal_balanced"
        }),
        signal: AbortSignal.timeout(180000)
      });
      if (!response.ok) throw new Error("The server returned status " + response.status + ".");
      const data = await response.json();
      if (typeof data.rewritten_text !== "string" || !data.risk_summary || typeof data.risk_summary !== "object") throw new Error("The server returned an incomplete analysis.");
      latest = {session_id:session,user_id:"demo_user",input_text:text,exported_at:new Date().toISOString(),client_latency_seconds:(performance.now()-start)/1000,response:data};
      render(data); $("connection").textContent = "Backend connected"; $("connection").dataset.state = "ok";
    } catch (error) {
      message("We could not complete this check. " + (error.name === "TimeoutError" ? "The request timed out; the server may still be processing this turn." : error.message) + " Your text is still here. Try Check my message again. Any previous results remain from the last successful analysis.");
      checkConnection();
    } finally { setBusy(false); }
  }
  $("analyze-btn").addEventListener("click", analyze);
  $("text-input").addEventListener("keydown", event => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) { event.preventDefault(); analyze(); }
  });
  for (const id of ["text-input","session-id"]) $(id).addEventListener("input", updateStale);
  $("sample-btn").addEventListener("click", () => { if (latest) startNewConversation(); $("text-input").value = "My manager keeps giving client-facing assignments to colleagues with master's degrees. I have been on the team for seven years and train new staff, but I do not have a postgraduate degree. How can I raise this fairly?"; updateStale(); $("text-input").focus(); });
  $("multiturn-btn").addEventListener("click", () => {
    // The example is a fresh four-turn story; don't stack it on unrelated turns.
    if (multiTurnIdx === 0 && latest) startNewConversation();
    const idx = multiTurnIdx % multiTurnDemo.length;
    $("text-input").value = multiTurnDemo[idx];
    multiTurnIdx = (idx + 1) % multiTurnDemo.length;
    updateTurnControls(latest?.response?.conversation_history ? latest.response.conversation_history.length + 1 : 1);
    updateStale();
    $("text-input").focus();
  });
  function startNewConversation() {
    const oldId = $("session-id").value.trim();
    if (oldId) {
      fetch("/session/" + encodeURIComponent(oldId), {
        method: "DELETE",
        headers: {"X-Session-Secret": sessionSecret}
      }).catch(() => {});
    }
    $("session-id").value = newId();
    sessionSecret = newSecret();
    aliasVault.clear();
    multiTurnIdx = 0;
    latest = null;
    $("results").hidden = true;
    $("empty").hidden = false;
    updateTurnControls(1);
    message("");
    updateStale();
  }
  $("new-session").addEventListener("click", startNewConversation);
  $("conversation-note-new").addEventListener("click", () => { startNewConversation(); analyze(); });
  $("edit-btn").addEventListener("click", () => setEditing($("suggested-text").hidden));
  $("restore-aliases-btn").addEventListener("click", () => {
    if (aliasVault.size === 0) return;
    let restored = $("suggested-text").value;
    const entries = Array.from(aliasVault.entries()).sort((a, b) => b[1].length - a[1].length);
    for (const [orig, tok] of entries) {
      restored = restored.split(tok).join(orig);
    }
    $("suggested-text").value = restored;
    edited = restored !== latest.response.rewritten_text;
    $("edit-note").hidden = !edited;
    $("check-edit-btn").hidden = !edited;
    renderComparison();
    $("action-status").textContent = "Restored real names locally in the preview (not sent to the server).";
  });
  $("suggested-text").addEventListener("input", () => {
    edited = $("suggested-text").value !== latest.response.rewritten_text;
    $("edit-note").hidden = !edited;
    $("check-edit-btn").hidden = !edited;
    $("action-status").textContent = "";
  });
  $("check-edit-btn").addEventListener("click", () => { $("text-input").value = $("suggested-text").value; updateStale(); analyze(); });
  $("copy-btn").addEventListener("click", async () => {
    if (!latest || isStale() || busy) return;
    try { await navigator.clipboard.writeText($("suggested-text").value); $("action-status").textContent = edited ? "Copied your edited text. It has not been re-evaluated." : "Copied. Review the text before sharing."; }
    catch (_) {
      if (busy || isStale()) return;
      setEditing(true);
      $("suggested-text").select();
      $("action-status").textContent = "Clipboard access is unavailable. The text is selected; use your browser’s Copy command.";
    }
  });
  $("download-btn").addEventListener("click", () => {
    if (!latest) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(latest,null,2)],{type:"application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = latest.session_id.replace(/[^a-zA-Z0-9_-]/g,"_")+".json"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  });

  // Split-Brain Live Chat Workspace
  const tabInspector = $("tab-inspector"), tabLivechat = $("tab-livechat");
  const inspectorView = $("inspector-view"), livechatView = $("livechat-view");
  if (tabInspector && tabLivechat && inspectorView && livechatView) {
    const switchTab = mode => {
      const isLive = mode === "livechat";
      inspectorView.hidden = isLive;
      livechatView.hidden = !isLive;
      tabInspector.classList.toggle("active", !isLive);
      tabLivechat.classList.toggle("active", isLive);
      tabInspector.setAttribute("aria-selected", String(!isLive));
      tabLivechat.setAttribute("aria-selected", String(isLive));
    };
    tabInspector.addEventListener("click", () => switchTab("inspector"));
    tabLivechat.addEventListener("click", () => switchTab("livechat"));

    let chatSessionId = "chat-" + newId();
    let chatSecret = newSecret();
    const chatAliasVault = new Map(); // token -> original surface
    let chatWireMessages = [];
    let chatDemoIdx = 0;
    const chatScenarios = {
      business_support: {
        privacy_mode: "business_support",
        turns: [
          "Hi, I'm Sarah Chen at 42 Boylston St, Apt 4B (sarah.chen@gmail.com). I'm a 62-year-old retired teacher in Boston on a fixed income—can you waive the $45 late fee on Order #88412?",
          "Also, I recently moved from 100 Beacon St to 42 Boylston St, Apt 4B (555-867-5309). Can you confirm which shipping address and phone number are on Order #88412 and email Ms. Chen the receipt?",
          "Under Massachusetts state consumer law, if the replacement arrives damaged, is Sarah entitled to a full refund on Order #88412 after 14 days?"
        ]
      },
      personal_workplace: {
        privacy_mode: "personal_balanced",
        turns: [
          "My manager Sarah Chen asked me to prep slides for our Q4 board review in Boston.",
          "Sarah wants the slides to highlight our post-doc lab's NIH grant.",
          "Can you draft a polite follow-up email to Ms. Chen asking for feedback before my thesis defense next month?"
        ]
      }
    };
    const getActiveScenario = () => {
      const key = $("chat-scenario")?.value || "business_support";
      return chatScenarios[key] || chatScenarios.business_support;
    };

    function appendRestoredReply(container, rawReply) {
      const div = document.createElement("div");
      div.className = "sb-msg assistant";
      const entries = Array.from(chatAliasVault.entries()).sort((a, b) => b[0].length - a[0].length);
      if (entries.length === 0) {
        div.textContent = rawReply;
        container.append(div);
        return;
      }
      const escaped = entries.map(([tok]) => tok.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      const re = new RegExp("(" + escaped.join("|") + ")", "g");
      const parts = rawReply.split(re);
      for (const part of parts) {
        if (chatAliasVault.has(part)) {
          const mark = document.createElement("mark");
          mark.className = "restored-alias";
          mark.title = "Cloud saw " + part;
          mark.textContent = chatAliasVault.get(part);
          div.append(mark);
        } else {
          div.append(part);
        }
      }
      container.append(div);
    }

    async function sendLiveChat(rawText) {
      const text = rawText.trim();
      if (!text) return;
      const sendBtn = $("chat-send-btn"), demoBtn = $("chat-demo-btn");
      sendBtn.disabled = true; demoBtn.disabled = true;
      const privLog = $("private-chat-log"), wireLog = $("cloud-wire-log");
      if (chatWireMessages.length === 0) { privLog.replaceChildren(); wireLog.replaceChildren(); }

      const userDiv = document.createElement("div");
      userDiv.className = "sb-msg user";
      userDiv.textContent = text;
      privLog.append(userDiv);
      privLog.scrollTop = privLog.scrollHeight;

      try {
        const activeScenario = getActiveScenario();
        const payload = {
          model: ($("chat-model")?.value || "gpt-4o-mini").trim(),
          session_id: chatSessionId,
          privacy_mode: activeScenario.privacy_mode,
          messages: [...chatWireMessages, {role: "user", content: text}],
          upstream_base_url: ($("chat-base-url")?.value || "").trim() || undefined,
          upstream_api_key: ($("chat-api-key")?.value || "").trim() || undefined
        };
        const resp = await fetch("/v1/chat/completions", {
          method: "POST",
          headers: {"Content-Type": "application/json", "X-Session-Secret": chatSecret},
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(180000)
        });
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          if (body?.detail?.error === "upstream_unavailable") {
            throw new Error("no AI model is connected. Add one under “Connect a live LLM” below, or start the server with IG_DEMO_MODE=true for demo replies.");
          }
          throw new Error("the server returned status " + resp.status + ".");
        }
        const data = await resp.json();
        const ig = data.inferenceguard || {};
        const sanitizedUser = ig.sanitized_user_message || text;
        const reply = data.choices?.[0]?.message?.content || "";

        if (ig.pseudonyms && typeof ig.pseudonyms === "object") {
          for (const [orig, tok] of Object.entries(ig.pseudonyms)) {
            if (!chatAliasVault.has(tok)) chatAliasVault.set(tok, orig);
          }
        }
        $("chat-alias-count").textContent = chatAliasVault.size + " local alias" + (chatAliasVault.size === 1 ? "" : "es");

        chatWireMessages.push({role: "user", content: sanitizedUser});
        chatWireMessages.push({role: "assistant", content: reply});

        const wireUser = document.createElement("div");
        wireUser.className = "sb-msg user";
        wireUser.textContent = "[USER SENT TO CLOUD]\n" + sanitizedUser;
        wireLog.append(wireUser);

        const wireAsst = document.createElement("div");
        wireAsst.className = "sb-msg assistant";
        wireAsst.textContent = "[CLOUD REPLY (" + (ig.upstream_mode || "demo") + ")]\n" + reply;
        wireLog.append(wireAsst);
        wireLog.scrollTop = wireLog.scrollHeight;

        appendRestoredReply(privLog, reply);
        privLog.scrollTop = privLog.scrollHeight;

        const cb = typeof ig.cumulative_risk_before === "number" ? ig.cumulative_risk_before.toFixed(2) : "—";
        const ca = typeof ig.cumulative_risk_after === "number" ? ig.cumulative_risk_after.toFixed(2) : "—";
        $("wire-risk-pill").textContent = "Mosaic Risk: " + cb + " → " + ca;
      } catch (err) {
        const errDiv = document.createElement("div");
        errDiv.className = "sb-msg assistant";
        errDiv.textContent = "Could not complete turn: " + err.message;
        privLog.append(errDiv);
      } finally {
        sendBtn.disabled = false; demoBtn.disabled = false;
      }
    }

    const resetLiveChat = () => {
      const turns = getActiveScenario().turns;
      fetch("/session/" + encodeURIComponent(chatSessionId), {
        method: "DELETE",
        headers: {"X-Session-Secret": chatSecret}
      }).catch(() => {});
      chatSessionId = "chat-" + newId();
      chatSecret = newSecret();
      chatAliasVault.clear();
      chatWireMessages = [];
      chatDemoIdx = 0;
      $("chat-demo-btn").textContent = "Run Next Demo Turn (1/" + turns.length + ")";
      $("chat-alias-count").textContent = "0 local aliases";
      $("wire-risk-pill").textContent = "Mosaic Risk: 0.00 → 0.00";
      $("private-chat-log").innerHTML = '<p class="small">Send a message or click “Run Next Demo Turn” to start a private multi-turn conversation.</p>';
      $("cloud-wire-log").innerHTML = '<p class="small">Exact sanitized messages sent over the wire and stored in CRM logs will appear here.</p>';
    };

    $("chat-send-btn").addEventListener("click", () => {
      const inp = $("chat-input");
      const val = inp.value;
      inp.value = "";
      sendLiveChat(val);
    });
    $("chat-input").addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!$("chat-send-btn").disabled) $("chat-send-btn").click();
      }
    });
    $("chat-demo-btn").addEventListener("click", () => {
      const turns = getActiveScenario().turns;
      const msg = turns[chatDemoIdx % turns.length];
      chatDemoIdx = (chatDemoIdx + 1) % turns.length;
      $("chat-demo-btn").textContent = "Run Next Demo Turn (" + (chatDemoIdx + 1) + "/" + turns.length + ")";
      sendLiveChat(msg);
    });
    $("chat-new-btn").addEventListener("click", resetLiveChat);
    $("chat-scenario")?.addEventListener("change", resetLiveChat);
  }
  checkConnection();
});
