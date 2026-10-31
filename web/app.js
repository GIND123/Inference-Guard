"use strict";
document.addEventListener("DOMContentLoaded", () => {
  const $ = id => document.getElementById(id);
  const attributes = ["age", "location", "occupation", "education"];
  const number = (v, digits=2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "Not available";
  const label = value => String(value).charAt(0).toUpperCase()+String(value).slice(1);
  const newId = () => "conversation-" + (globalThis.crypto?.randomUUID?.() || Date.now()+"-"+Math.random().toString(36).slice(2));
  const outcomes = {
    improved: ["Estimated privacy risk decreased.", "Review the suggested text to make sure it still says what you need.", "good"],
    improved_with_cost: ["Risk decreased, but meaning may have changed.", "Check for missing or altered details before using this version.", "caution"],
    no_change: ["No meaningful reduction in estimated risk.", "The suggestion did not meaningfully reduce estimated inference risk, even if some wording changed.", "caution"],
    worse: ["Estimated risk increased after rewriting.", "Do not assume this suggestion offers better protection. Review or edit it before use.", "bad"],
    explicit_pii_remains: ["Identifying information remains in the suggestion.", "Review detected identifying details before sharing this text.", "bad"]
  };
  let latest = null, busy = false, edited = false;
  $("session-id").value = newId();
  function isStale() { return !!latest && (latest.input_text !== $("text-input").value.trim() || latest.session_id !== $("session-id").value.trim()); }
  function updateStale() {
    const stale = isStale(); $("stale").hidden = !stale;
    $("copy-btn").disabled = busy || stale;
    $("edit-btn").disabled = busy || stale;
    $("check-edit-btn").disabled = busy || stale;
  }
  function message(text) { $("error").textContent = text; $("error").hidden = !text; }
  async function checkConnection() {
    $("connection").textContent = "Checking connection…";
    try {
      const response = await fetch("/health", {signal: AbortSignal.timeout(8000)});
      const health = await response.json();
      if (!response.ok || health.status !== "ok") throw new Error();
      $("connection").textContent = "Backend connected";
    } catch (_) { $("connection").textContent = "Backend unavailable"; }
  }
  function setBusy(value) {
    busy = value;
    for (const id of ["analyze-btn", "sample-btn", "new-session", "session-id", "text-input", "suggested-text"]) $(id).disabled = value;
    $("analyze-btn").textContent = value ? "Checking your message…" : "Check my message";
    $("results").setAttribute("aria-busy", String(value)); updateStale();
  }
  function render(data) {
    const comparison = data.comparison || {}, risk = data.risk_summary || {}, turn = data.turn_record || {}, utility = data.utility_metrics || {};
    const outcome = outcomes[data.verdict || comparison.verdict] || ["Protection outcome is unavailable.", "Review the text; a changed message alone does not establish improved protection.", "caution"];
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
    $("original-display").textContent = latest.input_text;
    $("suggested-text").value = data.rewritten_text;
    $("suggested-text").readOnly = true; edited = false; $("edit-note").hidden = true; $("check-edit-btn").hidden = true;
    $("action-status").textContent = "";
    const changed = latest.input_text.replace(/\s+/g, " ").trim() !== data.rewritten_text.replace(/\s+/g, " ").trim();
    $("change-description").textContent = changed ? "The wording changed. Review both versions for missing or altered details; a wording change alone is not proof of protection." : "The suggested text is unchanged. No different wording was provided.";
    $("pseudonym-display").textContent = data.pseudonymized_text || "Separate pseudonymization output is unavailable. The baseline is not a substitute for a persistent-alias result.";
    $("presidio-display").textContent = typeof data.presidio_text === "string" ? data.presidio_text : "Baseline output is unavailable.";
    const capacity = turn.residual_anonymity_capacity ?? turn.joint_entropy;
    $("capacity").textContent = typeof capacity === "number" ? number(capacity)+" / 4" : "Not available";
    $("leakage").textContent = typeof turn.leakage_delta === "number" ? (turn.leakage_delta >= 0 ? "+" : "")+number(turn.leakage_delta) : "Not available";
    for (const [id, key] of [["util-cosine","cosine_similarity"],["util-forward","nli_forward"],["util-backward","nli_backward"],["util-composite","utility_score"]]) $(id).textContent = number(utility[key],4);
    $("utility-method").textContent = utility.utility_backend === "model" ? "Method: model-based similarity and entailment evaluation." : utility.utility_backend === "fallback" ? "Method: fallback evaluation. One or more components use lexical heuristics rather than the full model evaluation." : "Evaluation method was not reported; model-based scoring is not assumed.";
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
      const response = await fetch("/analyze", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text, session_id:session, user_id:"demo_user"}),signal:AbortSignal.timeout(180000)});
      if (!response.ok) throw new Error("The server returned status " + response.status + ".");
      const data = await response.json();
      if (typeof data.rewritten_text !== "string" || !data.risk_summary || typeof data.risk_summary !== "object") throw new Error("The server returned an incomplete analysis.");
      latest = {session_id:session,user_id:"demo_user",input_text:text,exported_at:new Date().toISOString(),client_latency_seconds:(performance.now()-start)/1000,response:data};
      render(data); $("connection").textContent = "Backend connected";
    } catch (error) {
      message("We could not complete this check. " + (error.name === "TimeoutError" ? "The request timed out; the server may still be processing this turn." : error.message) + " Your text is still here. Try Check my message again. Any previous results remain from the last successful analysis.");
      checkConnection();
    } finally { setBusy(false); }
  }
  $("analyze-btn").addEventListener("click", analyze);
  for (const id of ["text-input","session-id"]) $(id).addEventListener("input", updateStale);
  $("sample-btn").addEventListener("click", () => { $("text-input").value = "My manager keeps giving client-facing assignments to colleagues with master's degrees. I have been on the team for seven years and train new staff, but I do not have a postgraduate degree. How can I raise this fairly?"; updateStale(); $("text-input").focus(); });
  $("new-session").addEventListener("click", () => { $("session-id").value = newId(); message(""); updateStale(); });
  $("edit-btn").addEventListener("click", () => { $("suggested-text").readOnly = false; $("suggested-text").focus(); });
  $("suggested-text").addEventListener("input", () => { edited = true; $("edit-note").hidden = false; $("check-edit-btn").hidden = false; $("action-status").textContent = ""; });
  $("check-edit-btn").addEventListener("click", () => { $("text-input").value = $("suggested-text").value; updateStale(); analyze(); });
  $("copy-btn").addEventListener("click", async () => {
    if (!latest || isStale() || busy) return;
    try { await navigator.clipboard.writeText($("suggested-text").value); $("action-status").textContent = edited ? "Copied your edited text. It has not been re-evaluated." : "Copied. Review the text before sharing."; }
    catch (_) { $("suggested-text").focus(); $("suggested-text").select(); $("action-status").textContent = "Clipboard access is unavailable. The text is selected; use your browser’s Copy command."; }
  });
  $("download-btn").addEventListener("click", () => {
    if (!latest) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(latest,null,2)],{type:"application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = latest.session_id.replace(/[^a-zA-Z0-9_-]/g,"_")+".json"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000);
  });
  checkConnection();
});
