---
team: InferenceGuard
session: 04
date: 2026-09-21
members:
  - name: Anna Thomas
    github: AnnaThomas2060
    hat: Data&Eval
  - name: Chatur Bandaru
    github: chaturbandaru
    hat: Engineering
  - name: Govind
    github: GIND123
    hat: Product
  - name: Yiwei Jin
    github: runyoucleverdr-beep
    hat: Users&Research
  - name: Bingqi Lian
    github: lianbingqi
    hat: Operations
north_star:
  metric: >-
    Target: attribute leakage rate under a held-out adversary. Not yet
    computable (Phi-4-mini-instruct not implemented). Session 04 proxy:
    overall risk delta, taken as the max across attributes, with util_sim
    as the utility guardrail.
  value: >-
    Proxy only, 3 dev-test messages in data/logs/session04_usage_log.csv:
    overall confidence delta 0.428 / 0.010 / 0.056; util_sim 0.916 / 0.907 /
    0.827. Team-generated, so not user evidence.
  previous: n/a — first measurement
---

## Shipped this week
- **Dataset foundation + taxonomy frozen**: Loaded `RobinSta/SynthPAI` (300 profiles, 103 threads, 7,823 comments), robust parsing for dict / JSON string / double-stringified `profile` field, audited distributions, frozen coarse taxonomy to `artifacts/taxonomy.json` — age [18-24,25-34,35-44,45-54,55+,unknown], education [high_school, undergraduate, graduate_professional, other_unknown], occupation [student, technology, healthcare, education, business_finance, service, other], location [USA_Northeast, USA_South, USA_Midwest, USA_West, Europe, Other, unknown] (evidence: `artifacts/taxonomy.json`, committed in PR #15; notebook section *Attribute Taxonomy Finalization*)

- Profile-disjoint splits: Deterministic split by author/username to prevent author-style memorization — 70% train / 15% val / 15% test, seed 42, saved to `artifacts/profile_splits.json` (committed in PR #15) with ~300 unique profiles verified. 

- Explicit PII baseline: Integrated Presidio Analyzer + Anonymizer. Demonstrated gap: On the phrase "My co-op ends in December, and I take the Green Line to campus", Presidio correctly caught explicit PII (detecting "December" as DATE_TIME and "Green Line" as LOCATION), but missed the inferred data we are trying to solve (e.g., "co-op" implying student status, or the combination implying Boston/undergraduate).

- Risk estimator v0: `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` zero-shot + `dbmdz/bert-large-cased-finetuned-conll03-english` NER. 5 dimensions: location, education, occupation, age, name. Implements `detect_entities()`, `modernbert_predict()` returning `confidence=max(risk_scores)`, `evidence`, `has_hard_leak` (PER/LOC/ORG)

- Privacy-preserving rewriter v0: `Qwen/Qwen2.5-0.5B-Instruct` with privacy system prompt + regex `_hard_scrub()` safety net + `rewrite_until_safe(threshold=0.4, max_attempts=3)` that always rewrites from ORIGINAL and picks best by `(has_hard_leak, confidence)`

- End-to-end vertical slice: `text -> Presidio -> risk estimation -> rewrite -> re-scoring` with structured JSON output `{original, protected, before_loc, after_loc, delta, util_sim, evidence_before, evidence_after}` and live interactive loop logging to `data/logs/session04_usage_log.csv` using `all-MiniLM-L6-v2` cosine for utility.

- Repo governance + product foundation (Product): branch protection enabled on `main` (1 approving review, admins included, force-push and deletion blocked) — the Session 03 requirement verified in Session 04. Peer review delivered on PR #14. Lean canvas, user-facing risk banding and a synthetic scenario bank are open for review in PR #18 (#17), **not yet merged**, so they are not counted as shipped this week.

## User evidence
- **We have no qualifying user evidence this week, and we are not claiming any.** The only capture is an interactive CLI loop (`input("You: ")`) run by the team against itself. Under the course evidence standard, team members using their own product does not count as validation. We are stating this rather than letting the log imply otherwise.
- Raw artifact: `data/logs/session04_usage_log.csv` (committed; timestamp, original, protected, per-attribute before/after, util_sim, evidence). Three messages, all team-generated. Treat it as a dev log, not user evidence.
- Blocking cause, now removed: we had no synthetic scenarios to recruit against, so no session could be run without asking a participant for real personal data. Six scenarios, a facilitator script and a fixed capture format are in PR #18 (`docs/product/user_test_scenarios.md`). First external session is the Session 05 priority.
- What we discovered - 
  - The models used for the rewriter and risk estimator are currently zero-shot prompted and not yet finetuned. Based on the responses in the user test logs, we observe that finetuning is heavily required. For instance, the v0 model over-scrubs terms like "Type 2 diabetes" to "an entity diabetes", and occasionally leaks PII in complex resumes.
  - Explicit PII only checks PER/LOC/ORG, misses EMAIL, PHONE, ACCOUNT, MRN, MONEY as seen in live tests.

- What we plan to improve - 
  - Model: Fine-tune ModernBERT multi-task with temp scaling + ECE, use Integrated Gradients for cues.
  - Pipeline: Two-stage : Presidio for explicit PII + inferential rewriter, QLoRA Qwen 0.8B with 2B fallback.

## Metrics snapshot
- Overall risk delta (max across attributes), 3 dev-test messages: 0.489 -> 0.061 (delta 0.428), 0.997 -> 0.987 (delta 0.010), 0.060 -> 0.004 (delta 0.056). util_sim 0.916 / 0.907 / 0.827.
- Single-attribute deltas are reported alongside, never alone. On the resume message location fell 0.859 -> 0.006 (delta 0.852) while overall risk moved 0.010 and the protected text kept the name, email and phone. Quoting the 0.852 in isolation would describe a failed rewrite as our best result; `src/product/risk_bands.py` in PR #18 enforces max-not-mean in code so the UI cannot repeat it.
- Measured on: 3 team-authored dev messages. No held-out split was involved, so these are smoke-test numbers, not an evaluation.
- Is this the same model that is running in the product? No, Current risk model is `ModernBERT-base-zeroshot-v2.0` proxy; final will be fine-tuned ModernBERT-base multi-task classifier with temperature scaling/ECE. Rewriter is base Qwen2.5-0.5B-Instruct, not yet QLoRA fine-tuned. Held-out adversary `Phi-4-mini-instruct` not yet implemented.

## What did not work
- Traditional PII tooling is insufficient for inferred data: Presidio successfully catches explicit entities like dates and locations, but leaves inferential cues like "co-op" and "campus" untouched, which are what reveal student status, education, and age range.
- Qwen2.5-0.5B leaks original entities despite system prompt; required regex scrub net and `scrub_entities` union from original detection. Model too small for reliable generalization, may need fallback to 2B.
- Coarse taxonomy loses nuance — e.g. "Bachelor of Science in Computer Science, Northeastern University" -> undergraduate bucket discards institution signal that matters for privacy.
- No calibration yet, zero-shot scores not ECE measured, thresholds 0.4 / LOW 0-0.29 MEDIUM 0.3-0.59 HIGH 0.6-1.0 are uncalibrated.
- Profile parsing brittle, HF loader returns mixed types (dict, JSON string, Python repr); initial `attr_df` extraction failed and only returned `_author`.


## Challenges / blockers
- Need GPU for ModernBERT + Qwen pipeline + SentenceTransformer together; local inference memory pressure
- Need SFT data generation from train-only profiles (cannot use test profiles) scored by privacy reduction + utility — methodology for filtering synthetic pairs not defined
- Held-out evaluation with Phi-4-mini-instruct and utility metrics (DeBERTa-v3 NLI cross-encoder) not yet implemented
- Multi-turn leakage history UI and stable pseudonym mapping not started
- User studies must use synthetic scenarios only, without logging real sensitive messages. The scenario bank in PR #18 resolves the "what do we hand a participant" half; the recruiting half is still open.
- **Hats were assigned late.** All five hats now have an owner (roster above), but Users&Research and Operations were only settled on 21 September — the last day of this session. That is the direct cause of the missing user evidence: nobody owned recruiting while there was still time to recruit. Not a headcount problem, a timing one, and it does not repeat next week.

## Next week's goal
- Train ModernBERT-base 4-head classifier on train_p with calibration (temperature scaling + reliability diagram + ECE) and implement Integrated Gradients cue attribution to replace token-masking baseline. Generate first SFT dataset for QLoRA on Qwen3.5-0.8B.

## Individual contributions
- Anna Thomas (Data&Eval):
    - Parsed SynthPAI (handled dict/JSON-string profile field), built DataFrame audit, finalized and froze coarse taxonomy for 4 aspects: age bucketing, education, occupation, location regions.
    - Implemented risk baseline: `RISK_DIMENSIONS` with ModernBERT zero-shot, `risk_scores` for 5 dims, `confidence = max(values)`, hard leak check PER/LOC/ORG, evidence extraction.
    - Implemented rewriter baseline: Qwen2.5-0.5B `qwen_rewrite` + `rewrite_until_safe` with safety key `(has_hard_leak, confidence)`, threshold 0.4, 3 attempts, and end-to-end pipeline `text -> PII -> risk -> rewrite -> rescore`.

- Govind (Product):
    - Enabled branch protection on `main` — 1 required approving review, `enforce_admins: true`, stale reviews dismissed, force-push and deletion blocked. This was due in Session 03 and was still off; it is the Session 04 verification item.
    - Reviewed and approved PR #14, catching that the proposed north-star value quoted one attribute's delta from a row where the rewrite left the name, email and phone intact. The north-star block above is the corrected version.
    - Wrote the lean canvas (`docs/lean-canvas.md`) — the guidelines require one and the repo had none, so previous "lean canvas changes" entries had nothing to diff against.
    - Built `src/product/risk_bands.py` + 16 tests: score-to-band translation with overall = max across attributes, unmeasured attributes omitted rather than defaulted to safe, and output phrased as attacker inference rather than fact about the user.
    - Wrote `docs/product/user_test_scenarios.md`: six synthetic scenarios, facilitator script, capture format — unblocks external user sessions without collecting anyone's real data.
    - Reviewed PR #15 (Engineering), catching that the branch carried a pre-#14 copy of this report which would have reverted the north-star and taxonomy corrections; resolved it by taking `main` whole. Filed #21 for the reproducibility bug found in the same review.
    - Audited the model stack and SynthPAI labels against what is actually published: the Qwen3.5 models named as our rewriter are vision-language models, and our frozen location taxonomy has four US-region classes for a dataset that is 7% US. Research only, no code (PR #22, open).
    - Reviewed PRs #24 and #25 (roster) and consolidated them into this PR.
    - (evidence: #17, PR #18; PR #19; PR #22 open; reviews on #14, #15, #24, #25; #21; branch protection settings on `main`)

- Chatur Bandaru (Engineering):
    - Set up the repo workflow: labels, the Session 04 milestone, and issues #2–#10 (including retroactive issues for Anna's Session 04 work, linked to PR #1).
    - Fixed the live-test logging so a user session produces one clean CSV row per interaction, with participant and scenario IDs, `latency_s`, and an accept/edit/reject `decision` field. The pre-fix log was malformed — each interaction spanned two misaligned rows — and `data/logs/README.md` documents that honestly rather than deleting it.
    - Made the profile-split and taxonomy artifacts reproducible via `scripts/make_artifacts.py` and committed them.
    - Reviewed and approved PR #1.
    - (evidence: PR #15, closes #8 and #9; PR #1 review; issues #2–#10. PR #16 was closed without merging, so its evidence-audit annotations have not landed — carried to Session 05.)

- Yiwei Jin (Users&Research):
    - Gave the approving reviews that unblocked PR #15 and PR #19 under the new branch-protection rule.
    - Opened issue #20 and PRs #24 and #25 correcting the member roster, which is what identified that two hats had been left unassigned in the frontmatter.
    - No external user session was run this week. See blockers.
    - (evidence: reviews on #15 and #19; #20; PRs #24, #25 — cherry-picked into this PR with authorship preserved)

- Bingqi Lian (Operations):
    - Added as a repository collaborator and opened PR #23 against the member roster; it was closed in favour of #25 and its content is carried here.
    - No other repo work this week, and the project board the Operations hat owns does not exist yet. Stating this plainly rather than padding the entry.
    - (evidence: PR #23, closed)

## Lean canvas changes (if any)

The canvas now exists as a file — `docs/lean-canvas.md`, PR #18. From Session 05 this section diffs against it instead of restating it.

- Distribution (new, previously unstated): our only surface is a Colab notebook, which reaches no users. The product belongs at the moment of sending, realistically a browser extension. Open question is whether anyone tolerates an extra step before send — cheaper to measure as abandonment in a user session than to build.
- Cost per request (new): ~$0 marginal, local-first. Latency is unmeasured and is the number the local-first premise depends on; three rewrite attempts through a 0.5B model on CPU is the risk.
- North-star metric (changed): from a single-attribute location delta to overall risk delta (max across attributes) with a utility guardrail, pending the held-out adversary. Reason is in Metrics snapshot.
- Biggest risk (sharpened): every number so far comes from the same estimator that drives the rewrite — a closed loop that can only flatter us. Cheapest test is running the held-out attacker against the rewrites we already have: no training, no new data, one script.
- User segment: No change. We continue to focus on privacy-conscious users who share personal context when interacting with LLMs.
- Problem and value proposition: Problem is refined. Explicit PII redaction is not enough because inferential leakage happens through ordinary clues that Presidio misses, for example Green Line implies Boston location even when explicit PII is scrubbed. Value proposition is now clearer as a local-first system that runs ModernBERT and Qwen 0.5B on device, highlights risk cues, rewrites text, and quantifies the privacy utility tradeoff with risk delta and utility similarity.
- Risks and mitigation: Two new risks added. First, SynthPAI is synthetic so results may not transfer to real populations. Second, Qwen 0.5B has quality limits and leaks entities, requiring our _hard_scrub fallback. Mitigation is to consider a 2B fallback model if quality remains low.

