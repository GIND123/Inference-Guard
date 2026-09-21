# data/logs

Raw logs from the live-test loop in `InferenceGuard_Implementation.ipynb`
(section *End-to-End Pipeline Integration*, cell starting `# User test`).

## Privacy rule

- Enter **synthetic scenario text only**. Never type real personal information.
- Participants are identified only by a pseudonymous ID (`P01`, `P02`, ...), never by name.
  The mapping from ID to person is not stored in this repo.

## Files

### `session04_usage_log.csv` (pre-fix, kept as-is)

Produced **before** the logging fix (issue #8). It is malformed:

- The header is a 27-column union of two schemas (a 7-column one and an older 20-column one).
- Each of the 3 interactions spans **two misaligned rows**: the first row fills
  `before_confidence` ... `delta_confidence`; the second row repeats `before_age` and fills
  `timestamp`, `original`, `protected`, the per-dimension scores, `util_sim`, and evidence columns.

To read one interaction, combine each odd data row with the row that follows it.
It is kept unchanged as the raw Session 04 artifact.

### `session04_user_test_<participant_id>_<YYYYMMDD-HHMM>.csv` (current schema)

One file per participant session, **one row per interaction**, appended after every
interaction so a crash does not lose data. Floats are rounded to 3 places.

| column | meaning |
|---|---|
| `timestamp` | ISO 8601 time the row was logged |
| `participant_id` | pseudonymous ID, e.g. `P01` |
| `scenario_id` | synthetic scenario the message came from, e.g. `S01` |
| `original` | message as typed |
| `protected` | output of `rewrite_until_safe(original)` |
| `before_location`, `before_education`, `before_occupation`, `before_age` | zero-shot risk per dimension on `original` |
| `before_confidence` | max over all risk dimensions on `original` |
| `after_location`, `after_education`, `after_occupation`, `after_age` | zero-shot risk per dimension on `protected` |
| `after_confidence` | max over all risk dimensions on `protected` |
| `delta_confidence` | `before_confidence - after_confidence` |
| `delta_location` | `before_location - after_location` |
| `has_hard_leak_before`, `has_hard_leak_after` | NER found a PER/LOC/ORG entity in `original` / `protected` |
| `util_sim` | `util_sim(original, protected)`: cosine similarity of `all-MiniLM-L6-v2` embeddings |
| `latency_s` | wall-clock seconds for the `rewrite_until_safe` call |
| `decision` | `would_use` (y), `would_use_with_edits` (e), `would_not_use` (n) |
| `comment` | optional one-line participant comment |

`confidence` is the max over all `RISK_DIMENSIONS`, which include `name`, so it can
exceed every per-dimension column logged here.
