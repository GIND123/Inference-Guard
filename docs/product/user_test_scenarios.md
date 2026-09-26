# Synthetic user-test scenarios

**Owner:** Govind (Product), for use by Users&Research · **Created:** 2026-09-21 (Session 04)

Every scenario below is invented. No persona corresponds to a real person, and no
participant is ever asked to supply their own personal information. This is both an
ethics requirement and a practical one  --  we cannot commit raw evidence to a public repo
if that evidence contains somebody's real address.

## Why this file exists

The course's evidence standard is narrow, and it is worth restating before anyone runs
a session:

**Counts.** A real user, not on this team, using the running product, captured while it
happens  --  screen recording, timestamped usage log, or a task test with completion and
timing, plus dated notes.

**Does not count.** Interviews about the idea. Reactions to mockups or slides. Feedback
on the report. Team members using our own product. Anything written up from memory weeks
later.

`data/logs/session04_usage_log.csv` is the team testing itself, so it is a dev log, not
user evidence, and the report should keep saying so.

Commit raw artifacts **the same week the claim is made**. Evidence filed late is graded
as no evidence.

---

## How to run a session (~20 minutes)

1. Start the pipeline and share your screen, or have the participant drive.
2. Give them the **Situation** only. Do not read them the prompt text  --  hand it over as
   something they are about to send, and let them read it themselves.
3. Ask them to send it and stop. **Before** they request a rewrite: *"What is this
   telling you?"* This measures whether the risk panel is legible on its own, which is
   the thing we most need to know and the thing we will be most tempted to coach.
4. Let them request the rewrite. Ask whether they would send the rewritten version
   as-is, with edits, or not at all  --  and why.
5. Record the outcome. Do not defend the product mid-session; note the objection and
   move on.

Start the recording before step 1. A session with good findings and no artifact is a
session we cannot cite.

---

## Scenarios

Each is written so the leak is inferential  --  remove the obvious identifiers and the
attributes are still recoverable.

### S-01 · University advice  --  `education`, `location`, `age`

**Situation.** You are asking an AI whether to move apartments before your next work
placement starts.

**Prompt.** *"My second co-op starts in January and my Green Line commute is already
brutal. Is it worth moving closer before the spring semester, or should I stick it out
until I graduate?"*

**Expected leak.** "co-op" + "semester" + "graduate" → current undergraduate; "second
co-op" → mid-degree, so roughly 20 - 22; "Green Line" → one metro area.
**Watch for.** Whether they notice location risk at all. There is no city named here.

### S-02 · Workplace conflict  --  `occupation`, `age`, `education`

**Situation.** You want advice on a disagreement with a colleague.

**Prompt.** *"My manager keeps assigning me the on-call rotation even though I'm the
only one on the team without a PhD. I've been doing ML infra for about eight years. How
do I raise this without it sounding like I'm complaining about workload?"*

**Expected leak.** Occupation to a narrow specialism; eight years' experience → roughly
30 - 35; education inferable by negation.
**Watch for.** Whether the rewrite keeps the *grievance* intact. If it generalises away
the PhD detail the advice becomes useless  --  this is the clearest utility-failure
scenario we have, and it should be run every week as a regression.

### S-03 · Health question  --  `age`, `location`, out-of-scope attribute

**Situation.** You are asking about a medication interaction.

**Prompt.** *"I started metformin last month and I'm also on a statin. I'm 58 and my
GP at the practice in Didsbury didn't seem worried, but should I be?"*

**Expected leak.** Explicit age; neighbourhood-level location; "GP" and "practice" →
UK. A health condition is also inferable, which v1 does **not** protect.
**Watch for.** Whether the participant assumes the health inference is covered. If they
do, that is the over-trust failure the canvas calls our highest-cost risk, and it is a
finding worth more than a quality complaint.

### S-04 · Travel planning  --  `occupation`, `location`, `age`

**Situation.** You are planning time off.

**Prompt.** *"I've got 12 days of leave to use before the fiscal year closes in March
and my partner teaches, so we're locked into half-term. Somewhere warm, direct flights
from the regional airport, not too expensive."*

**Expected leak.** "fiscal year closes in March" + "half-term" + "regional airport" →
country and rough region; partner's occupation is third-party leakage, which v1 does
not handle.
**Watch for.** Reaction to a rewrite that removes the constraints the request depends
on. "Somewhere warm" with no dates and no airport is unanswerable.

### S-05 · Writing assistance  --  explicit PII, as a control

**Situation.** You want help tightening a paragraph of your CV.

**Prompt.** *"Fix this: John Doe | 123 Oak Ave, Denver CO | jdoe@example.com | (303)
555-0192 | Born 1991  --  Senior analyst, six years in healthcare claims."*

**Expected leak.** Explicit identifiers, which redaction should already handle.
**Watch for.** This is the control, and it is currently a **known failure**. On the
Session 04 dev log the rewrite scrubbed the city and left the name, email and phone
intact. Run it anyway. If a participant spots it before we tell them, that is the
strongest possible evidence for prioritising the fix.

### S-06 · Multi-turn accumulation  --  cumulative leakage

**Situation.** A four-message conversation. Send them one at a time and watch the
leakage number after each.

1. *"What's a reasonable rent to budget for a one-bedroom?"*
2. *"I take the Green Line in most mornings, so somewhere on that side."*
3. *"My co-op ends in December and I'd want the lease to start after that."*
4. *"My professor mentioned the Canvas deadline moved, so I've got more time to look."*

**Expected leak.** No single turn identifies anyone. By turn 4: undergraduate, one
metro area, current co-op, 18 - 24.
**Watch for.** Whether the participant is surprised by the total. This scenario is the
one the whole project's differentiation rests on, and it needs a working conversation
view before it can be run properly  --  Session 08.

---

## Recording format

One JSON object per participant per scenario, in `user_studies/sessionNN/`:

```json
{
  "participant_id": "anonymous-01",
  "scenario_id": "S-02",
  "date": "2026-09-22",
  "facilitator": "<github handle>",
  "artifact": "user_studies/session04/anonymous-01.mp4",
  "understood_warning": true,
  "warning_in_their_words": "it thinks it can tell where I work from the PhD thing",
  "requested_rewrite": true,
  "rewrite_decision": "accept_with_edits",
  "edit_made": "put 'PhD' back in, the question doesn't work without it",
  "usefulness": 3,
  "would_use_before_sending": false,
  "comments": "said the extra step is fine for this but not for everyday questions",
  "time_to_decision_seconds": 74
}
```

`warning_in_their_words` is the field that matters most. A participant clicking "yes, I
understood" tells us nothing; a participant paraphrasing the warning back wrong tells us
exactly what to change. `would_use_before_sending` is the distribution risk from the
canvas  --  it needs to be asked every session, even when the answer is discouraging.

Record rejections in full. A rejected rewrite with a stated reason is more useful than an
accepted one with none, and "what did not work" is a scored section of every report.
