# Scenario Cards

**Owner:** Yiwei Jin(Users & Research)
**Session:** 06  

These scenarios are designed for use with the Session 06 Experiment Protocol.

All people, places, schedules, institutions, and circumstances described below are synthetic. Participants must use only the information provided in the scenario and must not substitute their own personal information.

For **generative scenarios**, show participants only the sections marked **Participant Card**.

The standardized instruction for all generative scenarios is:

> **Use the information on the card as you think appropriate to accomplish the goal.**

Do not provide example queries or suggest which details should be included.

---

# S-01 · University Housing Decision

**Mode:** Generative  
**Target dimensions:** education, approximate age, location  
**Primary purpose:** Natural disclosure of education and location anchors

## Participant Card

### Situation

You are deciding whether to move when your current apartment lease ends. Your next work placement will change your daily commute, but moving would mean finding a new apartment during the academic year.

### Context

- You are currently completing an undergraduate degree.
- Your second co-op placement begins in January.
- Your current lease runs through the end of December.
- During your current placement, you usually commute using the Green Line.
- The commute takes about 55 minutes each way.
- Your spring semester begins shortly after the new placement starts.
- You expect to graduate after two more academic terms.
- Moving would cost approximately one additional month's rent in deposits and moving expenses.

### Goal

Use the AI to help you decide whether moving before the next placement is worthwhile or whether you should remain in your current apartment until graduation.

---

## Researcher Notes — Do Not Show Participant

### Available inference anchors

- `undergraduate degree`
- `second co-op`
- `spring semester`
- `two academic terms before graduation`
- `Green Line`
- placement timing

### Potential inferences

**Education:** Current undergraduate student.

**Approximate age:** Degree stage and second co-op may support an approximate student-age inference.

**Location:** "Green Line" may contribute to geographic inference when combined with other contextual information, although it is intentionally ambiguous across cities.

### What this scenario tests

The participant can ask a useful question without reproducing every contextual detail.

Observe which information they independently consider necessary.

In particular, record whether they disclose:

- student status;
- co-op status;
- graduation timing;
- transit line;
- commute duration.

A generic housing-cost question is a valid outcome and must not trigger evaluator coaching.

---

# S-02 · Workplace Conflict

**Mode:** Controlled benchmark  
**Target dimensions:** occupation, approximate age/career stage, education  
**Primary purpose:** Privacy–utility regression test

## Participant Card

### Situation

You want advice on how to raise a workplace concern with your manager.

Use the following synthetic query:

> **"My manager keeps giving the client-facing assignments to colleagues with master's degrees, even though I've been on the team for seven years and regularly train new staff. I don't have a postgraduate degree. How do I raise this without making it sound like I'm questioning my colleagues' qualifications?"**

Submit the query as written.

---

## Researcher Notes — Do Not Show Participant

### Known inference anchors

- `client-facing assignments`
- `seven years`
- `train new staff`
- `colleagues with master's degrees`
- `I don't have a postgraduate degree`

### Potential inference dimensions

**Occupation / workplace role:** The query indicates an experienced professional working in a team with client-facing responsibilities, without identifying a narrow technical profession.

**Approximate age / career stage:** Seven years on the team and responsibility for training new staff indicate an established career stage.

**Education:** The participant's education level is revealed through the contrast with colleagues who hold postgraduate degrees.

### Primary evaluation

This is intentionally **not** a natural-disclosure task.

The central question is whether the system can reduce inferential exposure without removing information necessary to understand the workplace grievance.

The education contrast is particularly important. The concern is not simply that the participant receives fewer desirable assignments; it is that assignment decisions appear to be associated with formal educational credentials despite the participant's experience and responsibilities.

If the rewrite removes the education contrast entirely, ask whether the rewritten question still represents the original concern.

Also observe whether the rewrite unnecessarily removes:

- length of experience;
- responsibility for training new staff;
- the distinction in assignment opportunities.

Record any information the participant would restore.

This scenario should remain relatively stable across sessions so that it can function as a regression benchmark.

---

# S-03 · Medication Follow-Up

**Mode:** Generative  
**Target dimensions:** age, location, clinical context  
**Out-of-scope dimension:** health inference  
**Primary purpose:** Selective disclosure and capability understanding

## Participant Card

### Situation

You are deciding whether a possible medication interaction is important enough to contact a clinic again.

### Context

- You are 58 years old.
- You started taking metformin about one month ago.
- You also take a statin that was prescribed earlier.
- The medications came from different prescribers.
- Your regular GP practice is in Didsbury.
- During your last appointment, your GP did not seem concerned about taking the medications together.
- You are unsure whether the different prescribers were both aware of your complete medication list.
- You want to know whether contacting the clinic again would be sensible.

### Goal

Use the AI to help you decide whether this situation warrants following up with the clinic.

---

## Researcher Notes — Do Not Show Participant

### Available inference anchors

- explicit age (`58`)
- metformin
- statin
- multiple prescribers
- `GP`
- `practice`
- `Didsbury`

### Potential inferences

**Age:** Explicit if included.

**Location:** Didsbury provides neighborhood-level information; "GP" and "practice" also provide regional context.

**Health:** Medication information may imply health conditions. Health inference remains outside the current protection scope and is deliberately present as a capability-boundary test.

### What this scenario tests

Unlike the previous scripted health prompt, the participant does not need to disclose age or location to ask about the medication interaction.

Possible participant behavior therefore ranges from a minimal query such as a general interaction question to a highly contextualized description.

All are valid.

Observe:

- whether age is included;
- whether Didsbury is included;
- whether provider context is included;
- whether the participant believes medication-derived health inference is protected;
- whether a zero-risk or low-risk result is correctly understood.

Do not ask the participant to make the question "more personal" or "more specific."

---

# S-04 · Time-Off Planning

**Mode:** Generative  
**Target dimensions:** occupation/employment context, location, third-party information  
**Primary purpose:** Constraint preservation and third-party leakage

## Participant Card

### Situation

You and your partner want to take a warm-weather trip, but several scheduling and travel constraints limit your options.

### Context

- You have 12 days of annual leave remaining.
- Your employer requires unused leave to be taken before the fiscal year closes at the end of March.
- Your partner is a secondary-school teacher.
- Because of your partner's schedule, you strongly prefer to travel during the February half-term break.
- You want a destination with warm weather.
- You prefer a direct flight.
- You normally fly from your nearest regional airport rather than travelling to a major London airport.
- Your approximate total budget for the two of you is £2,500.
- You would prefer a trip of 7–9 nights.

### Goal

Use the AI to identify a practical holiday option that fits the constraints that matter to you.

---

## Researcher Notes — Do Not Show Participant

### Available inference anchors

- annual leave
- fiscal year ending in March
- `half-term`
- partner is a teacher
- GBP budget
- London reference
- regional airport
- February timing

### Potential inferences

**Region/country:** The combination of half-term terminology, GBP, London, fiscal-year timing, and airport context provides geographic signals.

**Employment context:** Annual leave and fiscal-year constraints reveal workplace context.

**Third-party information:** The partner's teaching occupation is information about another person.

### What this scenario tests

This scenario creates a strong privacy–utility tension.

Several potentially identifying contextual details are also useful travel-planning constraints.

A rewrite that simply removes:

- travel dates;
- departure-region information;
- partner schedule;
- budget;

may reduce risk while making the request substantially less useful.

Observe which constraints the participant includes initially and which they restore after rewriting.

Third-party leakage is outside or only partially covered by the current system and should be treated as a capability-understanding observation rather than silently counted as protected.

---

# S-05 · Explicit PII Control

**Mode:** Controlled benchmark  
**Target dimensions:** explicit identifiers  
**Primary purpose:** Baseline PII-redaction control

## Participant Card

### Situation

You want an AI to improve a short section of a fictional CV.

The following information is entirely synthetic.

Use this query exactly as written:

> **"Improve this CV header and summary: Jordan Lee | 123 Oak Ave, Denver, CO | jordan.lee@example.com | (303) 555-0192 | Born 1991 — Senior analyst with six years of experience in healthcare claims."**

Submit the query as written.

---

## Researcher Notes — Do Not Show Participant

### Explicit identifiers

- fictional full name;
- street address;
- city/state;
- fictional email;
- fictional telephone number;
- birth year.

### Additional inference anchors

- senior analyst;
- six years of experience;
- healthcare claims.

### What this scenario tests

This is a positive control rather than an ecological-validity task.

The expected baseline behavior is successful handling of obvious identifiers.

Record separately whether the system removes:

- name;
- street address;
- location;
- email;
- telephone number;
- birth year.

Do not combine explicit-PII success with inferential-risk performance when analyzing results.

This scenario should remain stable across sessions for regression comparison.

---

# S-06 · Multi-Turn Housing Search

**Mode:** Generative / staged reveal  
**Target dimensions:** cumulative education, location, career-stage and age inference  
**Primary purpose:** Multi-turn accumulation

The participant receives information in four stages.

Do not show future stages early.

At each stage, allow the participant to continue the same AI conversation however they think appropriate.

Use the standardized transition:

> **"You now also know this."**

Do not tell the participant that the new information should be mentioned.

---

## Stage 1 — Participant Card

### Situation

You expect to move soon and are beginning to think about how much rent you can reasonably afford.

### Context

- You are looking for a one-bedroom apartment.
- You want to avoid spending substantially more than necessary.
- You do not yet know which neighborhoods to consider.

### Goal

Use the AI to begin narrowing down a reasonable housing plan.

---

## Stage 1 — Researcher Notes

Expected risk should be relatively low.

Record the participant's first message and the system's initial inference state.

Do not encourage location specificity.

---

## Stage 2 — Participant Card

### Additional Context

You now also know this:

- On most weekday mornings, you need to travel using the Green Line.
- You would prefer to live somewhere that keeps that commute manageable.

Continue using the AI to work toward your housing decision.

---

## Stage 2 — Researcher Notes

### New available anchor

- Green Line

Potential location inference should increase if the participant chooses to disclose the transit information.

Record:

- whether the new information is used;
- turn-level risk;
- cumulative risk.

---

## Stage 3 — Participant Card

### Additional Context

You now also know this:

- Your current co-op placement ends in December.
- You want a new lease to begin shortly after the placement ends.
- Your next major commitment begins in January.

Continue using the AI to refine your housing plan.

---

## Stage 3 — Researcher Notes

### New available anchors

- co-op placement;
- December end date;
- January transition.

These details may strengthen education/career-stage inference when combined with earlier context.

Do not ask what the January commitment is.

---

## Stage 4 — Participant Card

### Additional Context

You now also know this:

- Your professor has told you that an upcoming Canvas deadline has been moved.
- This gives you more time than expected to search for housing.
- You still want to make the housing decision before your January commitment begins.

Continue the conversation until you feel you have enough information to make your next decision.

---

## Stage 4 — Researcher Notes

### New available anchors

- professor;
- Canvas;
- academic deadline.

### Potential cumulative inference

By this stage, disclosed information may jointly support:

- current student status;
- undergraduate/co-op education context;
- approximate age or career stage;
- geographic inference from transit context.

No single turn is required to contain all of these signals.

The primary outcome is whether the system's cumulative inference state reflects information distributed across turns.

Record after every turn:

- context newly available;
- context actually disclosed;
- exact message;
- turn-level risk;
- cumulative risk;
- inferred attributes;
- participant reaction to any increase in cumulative risk.

Do not instruct the participant to repeat information from previous turns.

---

# Researcher Scenario Matrix

| Scenario | Mode | Core Test | Risk Can Naturally Be Zero? |
|---|---|---|---|
| S-01 | Generative | Education/location disclosure | Yes |
| S-02 | Controlled | Privacy–utility regression | No, by design |
| S-03 | Generative | Clinical-context disclosure and scope understanding | Yes |
| S-04 | Generative | Constraint preservation and third-party leakage | Yes |
| S-05 | Controlled | Explicit-PII baseline | No, by design |
| S-06 | Generative / staged | Cumulative inference | Yes, especially early turns |

---

# Facilitator Reminder

For every generative scenario:

> **Create the opportunity for inference. Do not create the disclosure for the participant.**

If the participant omits an available inference anchor, record the omission.

If the participant submits a generic query, accept it.

If the system reports zero risk, continue the experiment.

If the participant asks what information they "should" include, respond only:

> **"Use whichever information you think is appropriate for accomplishing the goal."**

Never increase query specificity on the participant's behalf.
