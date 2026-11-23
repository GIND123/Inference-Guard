# InferenceGuard — 50-Persona 10-Domain GPU Benchmark Report

## Executive Summary
- **Total Personas Evaluated:** 50 across 10 domains (5 personas/domain)
- **Hardware:** CUDA=True | Peak VRAM=1830.2 MB (`UtilityEvaluator` on CPU, `ModernBERT` + `Qwen2.5-1.5B` 4-bit NF4 LoRA on `cuda:0`)
- **Rewrite Engine Breakdown:** `heuristic`: 33/50 (66%), `qwen_adapter`: 16/50 (32%), `presidio_fallback`: 1/50 (2%)
- **Mean Inferential Risk:** `0.8063` → **`0.4897`** (**`-39.3%`** relative reduction)
- **Zero Hard-PII Leakage Rate:** **`100.0%`** (50/50)
- **Zero Regex/NER Scar Rate:** **`98.0%`** (49/50)
- **Critical Number / Order ID / Statute Preservation:** **`98.0%`** (49/50)
- **Mean Cosine Similarity / Utility:** `0.9009` / `0.6633`
- **Mean Latency:** `4449 ms` (total wall time `222.5 s`)

## Per-Domain Results

| Domain | Mode | Qwen GPU | Risk Before | Risk After | Reduction | Cosine | Zero PII | IDs Kept |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Customer Support (B2B) | `business_support` | 1/5 | 0.818 | 0.334 | -59.1% | 0.819 | 5/5 | 5/5 |
| Healthcare & Clinical | `personal_balanced, personal_strict` | 3/5 | 0.845 | 0.544 | -35.6% | 0.882 | 5/5 | 5/5 |
| Software & Engineering | `personal_balanced` | 2/5 | 0.816 | 0.517 | -36.7% | 0.939 | 5/5 | 5/5 |
| Legal & Compliance | `personal_balanced` | 1/5 | 0.802 | 0.728 | -9.2% | 0.939 | 5/5 | 5/5 |
| Finance & Actuarial | `personal_balanced` | 0/5 | 0.803 | 0.430 | -46.4% | 0.918 | 5/5 | 5/5 |
| Academia & Education | `personal_balanced` | 2/5 | 0.888 | 0.756 | -14.8% | 0.883 | 5/5 | 4/5 |
| Workplace & HR | `personal_balanced, personal_strict` | 3/5 | 0.765 | 0.450 | -41.2% | 0.878 | 5/5 | 5/5 |
| Housing & Relocation | `personal_balanced` | 2/5 | 0.797 | 0.330 | -58.6% | 0.893 | 5/5 | 5/5 |
| Travel & Logistics | `personal_balanced` | 0/5 | 0.779 | 0.487 | -37.5% | 0.938 | 5/5 | 5/5 |
| Small Business & E-Commerce | `personal_balanced` | 2/5 | 0.750 | 0.321 | -57.2% | 0.920 | 5/5 | 5/5 |

## Highlighted Before / After Rewrites

### `cs_01` — Customer Support (B2B) (`heuristic`, `0.94` → `0.42`)
- **Original:** Hi, I'm Sarah Chen at 42 Boylston St, Apt 4B (sarah.chen@gmail.com). I'm a 62-year-old retired teacher in Boston on a fixed income—can you waive the $45 late fee on Order #88412?
- **Protected:** Hi, I'm PERSON_1 at ADDRESS_1 (EMAIL_1). I'm a teacher in my area on a fixed income—can you waive the $45 late fee on Order #88412?
- **Aliases:** `{'Sarah Chen': 'PERSON_1', '42 Boylston St, Apt 4B': 'ADDRESS_1', 'sarah.chen@gmail.com': 'EMAIL_1'}`

### `cs_02` — Customer Support (B2B) (`heuristic`, `0.77` → `0.42`)
- **Original:** My name is David Alvarez (555-234-8891) at 718 Oak Ave, Suite 200 in Austin, TX. As a 41-year-old single dad, I missed the renewal window on TKT-4402—can you honor the 15% loyalty discount?
- **Protected:** My name is PERSON_1 (PHONE_1) at ADDRESS_1. As a single dad, I missed the renewal window on TKT-4402—can you honor the 15% loyalty discount?
- **Aliases:** `{'David Alvarez': 'PERSON_1', '555-234-8891': 'PHONE_1', '718 Oak Ave, Suite 200 in Austin, TX': 'ADDRESS_1'}`

### `hc_01` — Healthcare & Clinical (`heuristic`, `0.87` → `0.84`)
- **Original:** I am a 28-year-old pediatric resident in Boston double-checking high-dose amoxicillin (80 mg/kg/day) for a 4-year-old patient weighing 16 kg with acute otitis media.
- **Protected:** I am a 28-year-old medical resident in Boston double-checking high-dose amoxicillin (80 mg/kg/day) for a 4-year-old patient weighing 16 kg with acute otitis media.
- **Aliases:** `{}`

### `sw_01` — Software & Engineering (`heuristic`, `0.88` → `0.42`)
- **Original:** As a 28-year-old backend architect in Bengaluru designing a multi-region Postgres cluster for 99.9% uptime, should I use logical replication or Citus sharding?
- **Protected:** As a backend architect designing a multi-region Postgres cluster for 99.9% uptime, should I use logical replication or Citus sharding?
- **Aliases:** `{}`

### `lg_01` — Legal & Compliance (`heuristic`, `0.78` → `0.78`)
- **Original:** As a 37-year-old family law attorney in Atlanta preparing a custody brief under O.C.G.A. § 19-9-3, how should I structure the best-interest factors?
- **Protected:** As a 37-year-old family law attorney preparing a custody brief under O.C.G.A. § 19-9-3, how should I structure the best-interest factors?
- **Aliases:** `{}`

### `fn_01` — Finance & Actuarial (`heuristic`, `0.76` → `0.52`)
- **Original:** As a pension actuary in Hartford modeling defined-benefit liabilities at a 4.5% discount rate, how sensitive is duration to mortality improvement tables?
- **Protected:** As a pension actuary modeling defined-benefit liabilities at a 4.5% discount rate, how sensitive is duration to mortality improvement tables?
- **Aliases:** `{}`

### `ac_01` — Academia & Education (`qwen_adapter`, `0.88` → `0.77`)
- **Original:** I'm a 21-year-old 3rd-year undergrad in Boston—how does Pell Grant eligibility carry over if I take 6 summer credits instead of full-time?
- **Protected:** For someone on my current income: How does Pell Grant eligibility carry over if I take 6 summer credits instead of full-time?
- **Aliases:** `{}`

### `hr_01` — Workplace & HR (`qwen_adapter`, `0.75` → `0.75`)
- **Original:** I'm a 36-year-old operations analyst in Chicago who discovered an $18,000 vendor kickback—how do I document an anonymous whistleblower report safely?
- **Protected:** I'm a 36-year-old operations analyst in my city who discovered an $18,000 vendor kickback—how do I document an anonymous whistleblower report safely?
- **Aliases:** `{}`
