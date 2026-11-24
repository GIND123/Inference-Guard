"""50-Persona 10-Domain Benchmark with Live Qwen GPU & Dual-Edition Modes.

Evaluates InferenceGuard across 10 real-world domains (5 personas each), including
B2B Customer Support tickets with street addresses & Order/Ticket/Invoice IDs,
Legal statutory citations, Healthcare dosages, Academia (Pell Grant / ordinals),
and "As a [Role]..." self-disclosures.

Saves:
  - reports/persona_50_domain_results.json
  - reports/persona_50_domain_report.md
"""

from __future__ import annotations

import json
import re
import secrets
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import web.api as api


PERSONAS: List[Dict[str, Any]] = [
    # 1. Customer Support (B2B Gateway)
    {
        "id": "cs_01",
        "domain": "Customer Support (B2B)",
        "privacy_mode": "business_support",
        "must_keep": ["#88412", "$45"],
        "text": "Hi, I'm Sarah Chen at 42 Boylston St, Apt 4B (sarah.chen@gmail.com). I'm a 62-year-old retired teacher in Boston on a fixed income—can you waive the $45 late fee on Order #88412?",
    },
    {
        "id": "cs_02",
        "domain": "Customer Support (B2B)",
        "privacy_mode": "business_support",
        "must_keep": ["TKT-4402", "15%"],
        "text": "My name is David Alvarez (555-234-8891) at 718 Oak Ave, Suite 200 in Austin, TX. As a 41-year-old single dad, I missed the renewal window on TKT-4402—can you honor the 15% loyalty discount?",
    },
    {
        "id": "cs_03",
        "domain": "Customer Support (B2B)",
        "privacy_mode": "business_support",
        "must_keep": ["INV-901", "$240"],
        "text": "Please send a corrected receipt for INV-901 ($240) to priya.nair@outlook.com and confirm delivery to 1205 Pine Road, Unit 9. I'm a 29-year-old graduate student in Seattle juggling two jobs.",
    },
    {
        "id": "cs_04",
        "domain": "Customer Support (B2B)",
        "privacy_mode": "business_support",
        "must_keep": ["#55190", "$89"],
        "text": "Hi support, Marcus Webb here at 88 Maple Street, Apt 3C. I'm a 53-year-old warehouse supervisor in Chicago dealing with night-shift fatigue and accidentally ordered twice on Order #55190—can you refund the $89 duplicate charge?",
    },
    {
        "id": "cs_05",
        "domain": "Customer Support (B2B)",
        "privacy_mode": "business_support",
        "must_keep": ["TKT-7731", "$120"],
        "text": "This is Elena Gomez (elena.gomez@yahoo.com) at 310 Cedar Blvd in Denver, CO. I'm a 36-year-old clinic nurse and need TKT-7731 expedited so the $120 replacement arrives before Friday.",
    },

    # 2. Healthcare & Clinical
    {
        "id": "hc_01",
        "domain": "Healthcare & Clinical",
        "privacy_mode": "personal_balanced",
        "must_keep": ["16 kg", "80"],
        "text": "I am a 28-year-old pediatric resident in Boston double-checking high-dose amoxicillin (80 mg/kg/day) for a 4-year-old patient weighing 16 kg with acute otitis media.",
    },
    {
        "id": "hc_02",
        "domain": "Healthcare & Clinical",
        "privacy_mode": "personal_balanced",
        "must_keep": ["1:8"],
        "text": "As a 44-year-old ICU charge nurse in Chicago dealing with night-shift staffing at a 1:8 ratio, what evidence-based handoff checklists reduce medication errors?",
    },
    {
        "id": "hc_03",
        "domain": "Healthcare & Clinical",
        "privacy_mode": "personal_balanced",
        "must_keep": ["500 mg"],
        "text": "I'm a 35-year-old clinical pharmacist in Seattle comparing renal dosing adjustments for 500 mg levofloxacin when CrCl drops below 50 mL/min.",
    },
    {
        "id": "hc_04",
        "domain": "Healthcare & Clinical",
        "privacy_mode": "personal_strict",
        "must_keep": ["12"],
        "text": "I'm a 51-year-old physical therapist in Denver recovering from my own rotator cuff tear—what 12-week progressive loading benchmarks are safest?",
    },
    {
        "id": "hc_05",
        "domain": "Healthcare & Clinical",
        "privacy_mode": "personal_balanced",
        "must_keep": ["30"],
        "text": "As a 31-year-old dental hygienist in Pittsburgh, how should I explain a 30-day periodontal recall plan clearly to anxious first-time patients?",
    },

    # 3. Software & Engineering
    {
        "id": "sw_01",
        "domain": "Software & Engineering",
        "privacy_mode": "personal_balanced",
        "must_keep": ["99.9%"],
        "text": "As a 28-year-old backend architect in Bengaluru designing a multi-region Postgres cluster for 99.9% uptime, should I use logical replication or Citus sharding?",
    },
    {
        "id": "sw_02",
        "domain": "Software & Engineering",
        "privacy_mode": "personal_balanced",
        "must_keep": ["200 ms"],
        "text": "I'm a 33-year-old staff SRE in San Francisco trying to cut p99 API latency below 200 ms on our Kubernetes ingress—what Envoy timeout settings matter most?",
    },
    {
        "id": "sw_03",
        "domain": "Software & Engineering",
        "privacy_mode": "personal_balanced",
        "must_keep": ["16 GB"],
        "text": "I work as a 26-year-old ML engineer in New York fine-tuning a 7B model on a single 16 GB GPU; what LoRA rank and gradient accumulation settings prevent OOM?",
    },
    {
        "id": "sw_04",
        "domain": "Software & Engineering",
        "privacy_mode": "personal_balanced",
        "must_keep": ["40%"],
        "text": "As a 39-year-old frontend lead in Austin migrating a legacy Angular app to React, how can we cut bundle size by 40% without breaking SSR?",
    },
    {
        "id": "sw_05",
        "domain": "Software & Engineering",
        "privacy_mode": "personal_balanced",
        "must_keep": ["3"],
        "text": "I'm a 42-year-old embedded firmware engineer in London debugging SPI DMA race conditions across 3 sensor channels on STM32—any checklist for bus contention?",
    },

    # 4. Legal & Compliance
    {
        "id": "lg_01",
        "domain": "Legal & Compliance",
        "privacy_mode": "personal_balanced",
        "must_keep": ["19-9-3"],
        "text": "As a 37-year-old family law attorney in Atlanta preparing a custody brief under O.C.G.A. § 19-9-3, how should I structure the best-interest factors?",
    },
    {
        "id": "lg_02",
        "domain": "Legal & Compliance",
        "privacy_mode": "personal_balanced",
        "must_keep": ["3553"],
        "text": "I'm a 46-year-old defense paralegal in New York summarizing mitigating sentencing factors under 18 U.S.C. § 3553(a)—can you outline a clear memo template?",
    },
    {
        "id": "lg_03",
        "domain": "Legal & Compliance",
        "privacy_mode": "personal_balanced",
        "must_keep": ["30"],
        "text": "As a 32-year-old privacy counsel in San Francisco advising SaaS teams across US states, how do we harmonize 30-day DSAR deletion deadlines?",
    },
    {
        "id": "lg_04",
        "domain": "Legal & Compliance",
        "privacy_mode": "personal_balanced",
        "must_keep": ["90"],
        "text": "I'm a 29-year-old immigration associate in Boston drafting an RFE response due in 90 days—what objective evidence best supports extraordinary ability?",
    },
    {
        "id": "lg_05",
        "domain": "Legal & Compliance",
        "privacy_mode": "personal_balanced",
        "must_keep": ["17"],
        "text": "As a 48-year-old compliance officer in London auditing GDPR Article 17 erasure logs, what proof satisfies regulators when backups are immutable?",
    },

    # 5. Finance & Actuarial
    {
        "id": "fn_01",
        "domain": "Finance & Actuarial",
        "privacy_mode": "personal_balanced",
        "must_keep": ["4.5%"],
        "text": "As a pension actuary in Hartford modeling defined-benefit liabilities at a 4.5% discount rate, how sensitive is duration to mortality improvement tables?",
    },
    {
        "id": "fn_02",
        "domain": "Finance & Actuarial",
        "privacy_mode": "personal_balanced",
        "must_keep": ["22%"],
        "text": "I'm a 34-year-old CPA in Chicago advising a self-employed client in the 22% federal bracket on whether an S-Corp election saves self-employment tax.",
    },
    {
        "id": "fn_03",
        "domain": "Finance & Actuarial",
        "privacy_mode": "personal_balanced",
        "must_keep": ["36%"],
        "text": "As a 41-year-old mortgage underwriter in Denver reviewing a borrower with a 36% debt-to-income ratio and variable bonus income, what two-year averaging rules apply?",
    },
    {
        "id": "fn_04",
        "domain": "Finance & Actuarial",
        "privacy_mode": "personal_balanced",
        "must_keep": ["$65,000"],
        "text": "I'm a 58-year-old electrical engineer in Pittsburgh planning retirement on $65,000/year—how does a Roth conversion ladder reduce RMDs?",
    },
    {
        "id": "fn_05",
        "domain": "Finance & Actuarial",
        "privacy_mode": "personal_balanced",
        "must_keep": ["60/40"],
        "text": "As a 27-year-old quantitative analyst in New York backtesting a 60/40 portfolio during high-inflation regimes, which Treasury duration hedges drawdowns best?",
    },

    # 6. Academia & Education
    {
        "id": "ac_01",
        "domain": "Academia & Education",
        "privacy_mode": "personal_balanced",
        "must_keep": ["Pell Grant", "3rd-year"],
        "text": "I'm a 21-year-old 3rd-year undergrad in Boston—how does Pell Grant eligibility carry over if I take 6 summer credits instead of full-time?",
    },
    {
        "id": "ac_02",
        "domain": "Academia & Education",
        "privacy_mode": "personal_balanced",
        "must_keep": ["3"],
        "text": "My manager Sarah Chen asked me to outline 3 aims for our post-doc lab's NIH grant renewal before my thesis defense next month.",
    },
    {
        "id": "ac_03",
        "domain": "Academia & Education",
        "privacy_mode": "personal_balanced",
        "must_keep": ["28"],
        "text": "As a 38-year-old high-school chemistry teacher in Seattle with 28 students per lab, what low-cost titration labs work safely in 45-minute periods?",
    },
    {
        "id": "ac_04",
        "domain": "Academia & Education",
        "privacy_mode": "personal_balanced",
        "must_keep": ["2"],
        "text": "I'm a 25-year-old PhD candidate at Carnegie Mellon preparing a 2-page NSF fellowship statement—how should I balance broader impacts with methodology?",
    },
    {
        "id": "ac_05",
        "domain": "Academia & Education",
        "privacy_mode": "personal_balanced",
        "must_keep": ["15"],
        "text": "As a 45-year-old community college advisor in Austin, what 15-minute orientation checklist helps first-generation students avoid financial aid holds?",
    },

    # 7. Workplace & HR
    {
        "id": "hr_01",
        "domain": "Workplace & HR",
        "privacy_mode": "personal_strict",
        "must_keep": ["$18,000"],
        "text": "I'm a 36-year-old operations analyst in Chicago who discovered an $18,000 vendor kickback—how do I document an anonymous whistleblower report safely?",
    },
    {
        "id": "hr_02",
        "domain": "Workplace & HR",
        "privacy_mode": "personal_balanced",
        "must_keep": ["10%"],
        "text": "My director Jordan Lee rejected my 10% raise request even though I led our Denver rollout—can you help me draft a data-backed counterproposal?",
    },
    {
        "id": "hr_03",
        "domain": "Workplace & HR",
        "privacy_mode": "personal_balanced",
        "must_keep": ["4"],
        "text": "As a 30-year-old technical recruiter in Seattle screening candidates for 4 senior roles, what structured interview rubrics reduce bias?",
    },
    {
        "id": "hr_04",
        "domain": "Workplace & HR",
        "privacy_mode": "personal_balanced",
        "must_keep": ["12"],
        "text": "I'm a 49-year-old plant safety coordinator in Pittsburgh managing 12-hour rotating shifts—what fatigue-mitigation schedules work best?",
    },
    {
        "id": "hr_05",
        "domain": "Workplace & HR",
        "privacy_mode": "personal_balanced",
        "must_keep": ["30"],
        "text": "As a 33-year-old remote UX designer in Boston transitioning to a manager role, what should my 30-60-90 day plan prioritize?",
    },

    # 8. Housing & Relocation
    {
        "id": "hs_01",
        "domain": "Housing & Relocation",
        "privacy_mode": "personal_balanced",
        "must_keep": ["$1,200", "$1,950"],
        "text": "I'm a 27-year-old data analyst in Boston comparing a $1,200 apartment with a 55-minute Green Line commute versus a $1,950 studio near work—which makes better financial sense?",
    },
    {
        "id": "hs_02",
        "domain": "Housing & Relocation",
        "privacy_mode": "personal_balanced",
        "must_keep": ["20%"],
        "text": "As a 35-year-old civil engineer in Denver saving for a 20% down payment, should I pause extra student loan payments to build my closing reserve?",
    },
    {
        "id": "hs_03",
        "domain": "Housing & Relocation",
        "privacy_mode": "personal_balanced",
        "must_keep": ["14"],
        "text": "I'm a 43-year-old tenant in San Francisco whose landlord hasn't fixed a broken heater in 14 days—can you draft a polite certified-mail repair notice?",
    },
    {
        "id": "hs_04",
        "domain": "Housing & Relocation",
        "privacy_mode": "personal_balanced",
        "must_keep": ["3"],
        "text": "As a 60-year-old retired librarian in Seattle downsizing from a 3-bedroom home to a condo, what HOA reserve-study red flags should I inspect?",
    },
    {
        "id": "hs_05",
        "domain": "Housing & Relocation",
        "privacy_mode": "personal_balanced",
        "must_keep": ["$2,400"],
        "text": "I'm a 31-year-old postdoctoral fellow relocating to Chicago on a $2,400 monthly housing budget—what lease clauses should I negotiate for early termination?",
    },

    # 9. Travel & Logistics
    {
        "id": "tr_01",
        "domain": "Travel & Logistics",
        "privacy_mode": "personal_balanced",
        "must_keep": ["5"],
        "text": "As a 40-year-old freight dispatcher in Chicago coordinating 5 refrigerated routes across US states, how do I buffer hours-of-service rules during winter storms?",
    },
    {
        "id": "tr_02",
        "domain": "Travel & Logistics",
        "privacy_mode": "personal_balanced",
        "must_keep": ["90"],
        "text": "I'm a 32-year-old freelance photographer in New York planning a 90-day Schengen itinerary—how does the rolling 180-day stay calculator work?",
    },
    {
        "id": "tr_03",
        "domain": "Travel & Logistics",
        "privacy_mode": "personal_balanced",
        "must_keep": ["48"],
        "text": "As a 47-year-old customs broker in Seattle clearing perishable air cargo within 48 hours, what HS code documentation mistakes trigger holds?",
    },
    {
        "id": "tr_04",
        "domain": "Travel & Logistics",
        "privacy_mode": "personal_balanced",
        "must_keep": ["120"],
        "text": "I'm a 29-year-old corporate event planner in Austin organizing a 120-person offsite—what hotel attrition clauses protect us if attendance drops?",
    },
    {
        "id": "tr_05",
        "domain": "Travel & Logistics",
        "privacy_mode": "personal_balanced",
        "must_keep": ["7"],
        "text": "As a 54-year-old airline mechanic in Denver traveling with elderly parents for 7 days in Tokyo, what rail pass options minimize station transfers?",
    },

    # 10. Small Business & E-Commerce
    {
        "id": "sb_01",
        "domain": "Small Business & E-Commerce",
        "privacy_mode": "personal_balanced",
        "must_keep": ["35%"],
        "text": "I'm a 39-year-old bakery owner in Pittsburgh whose butter and flour costs jumped 35%—how should I communicate a menu price increase without losing regulars?",
    },
    {
        "id": "sb_02",
        "domain": "Small Business & E-Commerce",
        "privacy_mode": "personal_balanced",
        "must_keep": ["2.5%"],
        "text": "As a 28-year-old Shopify store founder in Austin with a 2.5% conversion rate, what checkout UX changes reduce cart abandonment fastest?",
    },
    {
        "id": "sb_03",
        "domain": "Small Business & E-Commerce",
        "privacy_mode": "personal_balanced",
        "must_keep": ["$15,000"],
        "text": "I'm a 44-year-old freelance graphic designer in San Francisco pitching a $15,000 rebrand retainer—how should I structure milestone kill fees?",
    },
    {
        "id": "sb_04",
        "domain": "Small Business & E-Commerce",
        "privacy_mode": "personal_balanced",
        "must_keep": ["20"],
        "text": "As a 51-year-old restaurant manager in Boston scheduling 20 part-time servers, what tip-pooling rules keep payroll compliant?",
    },
    {
        "id": "sb_05",
        "domain": "Small Business & E-Commerce",
        "privacy_mode": "personal_balanced",
        "must_keep": ["14"],
        "text": "I'm a 33-year-old Etsy ceramicist in Denver scaling holiday fulfillment—what 14-day packaging workflow prevents breakage in cold weather?",
    },
]


def _check_scars(text: str, rewritten: str) -> List[str]:
    scars: List[str] = []
    if "locally-shift" in rewritten.lower():
        scars.append("city_tail_corruption")
    if re.search(r"\b(?:a|an)\s+an\s+adult\b", rewritten, re.IGNORECASE):
        scars.append("article_duplication")
    if "<DATE_TIME>" in rewritten:
        scars.append("presidio_date_time_scar")
    if "<LOCATION>" in rewritten:
        scars.append("presidio_location_scar")
    if "Pell Grant" in text and "Pell Grant" not in rewritten:
        scars.append("pell_grant_lost")
    return scars


def main() -> None:
    assert len(PERSONAS) == 50, f"Expected 50 personas, got {len(PERSONAS)}"

    # Warm up models
    rewriter = api.get_qwen_rewriter()
    rm = api.get_risk_model()
    cuda_ok = torch.cuda.is_available()
    vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1) if cuda_ok else 0.0
    print(
        f"Loaded risk_model={rm is not None} | qwen_adapter={bool(rewriter and rewriter.is_adapter_loaded)} "
        f"| cuda={cuda_ok} | vram_mb={vram_mb}"
    )

    results: List[Dict[str, Any]] = []
    engine_counts: Counter[str] = Counter()
    domain_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    t0_all = time.perf_counter()
    for idx, item in enumerate(PERSONAS, 1):
        sid = f"bench50p-{item['id']}-{secrets.token_hex(4)}"
        sec = secrets.token_hex(16)
        t0 = time.perf_counter()
        res = api.analyze_text(
            api.AnalyzeRequest(
                text=item["text"],
                session_id=sid,
                user_id="bench50p",
                privacy_mode=item["privacy_mode"],
            ),
            x_session_secret=sec,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        comp = res.comparison or {}
        risk_before = float(comp.get("inferential_before", res.risk_summary.get("overall", 0.0)))
        risk_after = float(comp.get("inferential_after", 0.0))
        hard_after = int(comp.get("hard_pii_after", 0))
        util = float((res.utility_metrics or {}).get("utility_score", 0.0))
        cos = float((res.utility_metrics or {}).get("cosine_similarity", 0.0))
        scars = _check_scars(item["text"], res.rewritten_text)
        missing_must = [tok for tok in item.get("must_keep", []) if tok.lower() not in res.rewritten_text.lower()]

        row = {
            "id": item["id"],
            "domain": item["domain"],
            "privacy_mode": item["privacy_mode"],
            "original": item["text"],
            "rewritten": res.rewritten_text,
            "rewrite_engine": res.rewrite_engine,
            "rewrite_changed": res.rewrite_changed,
            "risk_before": round(risk_before, 4),
            "risk_after": round(risk_after, 4),
            "risk_delta": round(risk_after - risk_before, 4),
            "hard_pii_after": hard_after,
            "utility_score": round(util, 4),
            "cosine_similarity": round(cos, 4),
            "pseudonyms": res.pseudonyms,
            "scars": scars,
            "missing_must_keep": missing_must,
            "latency_ms": elapsed_ms,
        }
        results.append(row)
        engine_counts[res.rewrite_engine] += 1
        domain_buckets[item["domain"]].append(row)
        print(
            f"[{idx:02d}/50] {item['id']} ({res.rewrite_engine:16s}) "
            f"risk {risk_before:.2f}->{risk_after:.2f} cos={cos:.2f} "
            f"hard={hard_after} scars={len(scars)} miss={len(missing_must)} ({elapsed_ms:.0f}ms)"
        )

    total_sec = round(time.perf_counter() - t0_all, 1)
    n = len(results)
    avg_before = sum(r["risk_before"] for r in results) / n
    avg_after = sum(r["risk_after"] for r in results) / n
    rel_red = ((avg_before - avg_after) / avg_before * 100.0) if avg_before > 0 else 0.0
    avg_cos = sum(r["cosine_similarity"] for r in results) / n
    avg_util = sum(r["utility_score"] for r in results) / n
    zero_pii_rate = sum(1 for r in results if r["hard_pii_after"] == 0) / n * 100.0
    zero_scar_rate = sum(1 for r in results if not r["scars"]) / n * 100.0
    must_keep_rate = sum(1 for r in results if not r["missing_must_keep"]) / n * 100.0
    avg_lat = sum(r["latency_ms"] for r in results) / n

    summary = {
        "total_personas": n,
        "domains": len(domain_buckets),
        "cuda_available": cuda_ok,
        "vram_allocated_mb": vram_mb,
        "total_wall_seconds": total_sec,
        "avg_latency_ms": round(avg_lat, 1),
        "engine_counts": dict(engine_counts),
        "avg_risk_before": round(avg_before, 4),
        "avg_risk_after": round(avg_after, 4),
        "relative_risk_reduction_pct": round(rel_red, 1),
        "avg_cosine_similarity": round(avg_cos, 4),
        "avg_utility_score": round(avg_util, 4),
        "zero_hard_pii_pct": round(zero_pii_rate, 1),
        "zero_scar_pct": round(zero_scar_rate, 1),
        "must_keep_preservation_pct": round(must_keep_rate, 1),
    }

    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "persona_50_domain_results.json"
    json_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")

    # Build Markdown report
    md_lines = [
        "# InferenceGuard — 50-Persona 10-Domain GPU Benchmark Report",
        "",
        "## Executive Summary",
        f"- **Total Personas Evaluated:** {n} across {len(domain_buckets)} domains (5 personas/domain)",
        f"- **Hardware:** CUDA={cuda_ok} | Peak VRAM={vram_mb} MB (`UtilityEvaluator` on CPU, `ModernBERT` + `Qwen2.5-1.5B` 4-bit NF4 LoRA on `cuda:0`)",
        f"- **Rewrite Engine Breakdown:** " + ", ".join(f"`{k}`: {v}/{n} ({v/n*100:.0f}%)" for k, v in engine_counts.most_common()),
        f"- **Mean Inferential Risk:** `{avg_before:.4f}` → **`{avg_after:.4f}`** (**`-{rel_red:.1f}%`** relative reduction)",
        f"- **Zero Hard-PII Leakage Rate:** **`{zero_pii_rate:.1f}%`** ({sum(1 for r in results if r['hard_pii_after'] == 0)}/{n})",
        f"- **Zero Regex/NER Scar Rate:** **`{zero_scar_rate:.1f}%`** ({sum(1 for r in results if not r['scars'])}/{n})",
        f"- **Critical Number / Order ID / Statute Preservation:** **`{must_keep_rate:.1f}%`** ({sum(1 for r in results if not r['missing_must_keep'])}/{n})",
        f"- **Mean Cosine Similarity / Utility:** `{avg_cos:.4f}` / `{avg_util:.4f}`",
        f"- **Mean Latency:** `{avg_lat:.0f} ms` (total wall time `{total_sec:.1f} s`)",
        "",
        "## Per-Domain Results",
        "",
        "| Domain | Mode | Qwen GPU | Risk Before | Risk After | Reduction | Cosine | Zero PII | IDs Kept |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dom, rows in domain_buckets.items():
        m_list = sorted({r["privacy_mode"] for r in rows})
        q_cnt = sum(1 for r in rows if r["rewrite_engine"] == "qwen_adapter")
        rb = sum(r["risk_before"] for r in rows) / len(rows)
        ra = sum(r["risk_after"] for r in rows) / len(rows)
        rr = ((rb - ra) / rb * 100.0) if rb > 0 else 0.0
        cs = sum(r["cosine_similarity"] for r in rows) / len(rows)
        zp = sum(1 for r in rows if r["hard_pii_after"] == 0)
        mk = sum(1 for r in rows if not r["missing_must_keep"])
        md_lines.append(
            f"| {dom} | `{', '.join(m_list)}` | {q_cnt}/{len(rows)} | {rb:.3f} | {ra:.3f} | -{rr:.1f}% | {cs:.3f} | {zp}/{len(rows)} | {mk}/{len(rows)} |"
        )

    md_lines.extend([
        "",
        "## Highlighted Before / After Rewrites",
        "",
    ])
    for sample_id in ("cs_01", "cs_02", "hc_01", "sw_01", "lg_01", "fn_01", "ac_01", "hr_01"):
        r = next(x for x in results if x["id"] == sample_id)
        md_lines.extend([
            f"### `{r['id']}` — {r['domain']} (`{r['rewrite_engine']}`, `{r['risk_before']:.2f}` → `{r['risk_after']:.2f}`)",
            f"- **Original:** {r['original']}",
            f"- **Protected:** {r['rewritten']}",
            f"- **Aliases:** `{r['pseudonyms']}`",
            "",
        ])

    md_path = reports_dir / "persona_50_domain_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSaved {json_path} and {md_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
