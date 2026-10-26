"""Generate high-quality SFT training and validation datasets for LLM User Queries.

Focuses on the core objective of InferenceGuard:
When a user queries an LLM to solve a problem and includes inferential demographic PII
(Age, Location, Occupation, Education) in their prompt, InferenceGuard neutralizes the
demographic cues while preserving 100% of the problem-solving utility and domain precision.

Dataset Composition:
1. Preamble Pruning (~55%): Prunes autobiographical framing clauses that are unneeded for the task.
2. Natural Abstraction (~25%): Smoothly generalizes entangled parameters (tenure, life stage, broad jurisdiction) without awkward placeholders.
3. Negative Controls / Pass-Through (~20%): Neutral technical/coding queries with zero PII, teaching the model invariance.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.rewriter.inference import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Seed queries representing realistic real-world prompts
CORE_SEED_QUERIES: List[Dict[str, Any]] = [
    # Medical & Healthcare
    {
        "category": "natural_abstraction",
        "domain": "software_engineering",
        "query": "I am 26 years old living in Denver, CO and working as a software engineer after graduating from Carnegie Mellon. Can you help me optimize an SQL query for user analytics?",
        "target": "Can you help me optimize an SQL query for user analytics?",
    },
    {
        "category": "natural_abstraction",
        "domain": "software_engineering",
        "query": "I am 26 years old living in Denver, CO and working as a software engineer after graduating from Carnegie Mellon.",
        "target": "I work as a software engineer with a computer science background.",
    },
    {
        "category": "preamble_pruning",
        "domain": "medical",
        "query": "I'm a 3rd-year pediatric resident at Boston Children's Hospital working 80-hour weeks. Can you help me differential-diagnose a 4-year-old presenting with high fever for 6 days, bilateral conjunctival injection, strawberry tongue, and cervical lymphadenopathy?",
        "target": "Can you help me differential-diagnose a 4-year-old presenting with high fever for 6 days, bilateral conjunctival injection, strawberry tongue, and cervical lymphadenopathy for clinical rounds?",
    },
    {
        "category": "natural_abstraction",
        "domain": "healthcare_workflow",
        "query": "I'm a charge nurse on the night shift in a level 1 trauma center ED in Chicago. We have a sudden surge of 15 incoming acute triage cases and only 2 open ICU beds. What is the optimal triage escalation protocol to manage patient throughput safely?",
        "target": "In an emergency department experiencing an acute surge of high-acuity triage cases with severe inpatient ICU bed constraints, what are the best protocols to prioritize patient throughput and safety?",
    },
    {
        "category": "preamble_pruning",
        "domain": "pharmacy",
        "query": "I'm a clinical pharmacist at Johns Hopkins Hospital reviewing inpatient orders. Can you check for potential cytochrome P450 3A4 drug interactions between voriconazole and oral tacrolimus in a renal transplant patient?",
        "target": "Can you check for potential cytochrome P450 3A4 drug interactions between voriconazole and oral tacrolimus in a renal transplant patient?",
    },
    {
        "category": "preamble_pruning",
        "domain": "pathology",
        "query": "I am a pathology fellow at Cleveland Clinic analyzing bone marrow aspirate smears. What are the key immunohistochemical markers to differentiate mantle cell lymphoma from chronic lymphocytic leukemia?",
        "target": "What are the key immunohistochemical markers to differentiate mantle cell lymphoma from chronic lymphocytic leukemia on bone marrow aspirate smears?",
    },
    {
        "category": "preamble_pruning",
        "domain": "oncology",
        "query": "I'm a medical oncology registrar at Memorial Sloan Kettering in NYC. What are the current second-line systemic therapy recommendations for metastatic triple-negative breast cancer with PD-L1 negative status following progression on anthracycline-taxane regimens?",
        "target": "What are current second-line systemic therapy recommendations for metastatic triple-negative breast cancer with PD-L1 negative status following progression on anthracycline-taxane regimens?",
    },

    # Legal & Regulatory
    {
        "category": "natural_abstraction",
        "domain": "employment_law",
        "query": "I work as a bar manager in an Austin live music venue on 6th Street. Our owner is retaining 25% of the collective credit card tip pool to cover glassware breakage. Is this legal under Texas labor code and the federal FLSA tip credit regulations?",
        "target": "Is it legal under the federal FLSA tip credit regulations and state labor laws for an employer to retain a portion of an employee credit card tip pool to offset glassware breakage or operational losses?",
    },
    {
        "category": "natural_abstraction",
        "domain": "housing_law",
        "query": "I rent an apartment in Islington, London, paying £2,200 pcm. My landlord just issued a Section 21 notice with only 4 weeks instead of the statutory notice period. What steps should I take with the local council housing ombudsman?",
        "target": "If a private landlord serves a Section 21 eviction notice that provides less than the statutory notice period, what immediate legal steps and complaints procedures should a tenant file?",
    },
    {
        "category": "preamble_pruning",
        "domain": "corporate_law",
        "query": "I'm a 2nd-year associate at a corporate boutique in Wilmington, Delaware. Can you draft a standard indemnification clause for a Delaware LLC operating agreement protecting managing members against non-willful fiduciary breaches?",
        "target": "Can you draft a standard indemnification clause for a Delaware LLC operating agreement protecting managing members against non-willful fiduciary breaches?",
    },
    {
        "category": "preamble_pruning",
        "domain": "intellectual_property",
        "query": "I'm an in-house patent attorney at a semiconductor company in San Jose, California. How does 35 U.S.C. 101 subject matter eligibility apply to algorithmic computer-implemented inventions under the USPTO Alice/Mayo step 2A guidance?",
        "target": "How does 35 U.S.C. 101 subject matter eligibility apply to algorithmic computer-implemented inventions under the USPTO Alice/Mayo step 2A guidance?",
    },
    {
        "category": "preamble_pruning",
        "domain": "data_privacy_law",
        "query": "I work as a compliance paralegal at a FinTech firm in Berlin. Under GDPR Article 49, what are the strict derogation conditions for transferring customer pseudonymized transactional logs to US-based AWS cloud regions without standard contractual clauses?",
        "target": "Under GDPR Article 49, what are the strict derogation conditions for transferring customer pseudonymized transactional logs to non-EEA cloud regions without standard contractual clauses?",
    },

    # Finance, Pensions & Retirement
    {
        "category": "natural_abstraction",
        "domain": "retirement_finance",
        "query": "I'm a 58-year-old high school physics teacher in upstate New York planning to retire in two years after 32 years of teaching. What are the best strategies to coordinate my NY state teacher pension with my 403b savings under current Windfall Elimination Provision rules?",
        "target": "What are the best strategies to coordinate a public educator state pension with supplemental 403(b) retirement accounts under current Windfall Elimination Provision rules?",
    },
    {
        "category": "natural_abstraction",
        "domain": "early_retirement",
        "query": "I am a 34-year-old senior software engineer in Seattle making $280k total compensation aiming for FIRE by age 42. How should I allocate between taxable brokerage, backdoor Roth IRA, and mega-backdoor Roth 401(k) to maximize early accessibility without 10% early withdrawal penalties?",
        "target": "How should an investor allocate savings between taxable brokerage accounts, backdoor Roth IRAs, and mega-backdoor Roth 401(k) contributions to optimize early retirement liquidity before age 59½ without incurring early withdrawal penalties?",
    },
    {
        "category": "natural_abstraction",
        "domain": "estate_planning",
        "query": "I am 67 years old, living in Denver, Colorado, and currently drawing PERA pension benefits. Can you explain how required minimum distributions (RMDs) from traditional IRAs interact with provisional income calculations for federal Social Security taxation?",
        "target": "How do required minimum distributions (RMDs) from traditional IRAs interact with provisional income calculations for federal Social Security benefit taxation?",
    },
    {
        "category": "preamble_pruning",
        "domain": "corporate_accounting",
        "query": "I work as a senior financial analyst at a consumer retail conglomerate in Atlanta. Under ASC 842, how do we calculate the right-of-use asset impairment when subleasing a warehouse facility at below-market commercial rates?",
        "target": "Under ASC 842, how is a right-of-use (ROU) asset impairment calculated when subleasing a commercial facility at below-market lease rates?",
    },

    # Engineering & Cloud Infrastructure
    {
        "category": "natural_abstraction",
        "domain": "career_transition",
        "query": "I started my IT career back in 1994 deploying Novell NetWare and Windows NT 3.51. After 30 years in enterprise system administration, how should I reformat my resume to highlight modern Kubernetes and Terraform cloud automation without signaling my age to automated applicant tracking systems?",
        "target": "How should an experienced enterprise infrastructure professional reformat a resume to emphasize modern Kubernetes, cloud, and Terraform automation skills while keeping career chronology concise for modern ATS systems?",
    },
    {
        "category": "preamble_pruning",
        "domain": "cloud_architecture",
        "query": "I'm a DevOps lead at an enterprise SaaS company in Toronto managing multi-region EKS clusters. How do I configure AWS Network Load Balancer with Cilium eBPF host routing to preserve client source IPs for strict SOC2 compliance auditing?",
        "target": "How do I configure AWS Network Load Balancer with Cilium eBPF host routing to preserve client source IPs for audit logging?",
    },
    {
        "category": "preamble_pruning",
        "domain": "embedded_hardware",
        "query": "I'm a firmware engineer at an automotive tier-1 supplier in Detroit working on AUTOSAR Classic. How do I implement a non-blocking CAN-FD message queue using FreeRTOS ring buffers with priority inversion prevention?",
        "target": "How do I implement a non-blocking CAN-FD message queue using FreeRTOS ring buffers with priority inversion prevention?",
    },
    {
        "category": "natural_abstraction",
        "domain": "urban_transit",
        "query": "I'm starting a new software role and need to bike from Somerville across the Longfellow Bridge into downtown Boston each morning. What are the safest street routes with continuous protected bike lanes that avoid major MBTA construction zones?",
        "target": "What are key criteria and navigation tools for planning an urban bicycle commute with continuous protected bike lanes that reliably avoids active municipal transit construction zones?",
    },

    # Higher Education & Academic Defense
    {
        "category": "natural_abstraction",
        "domain": "academic_defense",
        "query": "My doctoral committee at Purdue is challenging my dissertation defense on turbulent combustion simulations using OpenFOAM. How should I defend my choice of Large Eddy Simulation over Reynolds-Averaged Navier-Stokes for high Reynolds number boundary layers?",
        "target": "How should I technically defend the choice of Large Eddy Simulation (LES) over Reynolds-Averaged Navier-Stokes (RANS) for high Reynolds number boundary layers in turbulent combustion modeling?",
    },
    {
        "category": "natural_abstraction",
        "domain": "undergrad_interview",
        "query": "I'm a sophomore studying electrical engineering at Georgia Tech. I have my first technical interview tomorrow with an analog IC design firm. Can you quiz me on small-signal analysis of common-source amplifiers and Miller capacitance?",
        "target": "Can you quiz me on small-signal analysis of common-source amplifiers and the Miller effect in analog integrated circuit design for a technical interview?",
    },
    {
        "category": "natural_abstraction",
        "domain": "graduate_admissions",
        "query": "I graduated with my bachelor's from Stanford in 2021 with a 3.4 GPA in symbolic systems. I'm now applying to top NLP PhD programs. Will my moderate undergrad GPA hurt my chances if I have two first-author ACL workshop papers?",
        "target": "When applying to competitive computer science PhD programs, how do admissions committees weigh a moderate undergraduate GPA against multiple first-author peer-reviewed workshop publications?",
    },
    {
        "category": "preamble_pruning",
        "domain": "higher_ed",
        "query": "I'm a first-year biology PhD student at UC Berkeley preparing for my qualifying exam. Can you synthesize the current scientific consensus on CRISPR-Cas12a versus Cas9 off-target cleavage specificity in mammalian genome editing?",
        "target": "Can you synthesize the current scientific consensus on CRISPR-Cas12a versus Cas9 off-target cleavage specificity in mammalian genome editing?",
    },

    # Pass-Through / Negative Controls (Neutral queries containing ZERO PII)
    {
        "category": "pass_through",
        "domain": "algorithms",
        "query": "What is the average and worst-case time complexity of QuickSort, and under what conditions does worst-case performance degrade to O(n^2)?",
        "target": "What is the average and worst-case time complexity of QuickSort, and under what conditions does worst-case performance degrade to O(n^2)?",
    },
    {
        "category": "pass_through",
        "domain": "networking",
        "query": "Explain the architectural differences between HTTP/2 multiplexing and HTTP/3 QUIC connection migration over UDP.",
        "target": "Explain the architectural differences between HTTP/2 multiplexing and HTTP/3 QUIC connection migration over UDP.",
    },
    {
        "category": "pass_through",
        "domain": "database_internals",
        "query": "How does PostgreSQL implement Multi-Version Concurrency Control (MVCC), and what causes table bloat during high-frequency UPDATE operations?",
        "target": "How does PostgreSQL implement Multi-Version Concurrency Control (MVCC), and what causes table bloat during high-frequency UPDATE operations?",
    },
    {
        "category": "pass_through",
        "domain": "machine_learning",
        "query": "What is the mathematical formulation of FlashAttention-2, and how does tiling the Q, K, V blocks minimize high-bandwidth GPU HBM memory accesses?",
        "target": "What is the mathematical formulation of FlashAttention-2, and how does tiling the Q, K, V blocks minimize high-bandwidth GPU HBM memory accesses?",
    },
    {
        "category": "pass_through",
        "domain": "distributed_systems",
        "query": "How does the Raft consensus algorithm handle split-vote scenarios during candidate leader election timeouts?",
        "target": "How does the Raft consensus algorithm handle split-vote scenarios during candidate leader election timeouts?",
    },
    {
        "category": "pass_through",
        "domain": "cryptography",
        "query": "Compare the security assumptions and ciphertext sizes of Kyber (ML-KEM) lattice-based key encapsulation with classical RSA-4096.",
        "target": "Compare the security assumptions and ciphertext sizes of Kyber (ML-KEM) lattice-based key encapsulation with classical RSA-4096.",
    },
    {
        "category": "pass_through",
        "domain": "operating_systems",
        "query": "What is the mechanism of page fault handling in the Linux virtual memory manager, from CR2 register capture to PTE allocation?",
        "target": "What is the mechanism of page fault handling in the Linux virtual memory manager, from CR2 register capture to PTE allocation?",
    },
]

# Systematic Combinatorial Variations across Professions, Locations, Ages, and Questions
AUTOBIOGRAPHICAL_PREFIXES = [
    ("I'm a 3rd-year resident at {hospital} in {city} working {hours}-hour weeks.", ""),
    ("I am a {age}-year-old {occupation} living in {city}.", ""),
    ("I work as a {occupation} at a {org_type} in {city}.", ""),
    ("I've been working as a {occupation} for {years} years in {city}.", ""),
    ("I'm currently employed as a {occupation} in downtown {city}.", ""),
    ("As a {age}-year-old {occupation} based out of {city},", ""),
    ("I graduated from {university} with a degree in {major} and now work in {city}.", ""),
    ("I rent an apartment in {neighborhood}, {city}, paying {rent} a month.", ""),
    ("I'm an incoming {occupation} relocating to {city} next month.", ""),
]

DOMAINS_AND_QUESTIONS = [
    # Medical questions (patient age must NOT be touched)
    (
        "medical",
        "Can you help me differential-diagnose a 7-year-old child presenting with acute abdominal pain, right lower quadrant tenderness, and leukocytosis?",
        "Can you help me differential-diagnose a 7-year-old child presenting with acute abdominal pain, right lower quadrant tenderness, and leukocytosis?",
    ),
    (
        "medical",
        "What is the recommended first-line antimicrobial regimen for an immunocompromised 65-year-old patient with community-acquired pneumonia?",
        "What is the recommended first-line antimicrobial regimen for an immunocompromised 65-year-old patient with community-acquired pneumonia?",
    ),
    (
        "medical",
        "Can you review the electrocardiogram criteria for differentiating ventricular tachycardia from supraventricular tachycardia with aberrancy in an unstable patient?",
        "Can you review the electrocardiogram criteria for differentiating ventricular tachycardia from supraventricular tachycardia with aberrancy in an unstable patient?",
    ),
    (
        "medical",
        "What are the clinical indications and contraindications for initiating dual antiplatelet therapy (DAPT) following bare-metal versus drug-eluting stent placement?",
        "What are the clinical indications and contraindications for initiating dual antiplatelet therapy (DAPT) following bare-metal versus drug-eluting stent placement?",
    ),
    # Legal questions
    (
        "legal",
        "Is an employer legally permitted to deduct cash register shortages or equipment breakage from non-exempt employee paychecks under the Fair Labor Standards Act?",
        "Is an employer legally permitted to deduct cash register shortages or equipment breakage from non-exempt employee paychecks under the Fair Labor Standards Act?",
    ),
    (
        "legal",
        "What are the statutory requirements and minimum notice periods for non-fault residential lease terminations under state landlord-tenant regulations?",
        "What are the statutory requirements and minimum notice periods for non-fault residential lease terminations under state landlord-tenant regulations?",
    ),
    (
        "legal",
        "Can you draft an enforceable non-compete clause that complies with current FTC regulations and state blue-pencil doctrine limitations?",
        "Can you draft an enforceable non-compete clause that complies with current FTC regulations and state blue-pencil doctrine limitations?",
    ),
    (
        "legal",
        "What constitutes actionable hostile work environment discrimination under Title VII of the Civil Rights Act of 1964?",
        "What constitutes actionable hostile work environment discrimination under Title VII of the Civil Rights Act of 1964?",
    ),
    # Finance questions
    (
        "finance",
        "What are the mathematical formulas to calculate the breakeven age when comparing taking Social Security benefits at age 62 versus waiting until age 70?",
        "What are the mathematical formulas to calculate the breakeven age when comparing taking Social Security benefits at age 62 versus waiting until age 70?",
    ),
    (
        "finance",
        "How do tax-loss harvesting rules apply to wash sales across taxable brokerage accounts and spouse IRAs under IRS Publication 550?",
        "How do tax-loss harvesting rules apply to wash sales across taxable brokerage accounts and spouse IRAs under IRS Publication 550?",
    ),
    (
        "finance",
        "What are the trade-offs between traditional defined-benefit pensions and 401(k) lump-sum rollovers regarding longevity risk and inflation protection?",
        "What are the trade-offs between traditional defined-benefit pensions and 401(k) lump-sum rollovers regarding longevity risk and inflation protection?",
    ),
    # Technical & Engineering questions
    (
        "engineering",
        "How do I tune Linux sysctl TCP keepalive and epoll buffer parameters to sustain 500,000 concurrent WebSocket connections on a 32-core server?",
        "How do I tune Linux sysctl TCP keepalive and epoll buffer parameters to sustain 500,000 concurrent WebSocket connections on a 32-core server?",
    ),
    (
        "engineering",
        "What is the mathematical formulation of spatial discretization errors in finite volume methods when switching from upwind differencing to second-order central differencing?",
        "What is the mathematical formulation of spatial discretization errors in finite volume methods when switching from upwind differencing to second-order central differencing?",
    ),
    (
        "engineering",
        "How do I configure Prometheus alerts and Grafana dashboards to monitor p99 latency spikes caused by garbage collection pauses in the JVM?",
        "How do I configure Prometheus alerts and Grafana dashboards to monitor p99 latency spikes caused by garbage collection pauses in the JVM?",
    ),
    (
        "engineering",
        "Can you write a robust retry decorator in Python with exponential backoff and jitter to wrap transient HTTP 503 errors?",
        "Can you write a robust retry decorator in Python with exponential backoff and jitter to wrap transient HTTP 503 errors?",
    ),
]

HOSPITALS = ["Massachusetts General Hospital", "Cleveland Clinic", "Stanford Health Care", "Mayo Clinic", "Mount Sinai Hospital"]
CITIES = ["Boston", "Seattle", "Austin", "New York", "Chicago", "Denver", "San Francisco", "Atlanta", "London", "Toronto"]
NEIGHBORHOODS = ["Islington", "Somerville", "Capitol Hill", "Back Bay", "Lincoln Park", "Brooklyn", "SoMa", "Midtown"]
OCCUPATIONS = [
    "pediatric resident", "software engineer", "high school teacher", "paralegal",
    "cloud architect", "civil engineer", "charge nurse", "tax accountant",
    "data scientist", "mechanical engineer", "barista", "pharmacist",
]
UNIVERSITIES = ["Harvard", "MIT", "Stanford", "Purdue", "Georgia Tech", "UC Berkeley", "Carnegie Mellon", "Columbia"]
MAJORS = ["electrical engineering", "computer science", "biochemistry", "applied physics", "economics", "mechanical engineering"]


def generate_pruning_pairs(count: int = 80) -> List[Dict[str, Any]]:
    """Synthesize diverse preamble-pruning user query training pairs."""
    pairs: List[Dict[str, Any]] = []
    for _ in range(count):
        prefix_tmpl, _ = random.choice(AUTOBIOGRAPHICAL_PREFIXES)
        domain, question, target = random.choice(DOMAINS_AND_QUESTIONS)

        prefix = prefix_tmpl.format(
            hospital=random.choice(HOSPITALS),
            city=random.choice(CITIES),
            neighborhood=random.choice(NEIGHBORHOODS),
            hours=random.choice([60, 70, 80]),
            age=random.choice([24, 28, 35, 42, 51, 58, 63]),
            occupation=random.choice(OCCUPATIONS),
            org_type=random.choice(["startup", "hospital", "law firm", "consultancy", "school district", "retailer"]),
            years=random.choice([3, 5, 10, 15, 25, 30]),
            university=random.choice(UNIVERSITIES),
            major=random.choice(MAJORS),
            rent=f"${random.choice([1800, 2200, 2800, 3400])}",
        )

        query = f"{prefix} {question}"
        pairs.append({
            "category": "preamble_pruning",
            "domain": domain,
            "query": query,
            "target": target,
        })
    return pairs


def generate_natural_abstraction_pairs(count: int = 35) -> List[Dict[str, Any]]:
    """Synthesize pairs where parameters are entangled and need natural abstraction."""
    pairs: List[Dict[str, Any]] = []
    
    templates = [
        (
            "I'm a {age}-year-old {occupation} in {city} planning to retire in {years_left} years after {years_worked} years of service. What are the best options to coordinate my defined benefit pension with my {retirement_account} savings?",
            "What are the best strategies to coordinate a defined-benefit pension with supplemental {retirement_account} retirement savings when planning for retirement after an extended career?",
        ),
        (
            "I started in tech in {start_year} maintaining {legacy_sys} in {city}. After {decades} years in the industry, how do I position my resume for modern {modern_tech} roles without dating myself?",
            "How should an experienced technology professional reformat a resume to highlight modern {modern_tech} expertise while keeping career chronology focused and modern for applicant tracking systems?",
        ),
        (
            "I'm an incoming {occupation} commuting from {neighborhood} to downtown {city}. What are the most reliable multimodal transit routes to avoid peak rush-hour congestion?",
            "What are the most reliable route planning strategies and tools for an urban commuter seeking to avoid peak rush-hour transit congestion?",
        ),
        (
            "I graduated from {university} in {grad_year} with an undergraduate degree in {major} ({gpa} GPA). Will my undergraduate transcript weigh heavily when applying for competitive {grad_prog} PhD programs?",
            "When applying for competitive graduate PhD programs, how significantly do admissions committees weigh undergraduate GPA versus research experience and publications?",
        ),
        (
            "I work as a {occupation} at a {org_type} in {city}. Our management is withholding {deduct_pct}% of employee bonus pools for company property loss. Is this enforceable under state labor statutes?",
            "Is it enforceable under federal and state employment labor statutes for an employer to withhold a percentage of employee incentive or tip pools to cover operational losses?",
        ),
    ]

    for _ in range(count):
        tmpl_q, tmpl_t = random.choice(templates)
        age = random.choice([52, 56, 59, 61, 64])
        years_left = random.choice([1, 2, 3])
        years_worked = random.choice([25, 28, 32, 35])
        ret_acc = random.choice(["401(k)", "403(b)", "traditional IRA"])
        start_year = random.choice([1992, 1996, 1998, 2001])
        decades = random.choice(["twenty-five", "thirty", "twenty"])
        legacy_sys = random.choice(["Solaris Unix and Perl", "AS/400 and COBOL", "Windows NT and Novell NetWare", "Clipper and FoxPro"])
        modern_tech = random.choice(["Kubernetes and Terraform", "Go and distributed microservices", "Rust and cloud-native systems"])
        grad_year = random.choice([2019, 2020, 2022])
        gpa = random.choice(["3.3", "3.4", "3.5"])
        grad_prog = random.choice(["Computer Science", "Computational Biology", "Artificial Intelligence"])
        deduct_pct = random.choice([15, 20, 25])

        q = tmpl_q.format(
            age=age,
            occupation=random.choice(OCCUPATIONS),
            city=random.choice(CITIES),
            years_left=years_left,
            years_worked=years_worked,
            retirement_account=ret_acc,
            start_year=start_year,
            legacy_sys=legacy_sys,
            decades=decades,
            modern_tech=modern_tech,
            neighborhood=random.choice(NEIGHBORHOODS),
            university=random.choice(UNIVERSITIES),
            grad_year=grad_year,
            major=random.choice(MAJORS),
            gpa=gpa,
            grad_prog=grad_prog,
            org_type=random.choice(["startup", "retailer", "venue", "agency"]),
            deduct_pct=deduct_pct,
        )

        t = tmpl_t.format(
            retirement_account=ret_acc,
            modern_tech=modern_tech,
        )

        pairs.append({
            "category": "natural_abstraction",
            "domain": "general_advisory",
            "query": q,
            "target": t,
        })
    return pairs


def generate_pass_through_pairs(count: int = 30) -> List[Dict[str, Any]]:
    """Synthesize neutral technical/coding queries where query == target."""
    neutral_queries = [
        "What is the time complexity of the Floyd-Warshall all-pairs shortest path algorithm?",
        "How do you implement an LRU cache in Python using an OrderedDict or doubly linked list with a hash map?",
        "Explain how TLS 1.3 eliminates the round-trip handshake latency compared to TLS 1.2.",
        "What is the mathematical proof that the Halting Problem is undecidable via diagonalization?",
        "How does a B-Tree differ from a B+ Tree in database index storage engines?",
        "Explain the CAP theorem and the trade-offs between consistency and availability during network partitions.",
        "How does Python's asyncio event loop schedule coroutines, tasks, and future callbacks?",
        "What are the best practices for structuring Docker multi-stage builds to minimize production image footprint?",
        "Write an SQL query using window functions to find the top 3 highest-spending customers per geographic region.",
        "How do you configure CORS in an Express or FastAPI application for preflight OPTIONS requests?",
        "Explain the differences between symmetric AES-GCM and asymmetric RSA encryption in hybrid cryptosystems.",
        "What is the purpose of the virtual DOM in modern frontend web frameworks like React?",
        "How does garbage collection in the V8 JavaScript engine handle generational memory management?",
        "Explain how gradient descent with Adam optimizer computes adaptive learning rates using first and second moments.",
        "What is the difference between a process and a thread at the operating system kernel level?",
    ]
    pairs = []
    for _ in range(count):
        q = random.choice(neutral_queries)
        pairs.append({
            "category": "pass_through",
            "domain": "technical_neutral",
            "query": q,
            "target": q,
        })
    return pairs


def build_full_query_dataset(seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate train and validation datasets of user LLM queries with ChatML formatting."""
    random.seed(seed)

    all_pairs: List[Dict[str, Any]] = []
    # 1. Add core seed queries
    all_pairs.extend(CORE_SEED_QUERIES)

    # 2. Add synthetic varied queries
    all_pairs.extend(generate_pruning_pairs(count=85))
    all_pairs.extend(generate_natural_abstraction_pairs(count=35))
    all_pairs.extend(generate_pass_through_pairs(count=30))

    random.shuffle(all_pairs)

    # Train / Val split (85% train, 15% val)
    split_idx = int(len(all_pairs) * 0.85)
    train_raw = all_pairs[:split_idx]
    val_raw = all_pairs[split_idx:]

    def format_records(raw_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for r in raw_list:
            chatml_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{r['query']}<|im_end|>\n"
                f"<|im_start|>assistant\n{r['target']}<|im_end|>\n"
            )
            formatted.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": r["query"]},
                    {"role": "assistant", "content": r["target"]},
                ],
                "text": chatml_text,
                "category": r["category"],
                "domain": r["domain"],
            })
        return formatted

    return format_records(train_raw), format_records(val_raw)


def main():
    parser = argparse.ArgumentParser(description="Generate LLM User Query SFT datasets.")
    parser.add_argument("--output_train", type=str, default="data/sft/train_queries_sft.jsonl", help="Train jsonl path")
    parser.add_argument("--output_val", type=str, default="data/sft/val_queries_sft.jsonl", help="Val jsonl path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_data, val_data = build_full_query_dataset(seed=args.seed)

    out_train_path = Path(args.output_train)
    out_val_path = Path(args.output_val)
    out_train_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(out_val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_data)} train examples -> {out_train_path}")
    print(f"Generated {len(val_data)} validation examples -> {out_val_path}")


if __name__ == "__main__":
    main()
