import json
from pathlib import Path

BENCHMARK_ITEMS = [
    # =========================================================================
    # Group 1: Co-Occurring PII in Preamble (Multiple PII at Start)
    # =========================================================================
    {
        "id": "tc_01",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "education", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "I am a 34-year-old senior oncologist at Johns Hopkins in Baltimore earning $320k with a PhD from Stanford. What is the standard adjuvant protocol for HER2-positive stage II breast cancer?",
        "expected_behavior": "Prune all autobiographical framing (age, hospital, city, income, PhD). Start directly with adjuvant breast cancer protocol query.",
    },
    {
        "id": "tc_02",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "education", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "I'm a 22yo recent UC Berkeley computer science graduate working as a junior backend engineer in San Francisco making $140k. How do I implement consistent hashing in Go?",
        "expected_behavior": "Prune leading graduate and engineering persona. Retain only technical Go implementation query.",
    },
    {
        "id": "tc_03",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "education", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "As a 55-year-old tenured economics professor at Northwestern living in Evanston with a $190k salary, can you summarize the econometric evidence on minimum wage elasticities?",
        "expected_behavior": "Prune professor persona and university. Retain econometric query on minimum wage elasticities.",
    },
    {
        "id": "tc_04",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "education", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "I am a 29-year-old structural engineer in Miami earning $110k with a master's from Georgia Tech. What are the wind load design parameters for high-rise shear walls under ASCE 7-22?",
        "expected_behavior": "Prune engineer demographics. Retain technical ASCE 7-22 building standard query.",
    },
    {
        "id": "tc_05",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "I am a 42-year-old registered nurse practitioner in Denver working 60h weeks earning $115k. What are the contraindications for initiating dual antiplatelet therapy in an outpatient setting?",
        "expected_behavior": "Prune nurse practitioner disclosure. Retain clinical outpatient pharmacology question.",
    },
    {
        "id": "tc_06",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "As a 61-year-old chief financial officer living in Seattle earning $450k and planning retirement next year, what is the tax efficiency difference between municipal bonds and dividend index funds?",
        "expected_behavior": "Prune personal executive demographics. Retain core financial asset allocation query.",
    },
    {
        "id": "tc_07",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "education"],
        "jurisdiction_or_entity_required": False,
        "input": "I am a 26yo PhD student in robotics at Carnegie Mellon living in Pittsburgh. What is the mathematical formulation of the Iterative Closest Point algorithm with point-to-plane metric?",
        "expected_behavior": "Prune CMU robotics PhD persona. Retain mathematical ICP algorithm query.",
    },
    {
        "id": "tc_08",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "I am a 38-year-old commercial airline pilot living in Dallas earning $240k. Can you explain the aerodynamic differences between swept-wing stall and straight-wing stall?",
        "expected_behavior": "Prune pilot demographics. Retain aerodynamic stall comparison query.",
    },
    {
        "id": "tc_09",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "education"],
        "jurisdiction_or_entity_required": False,
        "input": "As an 18-year-old first-year pre-law undergraduate at NYU in Manhattan, can you provide a breakdown of the commerce clause jurisprudential doctrine?",
        "expected_behavior": "Prune undergraduate NYU student framing. Retain commerce clause legal theory query.",
    },
    {
        "id": "tc_10",
        "category": "co_occurring_pii_preamble",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "I am a 45-year-old petroleum geologist in Houston making $210k. How does seismic inversion differentiate between carbonate and clastic reservoir porosity?",
        "expected_behavior": "Prune petroleum geologist persona. Retain technical geophysics question.",
    },

    # =========================================================================
    # Group 2: PII Introduced at the End / Postamble / Trailing Notes
    # =========================================================================
    {
        "id": "tc_11",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "What is the recommended antibiotic regimen for acute bacterial meningitis? FYI, I am a 29-year-old internal medicine resident in Chicago working nights.",
        "expected_behavior": "Retain clinical meningitis question. Strip trailing resident note.",
    },
    {
        "id": "tc_12",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "Can you generate a PyTorch script to train a vision transformer with FlashAttention-2? Note that I'm a 24yo machine learning engineer in San Jose making $160k.",
        "expected_behavior": "Retain technical PyTorch query. Strip trailing ML engineer disclosure.",
    },
    {
        "id": "tc_13",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "How do I calculate the thermal resistance of a multi-layer composite wall? By the way, I am a 31-year-old HVAC engineer in Boston.",
        "expected_behavior": "Retain thermal resistance calculation query. Strip HVAC engineer disclosure.",
    },
    {
        "id": "tc_14",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["education", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "What is the difference between a mutex and a binary semaphore in real-time operating systems? I am an undergraduate computer engineering student at Purdue.",
        "expected_behavior": "Retain RTOS synchronization query. Strip student background.",
    },
    {
        "id": "tc_15",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "Can you explain the mechanism of action of GLP-1 receptor agonists in metabolic dysfunction? Note that I am a 50yo endocrinologist in Philadelphia.",
        "expected_behavior": "Retain GLP-1 pharmacology query. Strip endocrinologist trailing note.",
    },
    {
        "id": "tc_16",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["location", "occupation", "income"],
        "jurisdiction_or_entity_required": False,
        "input": "How do I optimize database connection pooling in a FastAPI application under high concurrency? FYI I am a junior developer at a startup in Atlanta earning $85k.",
        "expected_behavior": "Retain FastAPI database query. Strip trailing junior developer salary details.",
    },
    {
        "id": "tc_17",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["location", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "What are the core requirements for ISO 27001 certification in cloud environments? By the way, I work as an IT security analyst in Detroit.",
        "expected_behavior": "Retain ISO 27001 compliance question. Strip security analyst postamble.",
    },
    {
        "id": "tc_18",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "education"],
        "jurisdiction_or_entity_required": False,
        "input": "Can you provide a step-by-step proof of Cauchy's Integral Formula in complex analysis? (I am a 20yo math major at Columbia).",
        "expected_behavior": "Retain complex analysis mathematical proof. Strip trailing parenthetical.",
    },
    {
        "id": "tc_19",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "How do I calibrate a PID controller for an inverted pendulum system? (33yo robotics engineer in Minneapolis).",
        "expected_behavior": "Retain PID control systems engineering query. Strip parenthetical personal details.",
    },
    {
        "id": "tc_20",
        "category": "co_occurring_pii_postamble",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": False,
        "input": "What are the common side effects of SSRIs during the first two weeks of treatment? P.S. I am a 27yo psychiatric resident living in San Diego.",
        "expected_behavior": "Retain SSRI pharmacology query. Strip resident P.S. signoff.",
    },

    # =========================================================================
    # Group 3: Essential Legal Jurisdictions & Local Statutes (Must Preserve Location)
    # =========================================================================
    {
        "id": "tc_21",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "Under California Labor Code Section 2802, are remote employees entitled to mandatory reimbursement for internet and cell phone expenses?",
        "expected_behavior": "Preserve California Labor Code Section 2802 completely intact. Location is an essential statutory parameter.",
    },
    {
        "id": "tc_22",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "What is the statutory deadline for filing a mechanics lien in Texas on commercial real estate projects?",
        "expected_behavior": "Preserve Texas commercial mechanics lien jurisdiction. Location is legally mandatory.",
    },
    {
        "id": "tc_23",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "Under New York State General Business Law Section 349, what constitutes deceptive trade practices?",
        "expected_behavior": "Preserve New York State statutory reference intact. Essential legal parameter.",
    },
    {
        "id": "tc_24",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "What are the mandatory notice periods for terminating a residential lease in Florida under Chapter 83?",
        "expected_behavior": "Preserve Florida Chapter 83 landlord-tenant statute completely intact.",
    },
    {
        "id": "tc_25",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "Under the UK Employment Rights Act 1996, what is the qualifying period for unfair dismissal claims?",
        "expected_behavior": "Preserve UK Employment Rights Act reference intact. Essential legal parameter.",
    },
    {
        "id": "tc_26",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "What are the greenhouse gas emission limits for commercial buildings under NYC Local Law 97?",
        "expected_behavior": "Preserve NYC Local Law 97 building regulation. Jurisdiction is mandatory.",
    },
    {
        "id": "tc_27",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "Under Massachusetts General Laws Chapter 93A, what are the requirements for sending a 30-day demand letter?",
        "expected_behavior": "Preserve Massachusetts Chapter 93A reference. Mandatory state law parameter.",
    },
    {
        "id": "tc_28",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "What is the statute of limitations for medical malpractice in Pennsylvania under Title 42 Section 5524?",
        "expected_behavior": "Preserve Pennsylvania statutory limitation intact. Mandatory legal reference.",
    },
    {
        "id": "tc_29",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "Under the European Union General Data Protection Regulation Article 17, what are the legal grounds for the right to erasure?",
        "expected_behavior": "Preserve European Union GDPR Article 17 reference intact.",
    },
    {
        "id": "tc_30",
        "category": "essential_jurisdiction_law",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "What are the seismic reinforcement requirements for unreinforced masonry buildings under the Los Angeles Municipal Code?",
        "expected_behavior": "Preserve Los Angeles Municipal Code reference intact.",
    },

    # =========================================================================
    # Group 4: Essential Clinical Presentations & Third-Party Case Studies
    # =========================================================================
    {
        "id": "tc_31",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "A 3-year-old female patient presents with fever, strawberry tongue, and bilateral conjunctival injection. What is the diagnostic criteria for Kawasaki disease?",
        "expected_behavior": "Preserve 3-year-old female patient presentation. The age and sex are clinical diagnostic parameters, not user PII.",
    },
    {
        "id": "tc_32",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "A 72-year-old male with a history of COPD presents with acute dyspnea and wheezing. What is the initial ventilatory management?",
        "expected_behavior": "Preserve 72-year-old COPD patient presentation. Third-party clinical entity.",
    },
    {
        "id": "tc_33",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "In an 8-month-old infant with bilious vomiting and currant jelly stool, what is the first-line imaging study for intussusception?",
        "expected_behavior": "Preserve 8-month-old infant clinical scenario. Essential diagnostic parameter.",
    },
    {
        "id": "tc_34",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "A 45-year-old woman presents with symmetrical joint pain in her hands and morning stiffness lasting over an hour. Which serological tests should be ordered for rheumatoid arthritis?",
        "expected_behavior": "Preserve 45-year-old female patient presentation. Third-party clinical scenario.",
    },
    {
        "id": "tc_35",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "A 16-year-old athlete collapsed during soccer practice with sudden cardiac arrest. What are the common etiologies to investigate?",
        "expected_behavior": "Preserve 16-year-old athlete clinical scenario intact.",
    },
    {
        "id": "tc_36",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "In a 6-year-old child with sickle cell disease presenting with acute chest syndrome, what is the transfusion threshold?",
        "expected_behavior": "Preserve 6-year-old pediatric sickle cell parameters.",
    },
    {
        "id": "tc_37",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "A 28-year-old primigravida at 34 weeks gestation presents with blood pressure of 165/110 and proteinuria. What is the management of severe preeclampsia?",
        "expected_behavior": "Preserve gestational clinical parameters intact.",
    },
    {
        "id": "tc_38",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "What are the pediatric dosing guidelines for amoxicillin in acute otitis media for a 15 kg child?",
        "expected_behavior": "Preserve 15 kg pediatric patient weight parameter intact.",
    },
    {
        "id": "tc_39",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "A 58-year-old male smoker presents with painless gross hematuria. What diagnostic workup is indicated for suspected bladder cancer?",
        "expected_behavior": "Preserve 58-year-old male smoker clinical case study.",
    },
    {
        "id": "tc_40",
        "category": "essential_clinical_guideline",
        "tracked_pii": ["none"],
        "jurisdiction_or_entity_required": True,
        "input": "An 11-year-old male presents with acute scrotal pain and absent cremasteric reflex. What is the time window for surgical detorsion in testicular torsion?",
        "expected_behavior": "Preserve 11-year-old clinical presentation intact.",
    },

    # =========================================================================
    # Group 5: Hybrid Edge Cases (Essential Legal/Medical Parameter + User Persona)
    # =========================================================================
    {
        "id": "tc_41",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": True,
        "input": "Under California Civil Code 1950.5, how many days does a landlord have to return a security deposit? FYI, I am a 25-year-old software engineer living in Mountain View renting a 1BR.",
        "expected_behavior": "Preserve California Civil Code 1950.5. Prune trailing engineer Mountain View disclosure.",
    },
    {
        "id": "tc_42",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": True,
        "input": "I am a 32-year-old general practitioner in Phoenix earning $220k. What are the Arizona Department of Health Services reporting guidelines for infectious syphilis?",
        "expected_behavior": "Preserve Arizona Department of Health Services. Prune leading practitioner persona.",
    },
    {
        "id": "tc_43",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["location", "occupation", "income"],
        "jurisdiction_or_entity_required": True,
        "input": "What is the threshold for filing a gift tax return in the United States for tax year 2024? (Living in London as an expat making $250k)",
        "expected_behavior": "Preserve United States tax year 2024. Prune trailing London expat parenthetical.",
    },
    {
        "id": "tc_44",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": True,
        "input": "As a 40-year-old cardiologist in Dallas, what are the Texas Medical Board requirements for CME credits in medical ethics?",
        "expected_behavior": "Preserve Texas Medical Board reference. Prune Dallas cardiologist persona.",
    },
    {
        "id": "tc_45",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["location", "occupation"],
        "jurisdiction_or_entity_required": True,
        "input": "A 5-year-old patient presented with acute appendicitis. What are the surgical guidelines for pediatric laparoscopic appendectomy? Note that I am a first-year surgical resident in Miami.",
        "expected_behavior": "Preserve 5-year-old patient presentation. Prune trailing Miami resident note.",
    },
    {
        "id": "tc_46",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["age", "location", "education"],
        "jurisdiction_or_entity_required": True,
        "input": "Under Washington State RCW 59.18, can a landlord enter without notice during an emergency? (22yo undergrad at UW Seattle)",
        "expected_behavior": "Preserve Washington State RCW 59.18 statute. Prune trailing undergrad parenthetical.",
    },
    {
        "id": "tc_47",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["age", "location", "occupation"],
        "jurisdiction_or_entity_required": True,
        "input": "I'm a 30yo lawyer in Chicago. Can you explain the jurisdictional differences between Delaware Chancery Court and New York Commercial Division in derivative lawsuits?",
        "expected_behavior": "Preserve Delaware Chancery and New York Commercial Division courts. Prune leading Chicago lawyer persona.",
    },
    {
        "id": "tc_48",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["location", "occupation", "income"],
        "jurisdiction_or_entity_required": True,
        "input": "What is the overtime calculation method under Illinois One Day Rest in Seven Act? By the way, I work as an HR director in Rockford making $130k.",
        "expected_behavior": "Preserve Illinois One Day Rest in Seven Act. Prune trailing HR director Rockford details.",
    },
    {
        "id": "tc_49",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["location", "occupation"],
        "jurisdiction_or_entity_required": True,
        "input": "In an elderly 84-year-old patient with chronic kidney disease stage 4, how should enoxaparin dosing be adjusted? FYI I am a clinical pharmacist in Salt Lake City.",
        "expected_behavior": "Preserve 84-year-old CKD patient presentation. Prune trailing Salt Lake City pharmacist note.",
    },
    {
        "id": "tc_50",
        "category": "hybrid_jurisdiction_and_user_pii",
        "tracked_pii": ["age", "location", "occupation", "income"],
        "jurisdiction_or_entity_required": True,
        "input": "What are the seismic structural bracing rules under the Tokyo Metropolitan Building Ordinance? (36-year-old civil engineer in Yokohama earning 12M yen)",
        "expected_behavior": "Preserve Tokyo Metropolitan Building Ordinance. Prune trailing Yokohama engineer parenthetical.",
    },
]

# Write JSON and JSONL
json_path = Path("data/evaluation/benchmark_test_50.json")
jsonl_path = Path("data/evaluation/benchmark_test_50.jsonl")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(BENCHMARK_ITEMS, f, indent=2)

with open(jsonl_path, "w", encoding="utf-8") as f:
    for item in BENCHMARK_ITEMS:
        f.write(json.dumps(item) + "\n")

print(f"Successfully generated {len(BENCHMARK_ITEMS)} benchmark items:")
print(f"  - {json_path}")
print(f"  - {jsonl_path}")
