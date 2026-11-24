# Comparison matrix (split=test, n=50)

- attacker_model_id: `microsoft/Phi-4-mini-instruct`
- attacker_mode: `model`
- reportable_as_held_out: `True`
- risk_model_id: `AT707/inferenceguard-modernbert-risk`
- rewriter_model_id: `AT707/inferenceguard-rewriter-v1`

| Row | Attacker Acc | ASR reduction vs Raw | Abstain % | Bi-NLI | p95 Lat (ms) | Reportable |
| --- | --- | --- | --- | --- | --- | --- |
| Raw | 0.3650 | - | 0.5050 | 1.0000 | 0.00 | True |
| Presidio | 0.3600 | 0.0050 | 0.4900 | 0.7026 | 8.46 | True |
| Prompt-only | 0.1850 | 0.1800 | 0.5650 | 0.1087 | 3499.43 | True |
| Fine-tuned | 0.1500 | 0.2150 | 0.4050 | 0.0130 | 2232.27 | True |

> Held-out run: attacker_mode='model' / reportable_as_held_out=true. Numbers are the real held-out Phi-4-mini adversary results.
