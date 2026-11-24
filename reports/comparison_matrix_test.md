# Comparison matrix (split=test, n=50)

- attacker_model_id: `microsoft/Phi-4-mini-instruct`
- attacker_mode: `model`
- reportable_as_held_out: `True`
- risk_model_id: `AT707/inferenceguard-modernbert-risk`
- rewriter_model_id: `AT707/inferenceguard-rewriter-v1`

| Row | Attacker Acc | ASR reduction vs Raw | Abstain % | Bi-NLI | p95 Lat (ms) | Reportable |
| --- | --- | --- | --- | --- | --- | --- |
| Raw | 0.0050 | - | 0.9650 | 1.0000 | 0.00 | True |
| Presidio | 0.0050 | 0.0000 | 0.9650 | 1.0000 | 0.04 | True |
| Prompt-only | 0.0150 | -0.0100 | 0.8900 | 0.6103 | 2679.03 | True |
| Fine-tuned | 0.0000 | 0.0050 | 0.9450 | 0.5508 | 3481.31 | True |
| Conversation-aware | reportable | - | - | - | - | True |

> Held-out run: attacker_mode='model' / reportable_as_held_out=true. Numbers are the real held-out Phi-4-mini adversary results.
