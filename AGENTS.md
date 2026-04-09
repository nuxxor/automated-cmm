# Agent Notes

This repo is a local fine-tuning, serving, evaluation, and RAG scaffold for a crypto/community-manager assistant.

The public repository is intentionally generic. Do not add private source exports, raw conversation logs, personally identifying data, or unstable project facts to public commits.

## Critical Lessons

- Do not assume more SFT is the next move. Blind extra SFT or preference tuning can regress tone or damage strong categories.
- Always create a heldout eval before training again. Demo prompts are not enough.
- Separate raw model quality from runtime cleanup. A cleaned benchmark win is not automatically a model win. Report `raw_score`, `cleaned_score`, and `cleanup_delta`.
- Keep a broad OOD benchmark. A model can look good on a few style prompts while failing broad crypto support families such as bridge issues, third-party boundaries, rumor handling, and missing-context clarification.
- Promote only after rerunning the relevant benchmark. A style objective and a broad OOD support objective can conflict.
- Runtime/router/repair can be higher leverage than training. Benchmark-guided router and prompt-aware repair logic in `scripts/qwen3_ft_utils.py` produced the largest practical gains in this repo.
- Match repair rules against the latest user turn when possible. Matching the whole conversation can trigger false positives from old context.
- Preference tuning needs clean, diverse pairs. Do not run DPO/ORPO until preferred/rejected data clearly matches the target behavior.
- Fine-tune for behavior, not changing facts. Project-specific docs, policies, links, incidents, APR, bridge timing, and announcements belong in RAG.
- If a fact can change, do not bake it into the adapter.

## Current Public Architecture

- Base model family: Qwen3-14B.
- Expected local checkpoint: `runs/qwen3-14b-cm-v5c-humanness/checkpoint-6`.
- Deploy config: `configs/qwen3_14b_deploy_default.json`.
- Deploy prompt: `prompts/cm_deployment_prompt_v5.txt`.
- Default runtime: `runtime/cm_runtime_candidate_default.json`.
- Style-focused runtime: `runtime/cm_runtime_candidate_style_guardrails_v3.json`.
- Broad crypto OOD runtime: `runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json`.
- Broad OOD benchmark: `data/benchmarks/crypto_cm_ood_benchmark_v1.json`.
- RAG is scaffold/template only. See `docs/RAG_SYSTEM.md`. Real project docs, bot integrations, live logs, feedback capture, freshness checks, and production monitoring still need separate implementation.

## Evaluation Order For Future Iterations

1. Define the target: broad crypto CM behavior, human-style interaction quality, or project-specific factual support.
2. Pick or build the matching benchmark first.
3. Run the current baseline and save the report.
4. Make the smallest possible change.
5. Rerun the same benchmark plus at least one regression benchmark.
6. Promote only if the target improves without unacceptable regression elsewhere.

Useful broad OOD command:

```bash
python scripts/eval_cm_benchmark.py \
  --config configs/qwen3_14b_deploy_default.json \
  --system-prompt prompts/cm_deployment_prompt_v5.txt \
  --candidate-json runtime/cm_runtime_candidate_default.json \
  --secondary-candidate-json runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json \
  --benchmark-json data/benchmarks/crypto_cm_ood_benchmark_v1.json \
  --output reports/crypto_cm_ood_benchmark_next.json
```

Useful RAG smoke test:

```bash
python scripts/run_rag_pipeline.py \
  --config configs/rag_default.json \
  --build-index \
  --eval \
  --query "A helper wants remote access to troubleshoot my wallet" \
  --project-name aurora
```

## What To Avoid

- Do not optimize against a tiny handpicked prompt set and call it done.
- Do not treat production cleanup as proof the base adapter improved.
- Do not add private chat exports, personal data, or unstable project facts to RAG docs.
- Do not promote a training run just because loss went down.
- Do not add a new base model without first proving the current eval objective cannot be reached through data, router, or RAG changes.

