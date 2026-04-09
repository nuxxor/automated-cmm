# Automated CMM

Local Qwen3-14B community-manager assistant toolkit.

This public repo contains the generic serving, evaluation, broad crypto OOD benchmark, and modular RAG scaffold. It does not include private source exports or raw training conversations. The model is intended to learn reusable community-manager behavior: warm, clear, scam-aware, calm under pressure, and firm when needed.

## Current Status

- Fine-tuned serving checkpoint is expected locally at `runs/qwen3-14b-cm-v5c-humanness/checkpoint-6`.
- The public repo keeps runtime, benchmark, and RAG infrastructure, not model weights.
- Project-specific facts should come from RAG, not from fine-tune data.
- The RAG layer is a scaffold/template. Real project docs, bot integration, answer logging, live feedback, source freshness checks, and production monitoring still need separate implementation.

## Key Files

- `configs/qwen3_14b_deploy_default.json`: default inference config.
- `prompts/cm_deployment_prompt_v5.txt`: deployment system prompt.
- `runtime/cm_runtime_candidate_default.json`: baseline runtime profile.
- `runtime/cm_runtime_candidate_style_guardrails_v3.json`: style-focused runtime profile.
- `runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json`: broad crypto OOD runtime profile.
- `scripts/chat_cm_model.py`: local chat runner with optional router and RAG context.
- `scripts/eval_cm_benchmark.py`: generic CM benchmark runner with raw/cleaned scoring.
- `scripts/build_crypto_cm_ood_benchmark.py`: builds the synthetic broad crypto OOD benchmark.
- `data/benchmarks/crypto_cm_ood_benchmark_v1.json`: 384-case synthetic crypto CM OOD benchmark.
- `scripts/cm_rag_utils.py`: chunking, BM25-style retrieval, alias matching, and context rendering.
- `scripts/ingest_project_docs.py`: build a project-doc retrieval index.
- `scripts/query_project_kb.py`: query a local retrieval index.
- `scripts/eval_project_rag_benchmark.py`: retrieval benchmark runner.
- `scripts/run_rag_pipeline.py`: config-driven RAG build/query/eval wrapper.
- `configs/rag_default.json`: default RAG smoke-test config.
- `docs/RAG_SYSTEM.md`: implementer-facing RAG notes.
- `templates/rag_project_template/`: starter docs layout for adding a new project.

## Local Chat

Single runtime:

```bash
python scripts/chat_cm_model.py \
  --config configs/qwen3_14b_deploy_default.json \
  --system-prompt prompts/cm_deployment_prompt_v5.txt \
  --candidate-json runtime/cm_runtime_candidate_default.json \
  --message "Someone claiming to be support sent me a DM with a verification link. Safe or scam?" \
  --no-sample
```

Broad crypto router:

```bash
python scripts/chat_cm_model.py \
  --config configs/qwen3_14b_deploy_default.json \
  --system-prompt prompts/cm_deployment_prompt_v5.txt \
  --candidate-json runtime/cm_runtime_candidate_default.json \
  --secondary-candidate-json runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json \
  --message "A helper wants remote access to troubleshoot my wallet. Safe or scam?" \
  --no-sample
```

## Broad Crypto OOD Eval

```bash
python scripts/eval_cm_benchmark.py \
  --config configs/qwen3_14b_deploy_default.json \
  --system-prompt prompts/cm_deployment_prompt_v5.txt \
  --candidate-json runtime/cm_runtime_candidate_default.json \
  --secondary-candidate-json runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json \
  --benchmark-json data/benchmarks/crypto_cm_ood_benchmark_v1.json \
  --output reports/crypto_cm_ood_benchmark_next.json
```

Use this benchmark to catch broad support failures such as third-party issues, missing context, rumor confirmation, seed phrase scams, remote-access scams, and de-escalation.

## RAG Scaffold

The RAG system is for changing project facts: docs, FAQ, policy, links, status, governance, incidents, staking/rewards language, bridge behavior, and dashboard interpretation.

Config-driven smoke test:

```bash
python scripts/run_rag_pipeline.py \
  --config configs/rag_default.json \
  --build-index \
  --eval \
  --query "A helper wants remote access to troubleshoot my wallet" \
  --project-name aurora
```

Use RAG during live chat:

```bash
python scripts/chat_cm_model.py \
  --config configs/qwen3_14b_deploy_default.json \
  --system-prompt prompts/cm_deployment_prompt_v5.txt \
  --candidate-json runtime/cm_runtime_candidate_default.json \
  --secondary-candidate-json runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json \
  --rag-index-json data/knowledge_indexes/sample_projects_v1.json \
  --rag-project-name delta_swap \
  --message "Why do two dashboards show different APR numbers?" \
  --show-rag-context \
  --no-sample
```

See `docs/RAG_SYSTEM.md` for the full project knowledge layout and implementation notes.

## Training Notes

The public repo intentionally does not include private source exports or raw conversation datasets. If you fine-tune a new adapter, use your own properly licensed/authorized data and keep private data out of public commits.

Important lessons are summarized in `AGENTS.md`.
