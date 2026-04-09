# RAG System

This repo keeps two concerns separate:

- The tuned model and runtime profiles handle community-manager tone, safety behavior, de-escalation, and public-support boundaries.
- RAG handles changing project facts such as docs, FAQ, policies, links, incidents, governance status, staking/rewards language, bridge behavior, and dashboard interpretation.

Do not fine-tune project facts into the model unless they are stable behavior patterns. Anything that can change should live in local knowledge files and be retrieved at inference time.

## Current Status

This RAG system is a scaffold/template layer. It is meant to show the structure, scripts, configs, benchmark shape, and model integration pattern an implementer should extend.

It is not a finished production bot integration yet. The following still need separate implementation before launch:

- real project docs, FAQ, announcements, policies, status pages, and canonical links
- chat-platform bot integration
- per-project routing from user/channel/context to the right knowledge index
- answer logging and audit trails
- live user feedback capture
- source freshness checks and scheduled re-indexing
- retrieval failure handling in production
- monitoring for hallucination, stale facts, bad source matches, and unsafe answers
- deployment-specific privacy and retention policy

## Components

- `configs/rag_default.json`: default RAG config for local smoke tests.
- `scripts/ingest_project_docs.py`: chunks project docs and writes a JSON index.
- `scripts/query_project_kb.py`: retrieves top chunks for a query.
- `scripts/eval_project_rag_benchmark.py`: scores retrieval quality against expected sources and keywords.
- `scripts/run_rag_pipeline.py`: config-driven wrapper for build, query, and eval.
- `scripts/chat_cm_model.py`: can inject retrieved context into model inference via `--rag-index-json`.
- `scripts/cm_rag_utils.py`: shared chunking, tokenization, scoring, alias matching, and context rendering.

## Knowledge Contract

Each project should be a folder:

```text
data/knowledge/projects/project_slug/
  project.json
  faq.md
  support.md
  security.md
  governance.md
  incidents.md
```

`project.json` is metadata and is skipped during indexing. It should include:

- `project`: stable lowercase slug
- `display_name`: human-readable name
- `aliases`: names users may type
- `canonical_urls`: official source links
- `source_notes`: freshness and trust notes
- `owner`: optional maintainer/team field

The rest of the files are indexed. Keep paragraphs short and factual.

## Build An Index

```bash
python scripts/ingest_project_docs.py \
  --input-dir data/knowledge/projects \
  --multi-project-root \
  --output-json data/knowledge_indexes/projects_v1.json
```

For one project only:

```bash
python scripts/ingest_project_docs.py \
  --input-dir data/knowledge/projects/project_slug \
  --project-name project_slug \
  --output-json data/knowledge_indexes/project_slug_v1.json
```

## Query

```bash
python scripts/query_project_kb.py \
  --index-json data/knowledge_indexes/projects_v1.json \
  --project-name project_slug \
  --query "Why do two dashboards show different reward numbers?" \
  --top-k 4 \
  --render-context
```

## Use With The CM Model

```bash
python scripts/chat_cm_model.py \
  --config configs/qwen3_14b_deploy_default.json \
  --system-prompt prompts/cm_deployment_prompt_v5.txt \
  --candidate-json runtime/cm_runtime_candidate_default.json \
  --secondary-candidate-json runtime/cm_runtime_candidate_broad_crypto_guardrails_v4.json \
  --rag-index-json data/knowledge_indexes/projects_v1.json \
  --rag-project-name project_slug \
  --message "Why do two dashboards show different reward numbers?" \
  --show-rag-context \
  --no-sample
```

The injected RAG system message tells the model to use retrieved facts and not guess when context is missing.

## Eval

Add benchmark cases to a JSON file:

```json
{
  "name": "project_rag_benchmark",
  "cases": [
    {
      "id": "example_001",
      "project": "project_slug",
      "query": "Where should users check outage updates?",
      "expected_sources": ["incidents.md"],
      "expected_keywords": ["status page", "incident update"]
    }
  ]
}
```

Run:

```bash
python scripts/eval_project_rag_benchmark.py \
  --index-json data/knowledge_indexes/projects_v1.json \
  --benchmark-json data/benchmarks/project_rag_benchmark_v1.json \
  --top-k 4 \
  --output-json reports/project_rag_benchmark_v1.json
```

Important metrics:

- `source_hit_rate`: retrieved at least one expected source
- `top1_source_hit_rate`: expected source was top result
- `pass_rate_at_8`: high-confidence retrieval pass

## Scaling To 100-200 Projects

Use one folder per project. Build a single multi-project index first; if it becomes too large or slow, shard by ecosystem or product type.

Suggested groups:

- `wallets`
- `bridges`
- `l2s`
- `defi`
- `staking`
- `governance`
- `infra`

Start with BM25-style lexical retrieval because it is dependency-free and easy to audit. Add embeddings later only if lexical retrieval fails on real user wording.

## What Not To Do

- Do not train changing links, APR numbers, bridge timing, or policy details into the adapter.
- Do not include private community chat logs in RAG docs.
- Do not let the model answer factual project questions without retrieved context unless the answer is a general safety principle.
- Do not optimize only for style-heldout if the deployment target is broad crypto support.
