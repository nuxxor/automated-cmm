# Project Knowledge Layout

Use this folder for local project docs that should be retrieved at inference time. Do not fine-tune changing project facts into the model.

Recommended layout:

```text
data/knowledge/projects/
  project_slug/
    project.json
    faq.md
    support.md
    security.md
    governance.md
    incidents.md
```

`project.json` is metadata and is not indexed as a source document. The RAG indexer reads it for display name, aliases, source notes, and canonical URLs.

Supported source files:

- `.md`
- `.txt`
- `.json`

Source quality rules:

- Prefer official docs, FAQ, announcements, governance pages, status pages, and policy pages.
- Keep high-risk support policy in explicit files such as `security.md` or `support.md`.
- Do not include private community chat exports or personal data in project knowledge files.
- Add dates or source notes when content is time-sensitive.
- If a fact changes often, keep it in RAG, not in fine-tune examples.

The current sample projects under `sample_projects/` are only smoke-test fixtures. Replace or copy that structure for real projects.
