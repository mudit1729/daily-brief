---
name: AlphaXiv Paper Lookup
description: Look up any arxiv paper on alphaxiv.org to get a structured AI-generated overview. This is faster and more reliable than trying to read a raw PDF.
---

# AlphaXiv Paper Lookup

When a user provides an arXiv URL, paper ID, or asks you to explain a research paper, use this skill to fetch a structured AI-generated overview from alphaxiv.org.

## When to trigger

- User pastes an arXiv URL (e.g. `https://arxiv.org/abs/2301.00001`)
- User mentions a paper ID (e.g. `2301.00001` or `arxiv:2301.00001`)
- User asks you to summarize or explain an arXiv paper

## How to use

### Step 1 — Extract the paper ID

Strip the identifier from whatever format the user provides:

| Input | Extracted ID |
|---|---|
| `https://arxiv.org/abs/2301.00001` | `2301.00001` |
| `https://arxiv.org/pdf/2301.00001` | `2301.00001` |
| `arxiv:2301.00001` | `2301.00001` |
| `2301.00001` | `2301.00001` |
| `2301.00001v2` | `2301.00001` (strip version) |

### Step 2 — Get the versionId

Fetch the paper metadata:

```
GET https://api.alphaxiv.org/papers/v3/{PAPER_ID}
```

No authentication required. From the JSON response, extract the `versionId` field.

### Step 3 — Get the overview

```
GET https://api.alphaxiv.org/papers/v3/{VERSION_ID}/overview/en
```

Replace `en` with another language code if the user requests a different language.

## Response format

The overview response contains several useful fields:

- **`intermediateReport`** — Machine-optimized structured text. Prefer this when available; it is the most information-dense format.
- **`overview`** — Human-readable markdown summary.
- **`summary`** — Structured object with fields: `summary`, `originalProblem`, `solution`, `keyInsights`, `results`.
- **`citations`** — Referenced papers with context.

## Presenting results

1. Use the `intermediateReport` if available; fall back to `overview` or `summary`.
2. Present the key information conversationally — don't just dump raw JSON.
3. Highlight: what problem the paper solves, the approach, key results, and why it matters.
4. If the user asks follow-up questions, refer back to the fetched data rather than re-fetching.
