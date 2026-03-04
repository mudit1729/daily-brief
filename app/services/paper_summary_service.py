"""
Paper Summary Service — fetches arxiv papers and generates
implementation-focused 12-section markdown summaries using LLM.
"""
import logging
import os
import re
import requests
from flask import current_app

logger = logging.getLogger(__name__)

# --------------- Arxiv helpers ---------------

def _extract_arxiv_id(url_or_id: str) -> str | None:
    """Extract arxiv paper ID from various URL formats or bare ID."""
    url_or_id = url_or_id.strip()
    patterns = [
        r'arxiv\.org/abs/([0-9]+\.[0-9]+(?:v\d+)?)',
        r'arxiv\.org/pdf/([0-9]+\.[0-9]+(?:v\d+)?)',
        r'arxiv\.org/html/([0-9]+\.[0-9]+(?:v\d+)?)',
        r'^([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)$',
    ]
    for pat in patterns:
        m = re.search(pat, url_or_id)
        if m:
            return m.group(1)
    return None


def _fetch_arxiv_html(arxiv_id: str) -> tuple[str, str]:
    """Fetch paper HTML from arxiv. Returns (title, body_text).
    Tries HTML endpoint first, falls back to abstract page."""
    # Try the HTML rendering (best quality)
    html_url = f'https://arxiv.org/html/{arxiv_id}'
    try:
        resp = requests.get(html_url, timeout=60, headers={
            'User-Agent': 'PaperSummarizer/1.0 (research tool)',
        })
        if resp.status_code == 200 and len(resp.text) > 2000:
            body = resp.text
            # Extract title from <title> tag
            title_match = re.search(r'<title>(.*?)</title>', body, re.DOTALL)
            title = title_match.group(1).strip() if title_match else arxiv_id
            title = re.sub(r'\s*[-–—|]\s*arXiv.*$', '', title).strip()
            # Strip HTML tags for a rough plain-text version
            text = _html_to_text(body)
            return title, text
    except Exception as e:
        logger.warning(f"HTML fetch failed for {arxiv_id}: {e}")

    # Fallback: abstract page for title + whatever we can get
    abs_url = f'https://arxiv.org/abs/{arxiv_id}'
    try:
        resp = requests.get(abs_url, timeout=30, headers={
            'User-Agent': 'PaperSummarizer/1.0 (research tool)',
        })
        body = resp.text
        title_match = re.search(r'<meta name="citation_title" content="(.*?)"', body)
        title = title_match.group(1) if title_match else arxiv_id
        # Extract abstract
        abs_match = re.search(
            r'<blockquote class="abstract.*?">\s*<span class="descriptor">Abstract:</span>\s*(.*?)</blockquote>',
            body, re.DOTALL,
        )
        abstract = _html_to_text(abs_match.group(1)) if abs_match else ''
        text = f"Title: {title}\nAbstract: {abstract}\n\n[Full HTML not available — summary based on abstract + model knowledge]"
        return title, text
    except Exception as e:
        logger.error(f"Abstract fetch also failed for {arxiv_id}: {e}")
        raise ValueError(f"Could not fetch paper {arxiv_id} from arxiv")


def _html_to_text(html: str) -> str:
    """Quick HTML → plain text conversion."""
    # Remove script/style blocks
    text = re.sub(r'<(script|style|nav|header|footer)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
    # Convert <br> and block tags to newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</(p|div|h[1-6]|li|tr|section)>', '\n', text, flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&nbsp;', ' ')
    text = re.sub(r'&#?\w+;', '', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _slugify(title: str) -> str:
    """Convert paper title to a readable, filesystem-safe Title-Case slug.
    e.g. 'Lift, Splat, Shoot: Encoding Images...' → 'Lift-Splat-Shoot'
    """
    # Remove subtitle after colon (usually verbose)
    short = title.split(':')[0].split('—')[0].strip()
    # Keep only alphanumeric and spaces
    short = re.sub(r'[^a-zA-Z0-9 ]+', ' ', short)
    # Title-case each word, join with dashes
    words = short.split()[:6]  # max 6 words
    slug = '-'.join(w.capitalize() for w in words)
    return slug[:60]


# --------------- System prompt (12-section skill) ---------------

PAPER_SUMMARY_SYSTEM_PROMPT = """You are an expert ML research engineer creating implementation-focused paper summaries. Given a research paper's content, produce a comprehensive 12-section markdown summary that an engineer could use as a standalone reference to reimplement the paper.

## Output Format
The summary has exactly 12 sections, always in this order. Every section is mandatory — if the paper doesn't provide enough detail for a section, note what's missing and fill in plausible defaults clearly marked as such.

## The 12 Sections

### Section 1: One-page Overview
- **Paper metadata** as a header block: full title, authors, affiliations, venue, arxiv ID
- **Tasks solved**: What problem does this paper address? Cite the section. Format: `[PaperName | Sec X]`
- **Sensors / inputs assumed**: What data does it expect?
- **Key novelty**: 3–6 bullet points, each with a section citation.
- **"If you only remember 3 things"**: 3 numbered takeaways.

### Section 2: Problem Setup and Outputs (Precise)
Describe exact inputs and outputs with tensor shapes using markdown tables:
| Tensor | Shape | Description |

### Section 3: Coordinate Frames and Geometry
All coordinate frames, transforms, grid parameters, spatial augmentations, temporal alignment.
Include **Geometry sanity checks table**.

### Section 4: Architecture Deep Dive (Module-by-module)
Start with an **ASCII block diagram**. Then describe each module in a table:
| Module | Purpose | Input → Output | Key operations |

### Section 5: Forward Pass Pseudocode (Shape-annotated)
Complete `forward()` function in Python-style pseudocode with shape comments on every tensor.

### Section 6: Heads, Targets, and Losses
Tables for each head, loss terms with formulas/weights, assignment strategy.
**Loss debugging checklist** table: Bug | Symptom | Quick test.

### Section 7: Data Pipeline and Augmentations
All augmentations with parameter ranges. **Augmentation safety table**.

### Section 8: Training Pipeline (Reproducible)
Table of all hyperparameters. **Stability & convergence table**.

### Section 9: Dataset + Evaluation Protocol
Dataset details, splits, metrics, preprocessing.

### Section 10: Results Summary + Ablations
Main results + **top 3 ablations** with analysis.

### Section 11: Practical Insights
- **10 engineering takeaways** (numbered, actionable)
- **5 gotchas** (specific bugs and foot-guns)
- **Tiny-subset overfit plan** table

### Section 12: Minimal Reimplementation Checklist
- **Build order**: numbered dependency-ordered list
- **Unit tests** table: test name | what it checks
- **Minimal sanity scripts**: 2-3 one-sentence descriptions

## Style Guidelines
- Precision over prose. Tables, code blocks, structured formats.
- Cite with `[PaperName | Sec X]` throughout.
- Tensor shapes everywhere in backticks.
- ASCII diagrams over words.
- Be honest about gaps — mark as "Plausible default (not from paper)".
- Tables for: tensor shapes, hyperparameters, augmentations, debugging checklists, module descriptions, loss terms.
- The "gotchas" and practical insights are where this summary adds the most value.

## Markdown Formatting
- H1 (`#`) for the paper name
- H2 (`##`) for each numbered section
- H3 (`###`) for subsections
- Fenced code blocks with language tags
- Pipe tables with header separators
- Inline backticks for tensor shapes, variable names
- Bold for key terms on first use
- Citations in square brackets: `[PaperName | Sec X]`"""


# --------------- Main service ---------------

class PaperSummaryService:
    def __init__(self, app_config=None):
        self.config = app_config or current_app.config
        self.notes_dir = self.config.get('PREP_NOTES_DIR', 'notes')

    def summarize_arxiv(self, arxiv_url: str) -> dict:
        """Fetch an arxiv paper and generate a 12-section summary.

        Returns: {filename, title, arxiv_id, path}
        """
        arxiv_id = _extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            raise ValueError(f"Could not parse arxiv ID from: {arxiv_url}")

        logger.info(f"Fetching paper {arxiv_id} from arxiv...")
        title, paper_text = _fetch_arxiv_html(arxiv_id)
        logger.info(f"Fetched '{title}' ({len(paper_text)} chars)")

        # Truncate if extremely long (models have limits)
        max_input_chars = 120_000
        if len(paper_text) > max_input_chars:
            paper_text = paper_text[:max_input_chars] + "\n\n[... truncated for length ...]"

        # Generate summary with LLM
        logger.info(f"Generating 12-section summary for '{title}'...")
        from app.integrations.llm_gateway import LLMGateway
        llm = LLMGateway()

        result = llm.call(
            messages=[
                {'role': 'system', 'content': PAPER_SUMMARY_SYSTEM_PROMPT},
                {'role': 'user', 'content': (
                    f"Paper: {title} (arxiv: {arxiv_id})\n\n"
                    f"--- PAPER CONTENT ---\n{paper_text}"
                )},
            ],
            purpose=f'paper_summary.{arxiv_id}',
            section=None,  # counts against overall daily budget
            max_tokens=10000,
            provider='anthropic',
        )

        summary_md = result['content']
        logger.info(
            f"Summary generated: {result['total_tokens']} tokens, "
            f"${result['cost_usd']:.4f}"
        )

        # Save to notes directory
        slug = _slugify(title)
        filename = f"Paper-{slug}.md"
        filepath = os.path.join(self.notes_dir, filename)

        # Avoid clobbering existing files
        if os.path.exists(filepath):
            filename = f"Paper-{slug}-{arxiv_id.replace('.', '')}.md"
            filepath = os.path.join(self.notes_dir, filename)

        os.makedirs(self.notes_dir, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(summary_md)

        logger.info(f"Paper summary saved to {filepath}")

        return {
            'filename': filename,
            'title': title,
            'arxiv_id': arxiv_id,
            'path': filepath,
            'tokens_used': result['total_tokens'],
            'cost_usd': result['cost_usd'],
        }
