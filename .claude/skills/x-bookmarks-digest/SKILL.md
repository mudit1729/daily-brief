---
name: X Bookmarks Digest
description: Scrape X (Twitter) bookmarks using Chrome browser automation, then process each bookmark into a structured markdown document in the Prep viewer. Handles arxiv papers, videos, news articles, and tweets differently. Use when the user asks to process, digest, or import their X/Twitter bookmarks.
---

# X Bookmarks Digest

Scrape the user's X bookmarks via Chrome browser automation and create structured markdown notes for each bookmark in the Prep viewer.

## When to trigger

- User asks to "process my X bookmarks", "digest my bookmarks", "import bookmarks from X"
- User mentions X/Twitter bookmarks in context of the Prep viewer or study notes
- User says something like "get my bookmarks" or "bookmark digest"

## Prerequisites

- User must be logged into X (x.com) in their Chrome browser
- Claude in Chrome MCP tools must be available

## Step-by-step Process

### Phase 1: Scrape Bookmarks from X

1. **Get Chrome tab context** using `tabs_context_mcp` (createIfEmpty: true)
2. **Create a new tab** using `tabs_create_mcp`
3. **Navigate** to `https://x.com/i/bookmarks` using the `navigate` tool
4. **Wait** 3 seconds for the page to load
5. **Take a screenshot** to verify the bookmarks page loaded and user is logged in

6. **Extract bookmarks using JavaScript** — run `javascript_tool` to extract bookmark data:

```javascript
(function() {
  const articles = document.querySelectorAll('article[data-testid="tweet"]');
  const bookmarks = [];
  articles.forEach(article => {
    try {
      // Get author info
      const userLinks = article.querySelectorAll('a[role="link"]');
      let handle = '', displayName = '';
      for (const link of userLinks) {
        const href = link.getAttribute('href') || '';
        if (href.startsWith('/') && !href.includes('/status/') && href.length > 1) {
          handle = href.substring(1);
          displayName = link.textContent.trim();
          break;
        }
      }

      // Get tweet text
      const tweetTextEl = article.querySelector('[data-testid="tweetText"]');
      const tweetText = tweetTextEl ? tweetTextEl.innerText : '';

      // Get timestamp
      const timeEl = article.querySelector('time');
      const datetime = timeEl ? timeEl.getAttribute('datetime') : '';

      // Get tweet link
      const statusLinks = article.querySelectorAll('a[href*="/status/"]');
      let tweetUrl = '';
      for (const link of statusLinks) {
        const href = link.getAttribute('href') || '';
        if (href.match(/\/status\/\d+$/)) {
          tweetUrl = 'https://x.com' + href;
          break;
        }
      }

      // Get embedded URLs (links in tweet text, card links)
      const urls = [];
      const linkEls = article.querySelectorAll('a[href]');
      linkEls.forEach(a => {
        const href = a.getAttribute('href') || '';
        // External links (t.co redirects show the real URL in text)
        if (href.startsWith('https://t.co/')) {
          const displayUrl = a.textContent.trim();
          if (displayUrl && !displayUrl.startsWith('@') && !displayUrl.startsWith('#')) {
            urls.push(displayUrl);
          }
        }
      });

      // Get card link if present (for article previews)
      const card = article.querySelector('[data-testid="card.wrapper"] a[href]');
      if (card) {
        urls.push(card.getAttribute('href'));
      }

      if (tweetText || urls.length > 0) {
        bookmarks.push({
          handle,
          displayName,
          text: tweetText.substring(0, 500),
          datetime,
          tweetUrl,
          urls: [...new Set(urls)]
        });
      }
    } catch(e) {}
  });
  return JSON.stringify(bookmarks);
})()
```

7. **Scroll down** to load more bookmarks — use the `computer` tool with action `scroll`, direction `down`, repeat several times with waits between scrolls
8. **Re-extract** after each scroll batch to get newly loaded bookmarks
9. **Merge and deduplicate** bookmarks by tweetUrl
10. **Stop scrolling** when:
    - Bookmarks are older than the cutoff date (default: 2 months ago)
    - No new bookmarks loaded after scrolling
    - At least 3 scroll attempts with no new content

11. **Resolve t.co URLs** — For each bookmark with URLs, use `javascript_tool` to follow redirects:
```javascript
// For URLs that look like shortened links, try to get the card/preview URL
// The displayed text in the tweet usually shows the real domain
```

### Phase 2: Classify Bookmarks

For each bookmark, classify by checking embedded URLs:

| URL Pattern | Type |
|-------------|------|
| `arxiv.org` | paper |
| `youtube.com`, `youtu.be` | video |
| `github.com` | repo |
| News domains (nytimes, bbc, reuters, theverge, techcrunch, etc.) | article |
| Any other URL | article (default for URLs) |
| No embedded URL | tweet |

### Phase 3: Process Each Bookmark

Before processing, check if a file already exists:
```
Glob for: notes/XBookmark-*-{slug}*.md
```
If found, skip that bookmark (idempotent).

#### For Papers (arxiv links):
1. Extract the arxiv ID from the URL
2. Use WebFetch to get the alphaxiv overview: `https://api.alphaxiv.org/papers/v3/{PAPER_ID}`
3. Then fetch: `https://api.alphaxiv.org/papers/v3/{VERSION_ID}/overview/en`
4. Write a comprehensive paper summary using the intermediateReport or overview

#### For Videos (YouTube links):
1. Use WebFetch on the YouTube URL to get title, channel, description
2. Summarize the video topic based on available metadata and the tweet context

#### For Articles (news/blog links):
1. Use WebFetch to read the article content
2. Use WebSearch for additional context on the topic
3. Write an in-depth summary with context and analysis

#### For Tweets (no URL):
1. Capture the tweet text and author
2. If the tweet discusses a specific technical topic, use WebSearch to add context
3. Write a brief note with the tweet content and any relevant background

### Phase 4: Generate Markdown Files

Write each bookmark to `notes/XBookmark-{YYYY-MM-DD}-{slug}.md`:
- `YYYY-MM-DD` = the bookmark's date (from the tweet timestamp)
- `slug` = kebab-case of first 5-6 words of title, max 50 chars

**Template:**
```markdown
# {Title}

| Field | Value |
|-------|-------|
| Source | [{tweet URL}]({tweet URL}) |
| Type | Paper / Video / Article / Tweet |
| Date | {YYYY-MM-DD} |
| Author | @{handle} |
| Links | {embedded URLs} |

---

## Original Tweet
> {tweet text}

## Summary
{Content-type-specific detailed summary}

## Key Points
- {point 1}
- {point 2}
- ...

## Context & Analysis
{In-depth research, related work, why this matters}
```

### Phase 5: Progress Tracking

- Use TodoWrite to track which bookmarks have been processed
- Report a final summary: total bookmarks found, processed, skipped, and any errors

## Important Notes

- Always ask the user before starting if they want to filter by date range (default: last 2 months)
- The skill is idempotent — re-running skips already-processed bookmarks
- If Chrome is not on the bookmarks page or user is not logged in, stop and tell the user
- Rate-limit WebFetch calls — don't hammer external sites
- For very long articles, summarize rather than trying to capture everything
