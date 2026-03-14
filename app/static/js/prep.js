/* ── PDF.js lazy loader ──────────────────────────── */
var _pdfjsLib = null;
async function _ensurePdfjs() {
  if (_pdfjsLib) return _pdfjsLib;
  _pdfjsLib = await import('https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs');
  _pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs';
  return _pdfjsLib;
}

/* ── Prep: Notes Viewer ────────────────────────── */

let _currentNote = null;
let _currentArxivId = null;       // arxiv ID for the current note (null if not a paper)
let _currentSource = 'notes';     // 'notes' | 'alphaxiv' | 'pdf'
let _cachedNotesHtml = null;      // cached rendered markdown for quick toggle back
let _cachedAlphaxivHtml = null;   // cached alphaxiv content
let _pdfDoc = null;               // current pdf.js document
let _pdfRenderedPages = new Map();
let _pdfObserver = null;
let _pdfScale = 1.0;

function _isMobile() {
  return window.innerWidth <= 640;
}

function loadNote(filename, btn) {
  // Mark active
  document.querySelectorAll('.sb-notes-list__item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');

  _currentNote = filename;
  const content = document.getElementById('notesContent');
  const toolbar = document.getElementById('notesToolbar');
  const layout = document.getElementById('notesLayout');
  content.innerHTML = '<div class="sb-notes-empty"><p>Loading...</p></div>';
  toolbar.style.display = '';
  _updateNoteTitle();

  // Mobile: switch to content view and scroll to it
  if (_isMobile()) {
    layout.classList.add('sb-notes-layout--viewing');
    layout.scrollIntoView({ behavior: 'instant' });
  }

  // Reset source state
  _currentArxivId = null;
  _currentSource = 'notes';
  _cachedNotesHtml = null;
  _cachedAlphaxivHtml = null;
  var toggle = document.getElementById('sourceToggle');
  if (toggle) toggle.style.display = 'none';

  fetch('/api/prep/notes/' + encodeURIComponent(filename))
    .then(r => {
      if (!r.ok) throw new Error('Failed to load note');
      return r.json();
    })
    .then(data => {
      _cachedNotesHtml = data.html;
      content.innerHTML = '<div class="md-body">' + data.html + '</div>';
      // Apply syntax highlighting to all code blocks
      content.querySelectorAll('pre code').forEach(block => {
        if (typeof hljs !== 'undefined') hljs.highlightElement(block);
      });
      // Restore highlights from localStorage
      _restoreHighlights();
      // Scroll content to top
      content.scrollTop = 0;

      // Show source toggle if this note has an arxiv ID
      if (data.arxiv_id) {
        _currentArxivId = data.arxiv_id;
        if (toggle) {
          toggle.style.display = '';
          // Reset toggle buttons
          toggle.querySelectorAll('.sb-source-toggle__btn').forEach(function(b) {
            b.classList.toggle('active', b.dataset.source === 'notes');
          });
        }
      }
    })
    .catch(err => {
      content.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
    });
}

/* ── Source Toggle (Notes / AlphaXiv / PDF) ─────── */

function switchSource(source, btn) {
  if (source === _currentSource || !_currentArxivId) return;
  _currentSource = source;

  // Update button states
  var toggle = document.getElementById('sourceToggle');
  if (toggle) {
    toggle.querySelectorAll('.sb-source-toggle__btn').forEach(function(b) {
      b.classList.toggle('active', b === btn);
    });
  }

  var content = document.getElementById('notesContent');

  // Cleanup PDF when switching away
  _cleanupPdf();

  if (source === 'notes') {
    // Restore cached markdown
    if (_cachedNotesHtml) {
      content.innerHTML = '<div class="md-body">' + _cachedNotesHtml + '</div>';
      content.querySelectorAll('pre code').forEach(function(block) {
        if (typeof hljs !== 'undefined') hljs.highlightElement(block);
      });
      _restoreHighlights();
    }
    content.scrollTop = 0;
    return;
  }

  if (source === 'alphaxiv') {
    // Check cache first
    if (_cachedAlphaxivHtml) {
      content.innerHTML = _cachedAlphaxivHtml;
      content.querySelectorAll('pre code').forEach(function(block) {
        if (typeof hljs !== 'undefined') hljs.highlightElement(block);
      });
      content.scrollTop = 0;
      return;
    }

    content.innerHTML = '<div class="sb-notes-empty"><p>Loading AlphaXiv overview...</p></div>';
    fetch('/api/prep/alphaxiv/' + _currentArxivId)
      .then(function(r) { return r.json().then(function(d) { return { ok: r.ok, data: d }; }); })
      .then(function(res) {
        if (!res.ok) {
          content.innerHTML = '<div class="sb-notes-empty"><p>' + (res.data.error || 'Failed to load') + '</p></div>';
          return;
        }
        // Use intermediateReport first, fall back to overview, then summary
        var text = res.data.intermediateReport || res.data.overview || '';
        if (!text && res.data.summary) {
          var s = res.data.summary;
          text = '# Summary\n\n' + (s.summary || '') +
            '\n\n## Problem\n\n' + (s.originalProblem || '') +
            '\n\n## Solution\n\n' + (s.solution || '') +
            '\n\n## Key Insights\n\n' + (s.keyInsights || '') +
            '\n\n## Results\n\n' + (s.results || '');
        }
        if (!text) {
          content.innerHTML = '<div class="sb-notes-empty"><p>No AlphaXiv overview available for this paper yet.</p></div>';
          return;
        }
        // Render markdown — use a simple conversion to HTML
        _renderAlphaxivContent(content, text);
        _cachedAlphaxivHtml = content.innerHTML;
        content.scrollTop = 0;
      })
      .catch(function(err) {
        content.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
      });
    return;
  }

  if (source === 'pdf') {
    _loadPdfInPrep(content);
    return;
  }
}

function _renderAlphaxivContent(container, markdownText) {
  // Convert markdown to HTML (basic conversion for AlphaXiv content)
  // First, extract and protect images and links before escaping
  var imgPlaceholders = [];
  var linkPlaceholders = [];
  var text = markdownText;

  // Protect markdown images: ![alt](url)
  text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(_, alt, url) {
    var idx = imgPlaceholders.length;
    imgPlaceholders.push('<img src="' + url + '" alt="' + alt + '" style="max-width:100%;margin:8px 0;">');
    return '%%IMG' + idx + '%%';
  });

  // Protect markdown links: [text](url)
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function(_, linkText, url) {
    var idx = linkPlaceholders.length;
    linkPlaceholders.push('<a href="' + url + '" target="_blank" rel="noopener">' + linkText + '</a>');
    return '%%LINK' + idx + '%%';
  });

  var html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // Headers
    .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Bullet lists
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    // Horizontal rules
    .replace(/^---$/gm, '<hr>')
    // Paragraphs (double newline)
    .replace(/\n\n/g, '</p><p>')
    // Single newlines within paragraphs
    .replace(/\n/g, '<br>');

  // Wrap list items
  html = html.replace(/(<li>.*?<\/li>(\s*<br>)?)+/g, function(match) {
    return '<ul>' + match.replace(/<br>/g, '') + '</ul>';
  });

  // Restore images and links
  imgPlaceholders.forEach(function(tag, i) {
    html = html.replace('%%IMG' + i + '%%', tag);
  });
  linkPlaceholders.forEach(function(tag, i) {
    html = html.replace('%%LINK' + i + '%%', tag);
  });

  container.innerHTML = '<div class="md-body"><p>' + html + '</p></div>';

  // Syntax highlight code blocks
  container.querySelectorAll('pre code').forEach(function(block) {
    if (typeof hljs !== 'undefined') hljs.highlightElement(block);
  });
}

/* ── PDF Viewer (pdf.js) ──────────────────────── */

function _cleanupPdf() {
  if (_pdfObserver) _pdfObserver.disconnect();
  _pdfObserver = null;
  _pdfRenderedPages = new Map();
  _pdfDoc = null;
}

async function _loadPdfInPrep(content) {
  var pdfUrl = '/api/prep/arxiv-pdf/' + _currentArxivId;

  // Action bar + pages container
  content.innerHTML =
    '<div class="sb-pdf-actions">' +
      '<a href="' + pdfUrl + '" download="' + _currentArxivId + '.pdf" class="sb-pdf-actions__btn" title="Download">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
        '<span>Download</span>' +
      '</a>' +
      '<button class="sb-pdf-actions__btn" onclick="_printPdf()" title="Print">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>' +
        '<span>Print</span>' +
      '</button>' +
      '<a href="https://arxiv.org/abs/' + _currentArxivId + '" target="_blank" class="sb-pdf-actions__btn" title="Open on arXiv">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>' +
        '<span>arXiv</span>' +
      '</a>' +
      '<span class="sb-pdf-actions__page" id="prepPdfPageInfo"></span>' +
    '</div>' +
    '<div class="sb-pdf-pages" id="prepPdfPages">' +
      '<div class="sb-notes-empty"><p>Loading PDF...</p></div>' +
    '</div>';

  try {
    var resp = await fetch(pdfUrl);
    if (!resp.ok) throw new Error('Failed to load PDF');
    var buf = await resp.arrayBuffer();

    var pdfjs = await _ensurePdfjs();
    _pdfDoc = await pdfjs.getDocument({ data: buf, enableXfa: false }).promise;
    var pagesEl = document.getElementById('prepPdfPages');
    pagesEl.innerHTML = '';

    // Determine scale to fit width
    var firstPage = await _pdfDoc.getPage(1);
    var containerWidth = pagesEl.clientWidth - 32; // padding
    var baseViewport = firstPage.getViewport({ scale: 1 });
    _pdfScale = containerWidth / baseViewport.width;
    if (_pdfScale < 0.5) _pdfScale = 0.5;
    if (_pdfScale > 2.5) _pdfScale = 2.5;

    // Create placeholders
    for (var i = 1; i <= _pdfDoc.numPages; i++) {
      var div = document.createElement('div');
      div.className = 'sb-pdf-page';
      div.dataset.page = i;
      var vp = firstPage.getViewport({ scale: _pdfScale });
      div.style.width = vp.width + 'px';
      div.style.height = vp.height + 'px';
      pagesEl.appendChild(div);
      _pdfRenderedPages.set(i, { el: div, rendered: false, rendering: false });
    }

    var pageInfo = document.getElementById('prepPdfPageInfo');
    if (pageInfo) pageInfo.textContent = _pdfDoc.numPages + ' pages';

    // Lazy render with IntersectionObserver
    _pdfObserver = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var pageNum = parseInt(entry.target.dataset.page);
          var info = _pdfRenderedPages.get(pageNum);
          if (info && !info.rendered && !info.rendering) {
            _renderPrepPdfPage(pageNum);
          }
        }
      });
    }, { root: pagesEl, rootMargin: '300px 0px', threshold: 0 });

    _pdfRenderedPages.forEach(function(info) {
      _pdfObserver.observe(info.el);
    });
  } catch (err) {
    var pagesEl = document.getElementById('prepPdfPages');
    if (pagesEl) pagesEl.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
  }
}

async function _renderPrepPdfPage(pageNum) {
  var info = _pdfRenderedPages.get(pageNum);
  if (!info || info.rendered || info.rendering) return;
  info.rendering = true;

  try {
    var page = await _pdfDoc.getPage(pageNum);
    var viewport = page.getViewport({ scale: _pdfScale });

    info.el.style.width = viewport.width + 'px';
    info.el.style.height = viewport.height + 'px';

    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    canvas.width = viewport.width * dpr;
    canvas.height = viewport.height * dpr;
    canvas.style.width = viewport.width + 'px';
    canvas.style.height = viewport.height + 'px';
    ctx.scale(dpr, dpr);
    info.el.appendChild(canvas);

    await page.render({ canvasContext: ctx, viewport: viewport }).promise;
    info.rendered = true;
    info.rendering = false;
  } catch (err) {
    info.rendering = false;
    console.error('PDF render error page', pageNum, err);
  }
}

function _printPdf() {
  if (!_currentArxivId) return;
  var w = window.open('/api/prep/arxiv-pdf/' + _currentArxivId);
  w.addEventListener('load', function() { w.print(); });
}

/* ── Mobile back button ────────────────────────── */

function mobileBackToSidebar() {
  const layout = document.getElementById('notesLayout');
  layout.classList.remove('sb-notes-layout--viewing');
}

/* ── Theme Toggle ─────────────────────────────── */

var _HLJS_DARK = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css';
var _HLJS_LIGHT = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css';

function _syncThemeIcons() {
  var isDark = document.documentElement.getAttribute('data-bs-theme') !== 'light';
  var sun = document.getElementById('prepSunIcon');
  var moon = document.getElementById('prepMoonIcon');
  if (sun) sun.style.display = isDark ? 'none' : '';
  if (moon) moon.style.display = isDark ? '' : 'none';

  // Swap highlight.js theme
  var link = document.getElementById('hljsTheme');
  if (link) link.href = isDark ? _HLJS_DARK : _HLJS_LIGHT;
}

function togglePrepTheme() {
  var html = document.documentElement;
  var next = html.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-bs-theme', next);
  localStorage.setItem('sb-theme', next);
  _syncThemeIcons();
}

// Sync icons on page load (theme may already be set from localStorage via app.js)
document.addEventListener('DOMContentLoaded', _syncThemeIcons);

/* ── Fullscreen ────────────────────────────────── */

function toggleNotesFullscreen() {
  const layout = document.getElementById('notesLayout');
  layout.classList.toggle('sb-notes-layout--fullscreen');

  const isFs = layout.classList.contains('sb-notes-layout--fullscreen');
  document.getElementById('expandIcon').style.display = isFs ? 'none' : '';
  document.getElementById('collapseIcon').style.display = isFs ? '' : 'none';
  document.getElementById('notesExpandBtn').title = isFs ? 'Exit fullscreen (Esc)' : 'Fullscreen';

  // Prevent body scroll when fullscreen
  document.body.style.overflow = isFs ? 'hidden' : '';
}

// Esc key to exit fullscreen
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') {
    const layout = document.getElementById('notesLayout');
    if (layout && layout.classList.contains('sb-notes-layout--fullscreen')) {
      toggleNotesFullscreen();
    }
  }
});


/* ── Note Title & Rename ─────────────────────────── */

function _getDisplayName(filename) {
  var name = filename.replace(/\.md$/i, '').replace(/-/g, ' ').replace(/_/g, ' ');
  // Strip known prefixes
  var prefixes = ['Ilya30 ', 'BEV ', 'Paper ', 'Async ', 'MLTheory ', 'MLPaper ', 'Planner ', 'VLA ', 'CMU '];
  for (var i = 0; i < prefixes.length; i++) {
    if (name.startsWith(prefixes[i])) {
      name = name.substring(prefixes[i].length);
      // CMU lectures: "Lec01 Introduction..." → "01. Introduction..."
      var lecMatch = name.match(/^Lec(\d{2})\s+(.+)/);
      if (lecMatch) { name = lecMatch[1] + '. ' + lecMatch[2]; break; }
      // Strip leading number prefix like "01 "
      if (/^\d{2}\s/.test(name)) name = name.substring(3);
      // Strip trailing " summary"
      if (name.endsWith(' summary')) name = name.slice(0, -8);
      break;
    }
  }
  // Title-case
  return name.replace(/\b\w/g, function (c) { return c.toUpperCase(); });
}

function _updateNoteTitle() {
  var titleEl = document.getElementById('noteTitle');
  if (!titleEl) return;
  if (_currentNote) {
    titleEl.textContent = _getDisplayName(_currentNote);
    titleEl.style.display = '';
  } else {
    titleEl.textContent = '';
    titleEl.style.display = 'none';
  }
}

function startRenameNote() {
  if (!_currentNote) return;
  var titleEl = document.getElementById('noteTitle');
  if (!titleEl || titleEl.querySelector('input')) return; // Already editing

  var currentName = titleEl.textContent.trim();
  var input = document.createElement('input');
  input.type = 'text';
  input.className = 'sb-rename-input';
  input.value = currentName;

  titleEl.textContent = '';
  titleEl.appendChild(input);
  input.focus();
  input.select();

  function commit() {
    var newTitle = (input.value || '').trim();
    input.removeEventListener('blur', commit);
    input.removeEventListener('keydown', onKey);

    if (!newTitle || newTitle === currentName) {
      // Cancelled — restore original
      titleEl.textContent = currentName;
      return;
    }

    titleEl.textContent = 'Renaming...';
    fetch('/api/prep/notes/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ old_filename: _currentNote, new_title: newTitle }),
    })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        if (!res.ok) {
          alert(res.data.error || 'Rename failed');
          titleEl.textContent = currentName;
          return;
        }
        if (res.data.unchanged) {
          titleEl.textContent = currentName;
          return;
        }
        var newFilename = res.data.filename;
        // Update sidebar button
        var oldBtn = document.querySelector('.sb-notes-list__item[data-note="' + _currentNote + '"]');
        if (oldBtn) {
          oldBtn.setAttribute('data-note', newFilename);
          oldBtn.setAttribute('onclick', "loadNote('" + newFilename + "', this)");
          // Update display text
          var svgEl = oldBtn.querySelector('svg');
          oldBtn.textContent = '';
          if (svgEl) oldBtn.appendChild(svgEl);
          oldBtn.appendChild(document.createTextNode(' ' + _getDisplayName(newFilename)));
        }
        _currentNote = newFilename;
        _updateNoteTitle();
      })
      .catch(function (err) {
        alert('Rename failed: ' + err.message);
        titleEl.textContent = currentName;
      });
  }

  function onKey(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      input.blur();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      input.value = currentName;
      input.blur();
    }
  }

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', onKey);
}


/* ── Text Highlighting ─────────────────────────── */
// Select text → click highlight button → wraps in <mark>.
// Click a highlight to remove it. Stored in localStorage per note.

let _highlightMode = false;

function toggleHighlightMode() {
  _highlightMode = !_highlightMode;
  const btn = document.getElementById('highlightToggleBtn');
  btn.classList.toggle('active', _highlightMode);
  btn.title = _highlightMode ? 'Highlighting ON — select text to highlight' : 'Toggle highlighting';

  const content = document.getElementById('notesContent');
  content.classList.toggle('sb-highlight-mode', _highlightMode);
}

// Listen for mouseup inside notes content to highlight selection
document.addEventListener('mouseup', function (e) {
  if (!_highlightMode || !_currentNote) return;
  const content = document.getElementById('notesContent');
  if (!content || !content.contains(e.target)) return;

  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return;

  const range = sel.getRangeAt(0);
  // Don't highlight inside already-highlighted text
  if (range.commonAncestorContainer.closest && range.commonAncestorContainer.closest('mark.user-highlight')) return;
  // Ensure selection is within content
  if (!content.contains(range.commonAncestorContainer)) return;

  try {
    const mark = document.createElement('mark');
    mark.className = 'user-highlight';
    mark.addEventListener('click', _onHighlightClick);
    range.surroundContents(mark);
    sel.removeAllRanges();
    _saveHighlights();
  } catch (err) {
    // surroundContents fails on partial element selections — that's ok
    console.warn('Could not highlight selection:', err.message);
  }
});

function _onHighlightClick(e) {
  if (!_highlightMode) return;
  const mark = e.currentTarget;
  // Unwrap: replace <mark> with its text content
  const parent = mark.parentNode;
  while (mark.firstChild) {
    parent.insertBefore(mark.firstChild, mark);
  }
  parent.removeChild(mark);
  parent.normalize();
  _saveHighlights();
}

function _getHighlightKey() {
  return 'highlights_' + (_currentNote || '').replace(/[^a-zA-Z0-9]/g, '_');
}

function _saveHighlights() {
  if (!_currentNote) return;
  const content = document.getElementById('notesContent');
  const marks = content.querySelectorAll('mark.user-highlight');
  const highlights = [];
  marks.forEach(m => {
    // Save text + approximate path for restoration
    highlights.push({
      text: m.textContent,
      // Store context: a few chars before and after to locate position
      context: _getHighlightContext(m),
    });
  });
  try {
    localStorage.setItem(_getHighlightKey(), JSON.stringify(highlights));
  } catch (e) { /* quota exceeded — silently ignore */ }
}

function _getHighlightContext(mark) {
  const parent = mark.parentNode;
  if (!parent) return '';
  const fullText = parent.textContent || '';
  const markText = mark.textContent;
  const idx = fullText.indexOf(markText);
  if (idx === -1) return '';
  const before = fullText.substring(Math.max(0, idx - 30), idx);
  const after = fullText.substring(idx + markText.length, idx + markText.length + 30);
  return before + '|||' + markText + '|||' + after;
}

function _restoreHighlights() {
  if (!_currentNote) return;
  const key = _getHighlightKey();
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(key) || '[]');
  } catch (e) { return; }
  if (!saved.length) return;

  const content = document.getElementById('notesContent');
  const mdBody = content.querySelector('.md-body');
  if (!mdBody) return;

  // Walk text nodes and find matches
  saved.forEach(h => {
    if (!h.text) return;
    const walker = document.createTreeWalker(mdBody, NodeFilter.SHOW_TEXT, null, false);
    let node;
    while (node = walker.nextNode()) {
      const idx = node.textContent.indexOf(h.text);
      if (idx === -1) continue;

      // Verify context if available
      if (h.context) {
        const parts = h.context.split('|||');
        if (parts.length === 3) {
          const parentText = node.parentNode.textContent || '';
          const foundIdx = parentText.indexOf(h.text);
          if (foundIdx === -1) continue;
          const beforeCtx = parentText.substring(Math.max(0, foundIdx - 30), foundIdx);
          if (parts[0] && !beforeCtx.includes(parts[0].slice(-10))) continue;
        }
      }

      try {
        const range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + h.text.length);
        const mark = document.createElement('mark');
        mark.className = 'user-highlight';
        mark.addEventListener('click', _onHighlightClick);
        range.surroundContents(mark);
      } catch (e) { /* skip partial matches */ }
      break; // Only highlight first occurrence
    }
  });
}

/* ── Clear all highlights for current note ─────── */
function clearAllHighlights() {
  if (!_currentNote) return;
  const content = document.getElementById('notesContent');
  content.querySelectorAll('mark.user-highlight').forEach(mark => {
    const parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parent.normalize();
  });
  localStorage.removeItem(_getHighlightKey());
}


/* ── AI Chat Panel ───────────────────────────────── */

let _chatHistory = [];
let _selectedSection = null;
let _chatStreaming = false;
let _chatAbortController = null;

/* ── Smooth streaming buffer ─────────────────────── */
var _streamBuffer = '';       // text waiting to be displayed
var _streamDisplayed = '';    // text currently rendered
var _streamTargetEl = null;   // DOM element to render into
var _streamRafId = null;      // requestAnimationFrame handle
var _streamDone = false;      // SSE stream finished?
var _streamCharsPerFrame = 2; // base chars per frame (~120 chars/sec at 60fps)
var _streamLastRender = 0;    // timestamp of last markdown render
var _streamRenderInterval = 40; // ms between full markdown re-renders

function _streamTick(ts) {
  if (!_streamTargetEl) return;

  // Calculate how many chars to drain this frame
  var pending = _streamBuffer.length;
  // Speed up when buffer is large so we don't fall behind
  var speed = _streamCharsPerFrame;
  if (pending > 200) speed = Math.ceil(pending / 8);
  else if (pending > 80) speed = Math.ceil(pending / 15);
  else if (pending > 30) speed = Math.ceil(pending / 6);

  if (pending > 0) {
    var chunk = _streamBuffer.slice(0, speed);
    _streamBuffer = _streamBuffer.slice(speed);
    _streamDisplayed += chunk;

    // Throttle full markdown re-renders for performance
    if (ts - _streamLastRender >= _streamRenderInterval || _streamBuffer.length === 0) {
      _renderMarkdownInChat(_streamTargetEl, _streamDisplayed);
      _streamLastRender = ts;
      _scrollChatToBottom();
    }
  }

  // Keep ticking if there's still buffered text or stream hasn't ended
  if (_streamBuffer.length > 0 || !_streamDone) {
    _streamRafId = requestAnimationFrame(_streamTick);
  } else {
    // Final render to ensure everything is flushed
    _renderMarkdownInChat(_streamTargetEl, _streamDisplayed);
    _scrollChatToBottom();
    _streamTargetEl.classList.remove('is-streaming');
    _streamRafId = null;
  }
}

function _streamStart(targetEl) {
  _streamBuffer = '';
  _streamDisplayed = '';
  _streamTargetEl = targetEl;
  _streamDone = false;
  _streamLastRender = 0;
  targetEl.classList.add('is-streaming');
  if (_streamRafId) cancelAnimationFrame(_streamRafId);
  _streamRafId = requestAnimationFrame(_streamTick);
}

function _streamPush(text) {
  _streamBuffer += text;
  // Restart animation loop if it stopped
  if (!_streamRafId && _streamTargetEl) {
    _streamRafId = requestAnimationFrame(_streamTick);
  }
}

function _streamEnd() {
  _streamDone = true;
  // If animation already stopped but buffer has text, restart
  if (!_streamRafId && _streamBuffer.length > 0 && _streamTargetEl) {
    _streamRafId = requestAnimationFrame(_streamTick);
  }
}

function _streamGetFullText() {
  return _streamDisplayed + _streamBuffer;
}

function _streamCleanup() {
  if (_streamRafId) cancelAnimationFrame(_streamRafId);
  if (_streamTargetEl) _streamTargetEl.classList.remove('is-streaming');
  _streamRafId = null;
  _streamTargetEl = null;
  _streamBuffer = '';
  _streamDisplayed = '';
  _streamDone = true;
}

function toggleChat() {
  const layout = document.getElementById('notesLayout');
  const panel = document.getElementById('chatPanel');
  if (!layout || !panel) return;

  // Only allow chat in fullscreen mode
  if (!layout.classList.contains('sb-notes-layout--fullscreen')) return;

  const isOpen = panel.classList.contains('sb-chat-panel--open');
  panel.classList.toggle('sb-chat-panel--open', !isOpen);
  layout.classList.toggle('sb-chat-open', !isOpen);

  const btn = document.getElementById('chatToggleBtn');
  if (btn) btn.classList.toggle('active', !isOpen);

  if (!isOpen) {
    // Focus input when opening
    setTimeout(function () {
      var input = document.getElementById('chatInput');
      if (input) input.focus();
    }, 350);
  }
}

// Close chat when exiting fullscreen
(function () {
  var origToggle = toggleNotesFullscreen;
  toggleNotesFullscreen = function () {
    origToggle();
    var layout = document.getElementById('notesLayout');
    if (layout && !layout.classList.contains('sb-notes-layout--fullscreen')) {
      // Exiting fullscreen — close chat
      var panel = document.getElementById('chatPanel');
      if (panel) panel.classList.remove('sb-chat-panel--open');
      layout.classList.remove('sb-chat-open');
      var btn = document.getElementById('chatToggleBtn');
      if (btn) btn.classList.remove('active');
    }
  };
})();

function sendChatMessage() {
  if (_chatStreaming) return;
  var input = document.getElementById('chatInput');
  var message = (input.value || '').trim();
  if (!message || !_currentNote) return;

  input.value = '';
  autoResizeChatInput();

  // Add user message bubble
  _appendMessage('user', message);
  _chatHistory.push({ role: 'user', content: message });

  // Create assistant bubble (streaming)
  var assistantEl = _appendMessage('assistant', '');
  var textEl = assistantEl.querySelector('.sb-chat-msg__text');

  // Show typing indicator
  var dots = document.createElement('div');
  dots.className = 'sb-chat-typing';
  dots.innerHTML = '<span></span><span></span><span></span>';
  textEl.appendChild(dots);
  _scrollChatToBottom();

  _chatStreaming = true;
  _setChatInputState(false);
  _chatAbortController = new AbortController();

  var body = {
    filename: _currentNote,
    message: message,
    history: _chatHistory.slice(0, -1),  // Don't include current message (already sent separately)
    section: _selectedSection,
  };

  fetch('/api/prep/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: _chatAbortController.signal,
  })
    .then(function (response) {
      if (!response.ok) {
        if (response.status === 503) throw new Error('API key not configured. Set ANTHROPIC_API_KEY in your environment.');
        throw new Error('Chat request failed (' + response.status + ')');
      }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var sseBuffer = '';

      // Remove typing dots and start smooth stream
      var dotsEl = textEl.querySelector('.sb-chat-typing');
      if (dotsEl) dotsEl.remove();
      _streamStart(textEl);

      function read() {
        return reader.read().then(function (result) {
          if (result.done) {
            _streamEnd();
            _finishStreaming(_streamGetFullText());
            return;
          }
          sseBuffer += decoder.decode(result.value, { stream: true });
          var lines = sseBuffer.split('\n');
          sseBuffer = lines.pop() || '';

          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line.startsWith('data: ')) continue;
            try {
              var data = JSON.parse(line.slice(6));
              if (data.error) {
                _streamCleanup();
                textEl.textContent = 'Error: ' + data.error;
                _finishStreaming('');
                return;
              }
              if (data.done) {
                _streamEnd();
                _finishStreaming(_streamGetFullText(), data.usage);
                return;
              }
              if (data.text) {
                _streamPush(data.text);
              }
            } catch (e) { /* skip malformed lines */ }
          }
          return read();
        });
      }
      return read();
    })
    .catch(function (err) {
      if (err.name === 'AbortError') { _streamCleanup(); return; }
      _streamCleanup();
      var dotsEl = textEl.querySelector('.sb-chat-typing');
      if (dotsEl) dotsEl.remove();
      textEl.textContent = 'Error: ' + err.message;
      _finishStreaming('');
    });
}

function _finishStreaming(fullText, usage) {
  _chatStreaming = false;
  _setChatInputState(true);
  if (fullText) {
    _chatHistory.push({ role: 'assistant', content: fullText });
  }
  // Cap history at 20 messages
  if (_chatHistory.length > 20) {
    _chatHistory = _chatHistory.slice(-20);
  }
}

function _appendMessage(role, text) {
  var messages = document.getElementById('chatMessages');
  // Remove welcome message if present
  var welcome = messages.querySelector('.sb-chat-welcome');
  if (welcome) welcome.remove();

  var msg = document.createElement('div');
  msg.className = 'sb-chat-msg sb-chat-msg--' + role;

  var textEl = document.createElement('div');
  textEl.className = 'sb-chat-msg__text';
  if (text) {
    if (role === 'assistant') {
      _renderMarkdownInChat(textEl, text);
    } else {
      textEl.textContent = text;
    }
  }
  msg.appendChild(textEl);
  messages.appendChild(msg);
  _scrollChatToBottom();
  return msg;
}

function _renderMarkdownInChat(el, text) {
  // Simple markdown rendering for chat: bold, italic, code, code blocks, lists
  var html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Line breaks
    .replace(/\n/g, '<br>');
  el.innerHTML = html;
}

function _scrollChatToBottom() {
  var messages = document.getElementById('chatMessages');
  if (messages) {
    messages.scrollTop = messages.scrollHeight;
  }
}

function _setChatInputState(enabled) {
  var input = document.getElementById('chatInput');
  var btn = document.getElementById('chatSendBtn');
  if (input) {
    input.disabled = !enabled;
    if (enabled) input.focus();
  }
  if (btn) btn.disabled = !enabled;
}

function handleChatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
}

function autoResizeChatInput() {
  var input = document.getElementById('chatInput');
  if (!input) return;
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
}

/* ── Section selection for chat context ──────────── */

var _pendingContextText = null; // Stores selected text when "Use as context" button appears

document.addEventListener('mouseup', function (e) {
  // Don't show "Use as context" in highlight mode or if chat isn't open
  if (_highlightMode) return;
  var panel = document.getElementById('chatPanel');
  if (!panel || !panel.classList.contains('sb-chat-panel--open')) return;

  var content = document.getElementById('notesContent');
  if (!content || !content.contains(e.target)) return;

  // Ignore mouseup on the context button itself
  var btn = document.getElementById('chatUseContextBtn');
  if (btn && (e.target === btn || btn.contains(e.target))) return;

  var sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount || !btn) {
    btn.style.display = 'none';
    _pendingContextText = null;
    return;
  }

  var text = sel.toString().trim();
  if (text.length < 20) {
    btn.style.display = 'none';
    _pendingContextText = null;
    return;
  }

  // Store the text now — by the time the button is clicked the selection will be gone
  _pendingContextText = text;

  // Position button near the selection
  var range = sel.getRangeAt(0);
  var rect = range.getBoundingClientRect();
  var mainRect = content.closest('.sb-notes-main').getBoundingClientRect();

  btn.style.display = '';
  btn.style.top = (rect.bottom - mainRect.top + 4) + 'px';
  btn.style.left = (rect.left - mainRect.left + rect.width / 2 - 60) + 'px';
});

// Hide context button on click elsewhere (but NOT on the button itself)
document.addEventListener('mousedown', function (e) {
  var btn = document.getElementById('chatUseContextBtn');
  if (btn && e.target !== btn && !btn.contains(e.target)) {
    btn.style.display = 'none';
    _pendingContextText = null;
  }
});

function setChatSectionFromSelection() {
  // Use the stored text — selection is already collapsed by the time onclick fires
  var text = _pendingContextText;
  if (!text) return;

  _selectedSection = text;
  _pendingContextText = null;

  // Clear any remaining selection
  var sel = window.getSelection();
  if (sel) sel.removeAllRanges();

  // Update UI
  var contextBar = document.getElementById('chatContext');
  var contextText = document.getElementById('chatContextText');
  var btn = document.getElementById('chatUseContextBtn');
  if (btn) btn.style.display = 'none';

  if (contextBar) contextBar.style.display = '';
  if (contextText) {
    var preview = text.length > 60 ? text.substring(0, 60) + '…' : text;
    contextText.textContent = preview;
  }
}

function clearChatSection() {
  _selectedSection = null;
  var contextBar = document.getElementById('chatContext');
  if (contextBar) contextBar.style.display = 'none';
}

/* ── Reset chat on note change ───────────────────── */
(function () {
  var origLoadNote = loadNote;
  loadNote = function (filename, btn) {
    // Reset chat state when switching notes
    _chatHistory = [];
    _selectedSection = null;
    _streamCleanup();
    var messages = document.getElementById('chatMessages');
    if (messages) {
      messages.innerHTML = '<div class="sb-chat-welcome"><p>Ask anything about this document</p></div>';
    }
    clearChatSection();
    // Abort any in-progress stream
    if (_chatAbortController) {
      _chatAbortController.abort();
      _chatAbortController = null;
    }
    _chatStreaming = false;
    _setChatInputState(true);
    return origLoadNote(filename, btn);
  };
})();
