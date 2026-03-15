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
      // Typeset math with MathJax if available
      if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        MathJax.typesetPromise([content]).catch(function(err) {
          console.warn('MathJax typeset error:', err);
        });
      }
      // Load annotations (highlights + comments) from backend
      _loadAnnotations();
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

/* ── Browser Fullscreen (F11-style) ──────────── */

function toggleBrowserFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(function() {});
  } else {
    document.exitFullscreen();
  }
}

document.addEventListener('fullscreenchange', function() {
  var isFs = !!document.fullscreenElement;
  document.getElementById('browserFsEnter').style.display = isFs ? 'none' : '';
  document.getElementById('browserFsExit').style.display = isFs ? '' : 'none';
  document.getElementById('browserFullscreenBtn').title = isFs ? 'Exit browser fullscreen' : 'Browser fullscreen';
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


/* ── Text Highlighting & Comments ────────────────── */
// Highlights & comments are persisted to the backend (.annotations/ directory)
// so they survive across git pushes, browser clears, and devices.

let _highlightMode = false;
let _commentMode = false;
let _annotations = { highlights: [], comments: [] };
let _saveTimer = null;
let _commentIdCounter = 0;

function toggleHighlightMode() {
  _highlightMode = !_highlightMode;
  if (_highlightMode) _commentMode = false;  // mutual exclusion
  _updateAnnotationModes();
}

function toggleCommentMode() {
  _commentMode = !_commentMode;
  if (_commentMode) _highlightMode = false;  // mutual exclusion
  _updateAnnotationModes();
}

function _updateAnnotationModes() {
  var hlBtn = document.getElementById('highlightToggleBtn');
  var cmBtn = document.getElementById('commentToggleBtn');
  var content = document.getElementById('notesContent');

  if (hlBtn) {
    hlBtn.classList.toggle('active', _highlightMode);
    hlBtn.title = _highlightMode ? 'Highlighting ON — select text' : 'Toggle highlighting';
  }
  if (cmBtn) {
    cmBtn.classList.toggle('active', _commentMode);
    cmBtn.title = _commentMode ? 'Comment mode ON — select text to comment' : 'Add comment';
  }
  if (content) {
    content.classList.toggle('sb-highlight-mode', _highlightMode);
    content.classList.toggle('sb-comment-mode', _commentMode);
  }
}

// ── Annotation persistence (backend) ─────────────

function _loadAnnotations() {
  if (!_currentNote) return;
  fetch('/api/prep/annotations/' + encodeURIComponent(_currentNote))
    .then(function(r) { return r.json(); })
    .then(function(data) {
      _annotations = data;
      // Set counter past existing IDs
      _commentIdCounter = 0;
      (data.comments || []).forEach(function(c) {
        var num = parseInt((c.id || '').replace('c', ''), 10);
        if (num >= _commentIdCounter) _commentIdCounter = num + 1;
      });
      _restoreHighlights();
      _restoreComments();
      _renderCommentPanel();
    })
    .catch(function() {
      // Fallback: try localStorage migration
      _migrateFromLocalStorage();
    });
}

function _migrateFromLocalStorage() {
  var key = 'highlights_' + (_currentNote || '').replace(/[^a-zA-Z0-9]/g, '_');
  var saved;
  try { saved = JSON.parse(localStorage.getItem(key) || '[]'); } catch(e) { return; }
  if (!saved.length) return;
  _annotations.highlights = saved;
  _restoreHighlights();
  _saveAnnotations();
  // Clean up old localStorage entry after migration
  localStorage.removeItem(key);
}

function _saveAnnotations() {
  if (!_currentNote) return;
  // Debounce saves to avoid hammering the server
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(function() {
    _collectHighlightsFromDOM();
    fetch('/api/prep/annotations/' + encodeURIComponent(_currentNote), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_annotations),
    }).catch(function(e) { console.warn('Failed to save annotations:', e); });
  }, 500);
}

function _collectHighlightsFromDOM() {
  var content = document.getElementById('notesContent');
  if (!content) return;
  var marks = content.querySelectorAll('mark.user-highlight');
  var highlights = [];
  marks.forEach(function(m) {
    highlights.push({
      text: m.textContent,
      context: _getHighlightContext(m),
    });
  });
  _annotations.highlights = highlights;
}

// ── Selection toolbar & highlight/comment creation ──

var _pendingSelectionRange = null;
var _pendingSelectionText = null;

document.addEventListener('mouseup', function (e) {
  if (!_currentNote) return;
  var content = document.getElementById('notesContent');
  if (!content || !content.contains(e.target)) return;

  // Don't interfere with toolbar clicks
  var toolbar = document.getElementById('selectionToolbar');
  if (toolbar && (e.target === toolbar || toolbar.contains(e.target))) return;

  var sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) {
    _hideSelectionToolbar();
    return;
  }
  var range = sel.getRangeAt(0);
  if (!content.contains(range.commonAncestorContainer)) {
    _hideSelectionToolbar();
    return;
  }

  var selectedText = sel.toString().trim();
  if (!selectedText) {
    _hideSelectionToolbar();
    return;
  }

  if (_highlightMode) {
    // Direct highlight in highlight mode
    if (range.commonAncestorContainer.closest && range.commonAncestorContainer.closest('mark.user-highlight')) return;
    try {
      var mark = document.createElement('mark');
      mark.className = 'user-highlight';
      mark.addEventListener('click', _onHighlightClick);
      range.surroundContents(mark);
      sel.removeAllRanges();
      _saveAnnotations();
    } catch (err) {
      console.warn('Could not highlight selection:', err.message);
    }
  } else if (_commentMode) {
    // Direct comment in comment mode
    if (range.commonAncestorContainer.closest && range.commonAncestorContainer.closest('mark.user-comment')) return;
    _showCommentPopup(range, selectedText);
  } else {
    // Normal mode: show selection toolbar
    _pendingSelectionRange = range.cloneRange();
    _pendingSelectionText = selectedText;
    _showSelectionToolbar(range, content);
  }
});

function _showSelectionToolbar(range, content) {
  var toolbar = document.getElementById('selectionToolbar');
  if (!toolbar) return;

  var rect = range.getBoundingClientRect();
  var mainEl = content.closest('.sb-notes-main');
  if (!mainEl) return;
  var mainRect = mainEl.getBoundingClientRect();

  // Position above the selection
  toolbar.classList.add('visible');
  var toolbarWidth = toolbar.offsetWidth;
  var left = rect.left - mainRect.left + (rect.width / 2) - (toolbarWidth / 2);
  // Clamp so it doesn't overflow
  left = Math.max(4, Math.min(left, mainRect.width - toolbarWidth - 4));
  toolbar.style.top = (rect.top - mainRect.top - toolbar.offsetHeight - 8) + 'px';
  toolbar.style.left = left + 'px';
}

function _hideSelectionToolbar() {
  var toolbar = document.getElementById('selectionToolbar');
  if (toolbar) toolbar.classList.remove('visible');
  _pendingSelectionRange = null;
  _pendingSelectionText = null;
}

function _selectionToolbarHighlight() {
  if (!_pendingSelectionRange || !_pendingSelectionText) return;
  var range = _pendingSelectionRange;
  // Don't double-highlight
  if (range.commonAncestorContainer.closest && range.commonAncestorContainer.closest('mark.user-highlight')) {
    _hideSelectionToolbar();
    return;
  }
  try {
    var mark = document.createElement('mark');
    mark.className = 'user-highlight';
    mark.addEventListener('click', _onHighlightClick);
    range.surroundContents(mark);
    window.getSelection().removeAllRanges();
    _saveAnnotations();
  } catch (err) {
    console.warn('Could not highlight selection:', err.message);
  }
  _hideSelectionToolbar();
}

function _selectionToolbarComment() {
  if (!_pendingSelectionRange || !_pendingSelectionText) return;
  var range = _pendingSelectionRange;
  var text = _pendingSelectionText;
  // Don't comment inside existing comment marks
  if (range.commonAncestorContainer.closest && range.commonAncestorContainer.closest('mark.user-comment')) {
    _hideSelectionToolbar();
    return;
  }
  _hideSelectionToolbar();
  _showCommentPopup(range, text);
}

// Hide toolbar on click elsewhere
document.addEventListener('mousedown', function (e) {
  var toolbar = document.getElementById('selectionToolbar');
  if (toolbar && !toolbar.contains(e.target)) {
    _hideSelectionToolbar();
  }
});

function _onHighlightClick(e) {
  if (!_highlightMode) return;
  var mark = e.currentTarget;
  var parent = mark.parentNode;
  while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
  parent.removeChild(mark);
  parent.normalize();
  _saveAnnotations();
}

function _getHighlightContext(mark) {
  var parent = mark.parentNode;
  if (!parent) return '';
  var fullText = parent.textContent || '';
  var markText = mark.textContent;
  var idx = fullText.indexOf(markText);
  if (idx === -1) return '';
  var before = fullText.substring(Math.max(0, idx - 30), idx);
  var after = fullText.substring(idx + markText.length, idx + markText.length + 30);
  return before + '|||' + markText + '|||' + after;
}

function _restoreHighlights() {
  var saved = _annotations.highlights || [];
  if (!saved.length) return;

  var content = document.getElementById('notesContent');
  var mdBody = content ? content.querySelector('.md-body') : null;
  if (!mdBody) return;

  saved.forEach(function(h) {
    if (!h.text) return;
    var walker = document.createTreeWalker(mdBody, NodeFilter.SHOW_TEXT, null, false);
    var node;
    while (node = walker.nextNode()) {
      var idx = node.textContent.indexOf(h.text);
      if (idx === -1) continue;
      if (h.context) {
        var parts = h.context.split('|||');
        if (parts.length === 3) {
          var parentText = node.parentNode.textContent || '';
          var foundIdx = parentText.indexOf(h.text);
          if (foundIdx === -1) continue;
          var beforeCtx = parentText.substring(Math.max(0, foundIdx - 30), foundIdx);
          if (parts[0] && !beforeCtx.includes(parts[0].slice(-10))) continue;
        }
      }
      try {
        var range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + h.text.length);
        var mark = document.createElement('mark');
        mark.className = 'user-highlight';
        mark.addEventListener('click', _onHighlightClick);
        range.surroundContents(mark);
      } catch (e) { /* skip partial matches */ }
      break;
    }
  });
}

function clearAllHighlights() {
  if (!_currentNote) return;
  var content = document.getElementById('notesContent');
  content.querySelectorAll('mark.user-highlight').forEach(function(mark) {
    var parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parent.normalize();
  });
  _annotations.highlights = [];
  _saveAnnotations();
}


/* ── Comments System ─────────────────────────────── */

function _showCommentPopup(range, selectedText) {
  // Remove any existing popup
  var existing = document.getElementById('commentPopup');
  if (existing) existing.remove();

  var rect = range.getBoundingClientRect();
  var popup = document.createElement('div');
  popup.id = 'commentPopup';
  popup.className = 'sb-comment-popup';
  popup.innerHTML =
    '<textarea class="sb-comment-popup__input" placeholder="Add a comment..." rows="3" autofocus></textarea>' +
    '<div class="sb-comment-popup__actions">' +
      '<button class="sb-comment-popup__cancel" onclick="document.getElementById(\'commentPopup\').remove()">Cancel</button>' +
      '<button class="sb-comment-popup__save">Save</button>' +
    '</div>';

  popup.style.top = (rect.bottom + window.scrollY + 8) + 'px';
  popup.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - 320)) + 'px';
  document.body.appendChild(popup);

  var textarea = popup.querySelector('textarea');
  textarea.focus();

  var saveBtn = popup.querySelector('.sb-comment-popup__save');
  saveBtn.addEventListener('click', function() {
    var commentText = textarea.value.trim();
    if (!commentText) return;
    _createComment(range, selectedText, commentText);
    popup.remove();
  });

  // Save on Ctrl+Enter
  textarea.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      saveBtn.click();
    } else if (e.key === 'Escape') {
      popup.remove();
    }
  });

  // Close popup when clicking outside
  setTimeout(function() {
    document.addEventListener('mousedown', function handler(e) {
      if (!popup.contains(e.target)) {
        popup.remove();
        document.removeEventListener('mousedown', handler);
      }
    });
  }, 100);
}

function _createComment(range, selectedText, commentText) {
  var id = 'c' + (_commentIdCounter++);
  var context = '';

  // Wrap selected text in a comment mark
  try {
    var mark = document.createElement('mark');
    mark.className = 'user-comment';
    mark.dataset.commentId = id;
    mark.title = commentText;
    mark.addEventListener('click', function() { _scrollToComment(id); });

    // Add comment badge
    var badge = document.createElement('span');
    badge.className = 'sb-comment-badge';
    badge.innerHTML = '<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
    badge.addEventListener('click', function(e) { e.stopPropagation(); _scrollToComment(id); });

    range.surroundContents(mark);
    mark.appendChild(badge);

    // Get context for restoration
    context = _getHighlightContext(mark);
  } catch (err) {
    console.warn('Could not create comment mark:', err.message);
    return;
  }

  window.getSelection().removeAllRanges();

  // Add to annotations
  _annotations.comments.push({
    id: id,
    text: selectedText,
    comment: commentText,
    context: context,
    timestamp: new Date().toISOString(),
  });

  _saveAnnotations();
  _renderCommentPanel();
  _updateCommentCount();

  // Show comment panel
  var panel = document.getElementById('commentPanel');
  if (panel && !panel.classList.contains('sb-comment-panel--open')) {
    toggleCommentPanel();
  }
}

function _restoreComments() {
  var comments = _annotations.comments || [];
  if (!comments.length) return;

  var content = document.getElementById('notesContent');
  var mdBody = content ? content.querySelector('.md-body') : null;
  if (!mdBody) return;

  comments.forEach(function(c) {
    if (!c.text) return;
    var walker = document.createTreeWalker(mdBody, NodeFilter.SHOW_TEXT, null, false);
    var node;
    while (node = walker.nextNode()) {
      var idx = node.textContent.indexOf(c.text);
      if (idx === -1) continue;

      // Verify context
      if (c.context) {
        var parts = c.context.split('|||');
        if (parts.length === 3) {
          var parentText = node.parentNode.textContent || '';
          var foundIdx = parentText.indexOf(c.text);
          if (foundIdx === -1) continue;
          var beforeCtx = parentText.substring(Math.max(0, foundIdx - 30), foundIdx);
          if (parts[0] && !beforeCtx.includes(parts[0].slice(-10))) continue;
        }
      }

      try {
        var range = document.createRange();
        range.setStart(node, idx);
        range.setEnd(node, idx + c.text.length);
        var mark = document.createElement('mark');
        mark.className = 'user-comment';
        mark.dataset.commentId = c.id;
        mark.title = c.comment;
        mark.addEventListener('click', function() { _scrollToComment(c.id); });

        var badge = document.createElement('span');
        badge.className = 'sb-comment-badge';
        badge.innerHTML = '<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
        badge.addEventListener('click', function(ev) { ev.stopPropagation(); _scrollToComment(c.id); });

        range.surroundContents(mark);
        mark.appendChild(badge);
      } catch (e) { /* skip */ }
      break;
    }
  });

  _updateCommentCount();
}

function _scrollToComment(commentId) {
  // Highlight the comment in the sidebar panel
  var panel = document.getElementById('commentPanel');
  if (panel && !panel.classList.contains('sb-comment-panel--open')) {
    toggleCommentPanel();
  }

  // Highlight in panel
  panel.querySelectorAll('.sb-comment-item').forEach(function(item) {
    item.classList.remove('sb-comment-item--active');
  });
  var panelItem = panel.querySelector('[data-comment-id="' + commentId + '"]');
  if (panelItem) {
    panelItem.classList.add('sb-comment-item--active');
    panelItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Scroll to the mark in the content
  var mark = document.querySelector('mark.user-comment[data-comment-id="' + commentId + '"]');
  if (mark) {
    mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
    mark.classList.add('sb-comment-flash');
    setTimeout(function() { mark.classList.remove('sb-comment-flash'); }, 1500);
  }
}

function _scrollToMark(commentId) {
  var mark = document.querySelector('mark.user-comment[data-comment-id="' + commentId + '"]');
  if (mark) {
    mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
    mark.classList.add('sb-comment-flash');
    setTimeout(function() { mark.classList.remove('sb-comment-flash'); }, 1500);
  }
}

function toggleCommentPanel() {
  var panel = document.getElementById('commentPanel');
  var layout = document.getElementById('notesLayout');
  if (!panel || !layout) return;

  var isOpen = panel.classList.contains('sb-comment-panel--open');
  panel.classList.toggle('sb-comment-panel--open', !isOpen);
  layout.classList.toggle('sb-comments-open', !isOpen);
}

function _renderCommentPanel() {
  var list = document.getElementById('commentList');
  if (!list) return;

  var comments = _annotations.comments || [];
  if (!comments.length) {
    list.innerHTML = '<div class="sb-comment-empty">No comments yet. Select text and use the comment tool to add one.</div>';
    return;
  }

  var html = '';
  comments.forEach(function(c) {
    var date = c.timestamp ? new Date(c.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '';
    html += '<div class="sb-comment-item" data-comment-id="' + c.id + '" onclick="_scrollToMark(\'' + c.id + '\')">' +
      '<div class="sb-comment-item__quote">"' + _escapeHtml(c.text.length > 80 ? c.text.substring(0, 80) + '...' : c.text) + '"</div>' +
      '<div class="sb-comment-item__text">' + _escapeHtml(c.comment) + '</div>' +
      '<div class="sb-comment-item__meta">' +
        '<span>' + date + '</span>' +
        '<button class="sb-comment-item__delete" onclick="event.stopPropagation(); _deleteComment(\'' + c.id + '\')" title="Delete comment">&times;</button>' +
      '</div>' +
    '</div>';
  });
  list.innerHTML = html;
}

function _deleteComment(commentId) {
  // Remove from DOM
  var mark = document.querySelector('mark.user-comment[data-comment-id="' + commentId + '"]');
  if (mark) {
    // Remove badge
    var badge = mark.querySelector('.sb-comment-badge');
    if (badge) badge.remove();
    // Unwrap mark
    var parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    parent.normalize();
  }

  // Remove from annotations
  _annotations.comments = (_annotations.comments || []).filter(function(c) { return c.id !== commentId; });
  _saveAnnotations();
  _renderCommentPanel();
  _updateCommentCount();
}

function _updateCommentCount() {
  var badge = document.getElementById('commentCountBadge');
  if (!badge) return;
  var count = (_annotations.comments || []).length;
  badge.textContent = count;
  badge.style.display = count > 0 ? '' : 'none';
}

function _escapeHtml(str) {
  var div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
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
