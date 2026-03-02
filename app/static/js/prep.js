/* ── Prep: Notes Viewer ────────────────────────── */

let _currentNote = null;

function loadNote(filename, btn) {
  // Mark active
  document.querySelectorAll('.sb-notes-list__item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');

  _currentNote = filename;
  const content = document.getElementById('notesContent');
  const toolbar = document.getElementById('notesToolbar');
  content.innerHTML = '<div class="sb-notes-empty"><p>Loading...</p></div>';
  toolbar.style.display = '';

  fetch('/api/prep/notes/' + encodeURIComponent(filename))
    .then(r => {
      if (!r.ok) throw new Error('Failed to load note');
      return r.json();
    })
    .then(data => {
      content.innerHTML = '<div class="md-body">' + data.html + '</div>';
      // Apply syntax highlighting to all code blocks
      content.querySelectorAll('pre code').forEach(block => {
        if (typeof hljs !== 'undefined') hljs.highlightElement(block);
      });
      // Restore highlights from localStorage
      _restoreHighlights();
    })
    .catch(err => {
      content.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
    });
}

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
