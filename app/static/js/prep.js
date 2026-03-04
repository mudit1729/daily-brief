/* ── Prep: Notes Viewer ────────────────────────── */

let _currentNote = null;

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
      // Scroll content to top
      content.scrollTop = 0;
    })
    .catch(err => {
      content.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
    });
}

/* ── Mobile back button ────────────────────────── */

function mobileBackToSidebar() {
  const layout = document.getElementById('notesLayout');
  layout.classList.remove('sb-notes-layout--viewing');
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


/* ── Note Title & Rename ─────────────────────────── */

function _getDisplayName(filename) {
  var name = filename.replace(/\.md$/i, '').replace(/-/g, ' ').replace(/_/g, ' ');
  // Strip known prefixes
  var prefixes = ['Ilya30 ', 'BEV ', 'Paper ', 'Async ', 'MLTheory ', 'MLPaper '];
  for (var i = 0; i < prefixes.length; i++) {
    if (name.startsWith(prefixes[i])) {
      name = name.substring(prefixes[i].length);
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

document.addEventListener('mouseup', function (e) {
  // Don't show "Use as context" in highlight mode or if chat isn't open
  if (_highlightMode) return;
  var panel = document.getElementById('chatPanel');
  if (!panel || !panel.classList.contains('sb-chat-panel--open')) return;

  var content = document.getElementById('notesContent');
  if (!content || !content.contains(e.target)) return;

  var sel = window.getSelection();
  var btn = document.getElementById('chatUseContextBtn');
  if (!sel || sel.isCollapsed || !sel.rangeCount || !btn) {
    btn.style.display = 'none';
    return;
  }

  var text = sel.toString().trim();
  if (text.length < 20) {
    btn.style.display = 'none';
    return;
  }

  // Position button near the selection
  var range = sel.getRangeAt(0);
  var rect = range.getBoundingClientRect();
  var mainRect = content.closest('.sb-notes-main').getBoundingClientRect();

  btn.style.display = '';
  btn.style.top = (rect.bottom - mainRect.top + 4) + 'px';
  btn.style.left = (rect.left - mainRect.left + rect.width / 2 - 60) + 'px';
});

// Hide context button on click elsewhere
document.addEventListener('mousedown', function (e) {
  var btn = document.getElementById('chatUseContextBtn');
  if (btn && e.target !== btn && !btn.contains(e.target)) {
    btn.style.display = 'none';
  }
});

function setChatSectionFromSelection() {
  var sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  var text = sel.toString().trim();
  if (!text) return;

  _selectedSection = text;
  sel.removeAllRanges();

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
