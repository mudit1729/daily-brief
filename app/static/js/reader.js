/* ── PDF Reader ────────────────────────────────── */

import * as pdfjsLib from 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.min.mjs';

pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.4.168/pdf.worker.min.mjs';

// ── State ──
let pdfDoc = null;
let currentScale = 1.5;
let currentFile = null;
let totalPages = 0;
let renderedPages = new Map();   // pageNum -> { canvas, textLayer, rendered }
let highlightMode = false;
let observer = null;             // IntersectionObserver for lazy rendering
let isFullscreen = false;

const BASE_SCALE = 1.5;
const ZOOM_STEP = 0.25;
const MIN_SCALE = 0.5;
const MAX_SCALE = 4;

// ── DOM refs ──
const viewer = document.getElementById('pdfViewer');
const pagesContainer = document.getElementById('pdfPages');
const pageInput = document.getElementById('pageInput');
const pageCountEl = document.getElementById('pageCount');
const zoomLabel = document.getElementById('zoomLabel');
const emptyEl = document.getElementById('readerEmpty');
const loadingEl = document.getElementById('readerLoading');
const titleEl = document.getElementById('pdfTitle');
const layout = document.getElementById('readerLayout');

// ── Reader sidebar toggle ──
window.toggleReaderSidebar = function() {
  layout.classList.toggle('rd-sidebar-hidden');
};

// ── Sidebar section toggle ──
window.toggleSection = function(btn) {
  const section = btn.closest('.rd-course') || btn.closest('.rd-section');
  section.classList.toggle('collapsed');
};

// ── Theme sync ──
function syncReaderTheme() {
  const t = localStorage.getItem('sb-theme');
  const dark = !t || t === 'dark';
  document.getElementById('rdSunIcon').style.display = dark ? 'none' : '';
  document.getElementById('rdMoonIcon').style.display = dark ? '' : 'none';
}
syncReaderTheme();

window.toggleReaderTheme = function() {
  const html = document.documentElement;
  const cur = html.getAttribute('data-bs-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-bs-theme', next);
  localStorage.setItem('sb-theme', next);
  syncReaderTheme();
};

// ── Load PDF ──
window.loadPDF = async function(filename, btn) {
  // Mark active in sidebar
  document.querySelectorAll('.rd-book-item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');

  currentFile = filename;
  titleEl.textContent = filename.replace('.pdf', '').replace(/[-_]/g, ' ');

  // Mobile: switch to viewing mode
  if (window.innerWidth <= 640) {
    layout.classList.add('rd-viewing');
  }

  // Show loading
  emptyEl.style.display = 'none';
  pagesContainer.style.display = 'none';
  loadingEl.style.display = '';
  loadingEl.innerHTML = '<div class="rd-spinner"></div><p>Loading PDF...</p>';

  // Cleanup previous
  cleanup();

  try {
    const url = '/api/reader/file/' + encodeURIComponent(filename);
    await openPDF({ url, enableXfa: false });
  } catch (err) {
    loadingEl.innerHTML = '<p style="color:var(--sb-text-muted)">Failed to load PDF</p>';
    console.error('PDF load error:', err);
  }
};

// ── Handle local file upload ──
window.handleLocalFile = async function(input) {
  const file = input.files[0];
  if (!file) return;

  // Add to sidebar
  const displayName = file.name.replace('.pdf', '').replace(/[-_]/g, ' ');
  const li = document.createElement('li');
  const btn = document.createElement('button');
  btn.className = 'rd-book-item';
  btn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    <span>${displayName}</span>`;
  li.appendChild(btn);
  document.getElementById('bookList').appendChild(li);

  // Mark active
  document.querySelectorAll('.rd-book-item').forEach(el => el.classList.remove('active'));
  btn.classList.add('active');

  // Store blob URL for re-opening
  const blobUrl = URL.createObjectURL(file);
  btn.onclick = () => loadLocalBlobPDF(blobUrl, file.name, btn);

  currentFile = 'local:' + file.name;
  titleEl.textContent = displayName;

  if (window.innerWidth <= 640) {
    layout.classList.add('rd-viewing');
  }

  emptyEl.style.display = 'none';
  pagesContainer.style.display = 'none';
  loadingEl.style.display = '';
  loadingEl.innerHTML = '<div class="rd-spinner"></div><p>Loading PDF...</p>';

  cleanup();

  try {
    const data = await file.arrayBuffer();
    await openPDF({ data, enableXfa: false });
  } catch (err) {
    loadingEl.innerHTML = '<p style="color:var(--sb-text-muted)">Failed to load PDF</p>';
    console.error('Local PDF load error:', err);
  }

  // Reset input so same file can be reopened
  input.value = '';
};

// Re-open a local blob
async function loadLocalBlobPDF(blobUrl, filename, btn) {
  document.querySelectorAll('.rd-book-item').forEach(el => el.classList.remove('active'));
  btn.classList.add('active');

  currentFile = 'local:' + filename;
  titleEl.textContent = filename.replace('.pdf', '').replace(/[-_]/g, ' ');

  if (window.innerWidth <= 640) {
    layout.classList.add('rd-viewing');
  }

  emptyEl.style.display = 'none';
  pagesContainer.style.display = 'none';
  loadingEl.style.display = '';
  loadingEl.innerHTML = '<div class="rd-spinner"></div><p>Loading PDF...</p>';

  cleanup();

  try {
    await openPDF({ url: blobUrl, enableXfa: false });
  } catch (err) {
    loadingEl.innerHTML = '<p style="color:var(--sb-text-muted)">Failed to load PDF</p>';
    console.error('PDF load error:', err);
  }
}

// ── Load Google Drive PDF ──
window.loadGDrivePDF = async function(fileId, displayName, btn) {
  document.querySelectorAll('.rd-book-item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');

  currentFile = 'gdrive:' + fileId;
  titleEl.textContent = displayName;

  if (window.innerWidth <= 640) {
    layout.classList.add('rd-viewing');
  }

  emptyEl.style.display = 'none';
  pagesContainer.style.display = 'none';
  loadingEl.style.display = '';
  loadingEl.innerHTML = `
    <div class="rd-spinner"></div>
    <p class="rd-dl-status">Downloading from Google Drive...</p>
    <div class="rd-dl-progress" style="display:none">
      <div class="rd-dl-progress__bar"><div class="rd-dl-progress__fill" id="dlFill"></div></div>
      <span class="rd-dl-progress__text" id="dlText"></span>
    </div>`;

  cleanup();

  try {
    const url = '/api/reader/gdrive/' + encodeURIComponent(fileId);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('Download failed: ' + resp.status);

    const contentLength = resp.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength) : 0;

    // Show progress bar if we know the total size
    const progressEl = loadingEl.querySelector('.rd-dl-progress');
    const fillEl = document.getElementById('dlFill');
    const textEl = document.getElementById('dlText');
    if (total > 0) progressEl.style.display = '';

    // Stream-read the response body to track progress
    const reader = resp.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;

      if (total > 0) {
        const pct = Math.min(100, (received / total) * 100);
        fillEl.style.width = pct.toFixed(1) + '%';
        textEl.textContent = _formatBytes(received) + ' / ' + _formatBytes(total);
      } else {
        textEl.textContent = _formatBytes(received);
        if (progressEl.style.display === 'none') progressEl.style.display = '';
      }
    }

    // Combine chunks into one ArrayBuffer
    const combined = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    loadingEl.querySelector('.rd-dl-status').textContent = 'Rendering PDF...';
    await openPDF({ data: combined.buffer, enableXfa: false });
  } catch (err) {
    loadingEl.innerHTML = '<p style="color:var(--sb-text-muted)">Failed to download PDF</p>';
    console.error('GDrive PDF load error:', err);
  }
};

function _formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(0) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Shared PDF open logic ──
async function openPDF(source) {
  pdfDoc = await pdfjsLib.getDocument(source).promise;
  totalPages = pdfDoc.numPages;
  pageCountEl.textContent = totalPages;
  pageInput.value = 1;
  pageInput.max = totalPages;

  loadingEl.style.display = 'none';
  pagesContainer.style.display = '';

  createPagePlaceholders();
  setupObserver();
  await fitWidthScale();
  updateZoomLabel();
}

function cleanup() {
  if (observer) observer.disconnect();
  if (unloadObserver) unloadObserver.disconnect();
  renderedPages.clear();
  pagesContainer.innerHTML = '';
  pdfDoc = null;
}

// ── Page placeholders ──
function createPagePlaceholders() {
  for (let i = 1; i <= totalPages; i++) {
    const div = document.createElement('div');
    div.className = 'rd-page';
    div.dataset.page = i;
    // Estimate page size (will be corrected on render)
    div.style.width = '612px';
    div.style.height = '792px';
    pagesContainer.appendChild(div);
    renderedPages.set(i, { el: div, rendered: false });
  }
}

// ── Lazy rendering via IntersectionObserver ──
// We use TWO observers:
// 1. renderObserver: tight margin — renders pages near viewport
// 2. unloadObserver: wide margin — unloads pages far from viewport to free canvas memory
let unloadObserver = null;
const MAX_RENDERED = 10;  // Max pages to keep rendered at once

function setupObserver() {
  // Render observer: render pages when they approach the viewport
  observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const pageNum = parseInt(entry.target.dataset.page);
        const info = renderedPages.get(pageNum);
        if (info && !info.rendered && !info.rendering) {
          renderPage(pageNum);
          // After rendering, check if we need to unload distant pages
          trimRenderedPages(pageNum);
        }
      }
    }
  }, {
    root: viewer,
    rootMargin: '600px 0px',  // Pre-render 600px above/below viewport
    threshold: 0
  });

  renderedPages.forEach((info) => {
    observer.observe(info.el);
  });
}

function trimRenderedPages(currentPage) {
  // Count rendered pages
  const rendered = [];
  renderedPages.forEach((info, num) => {
    if (info.rendered) rendered.push(num);
  });

  if (rendered.length <= MAX_RENDERED) return;

  // Sort by distance from current page, unload the farthest
  rendered.sort((a, b) => Math.abs(a - currentPage) - Math.abs(b - currentPage));
  const toUnload = rendered.slice(MAX_RENDERED);

  for (const pageNum of toUnload) {
    unloadPage(pageNum);
  }
}

function unloadPage(pageNum) {
  const info = renderedPages.get(pageNum);
  if (!info || !info.rendered) return;

  // Preserve the placeholder dimensions before clearing
  const w = info.el.style.width;
  const h = info.el.style.height;
  info.el.innerHTML = '';
  info.el.style.width = w;
  info.el.style.height = h;
  info.rendered = false;
  info.rendering = false;
  info.canvas = null;
  info.textDiv = null;
  info.viewport = null;
}

// ── Render a single page ──
async function renderPage(pageNum) {
  const info = renderedPages.get(pageNum);
  if (!info || info.rendered || info.rendering) return;
  info.rendering = true;

  try {
    const page = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: currentScale });

    // Size the container
    info.el.style.width = viewport.width + 'px';
    info.el.style.height = viewport.height + 'px';

    // Canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);  // Cap DPR to save canvas memory
    canvas.width = viewport.width * dpr;
    canvas.height = viewport.height * dpr;
    canvas.style.width = viewport.width + 'px';
    canvas.style.height = viewport.height + 'px';
    ctx.scale(dpr, dpr);
    info.el.appendChild(canvas);

    await page.render({ canvasContext: ctx, viewport }).promise;

    // Text layer
    const textContent = await page.getTextContent();
    const textDiv = document.createElement('div');
    textDiv.className = 'textLayer';
    textDiv.style.width = viewport.width + 'px';
    textDiv.style.height = viewport.height + 'px';
    info.el.appendChild(textDiv);

    renderTextLayer(textContent, textDiv, viewport);

    // Restore highlights
    restoreHighlights(pageNum, textDiv);

    info.rendered = true;
    info.rendering = false;
    info.canvas = canvas;
    info.textDiv = textDiv;
    info.viewport = viewport;
  } catch (err) {
    info.rendering = false;
    console.error('Render error page', pageNum, err);
  }
}

// ── Text layer rendering ──
function renderTextLayer(textContent, container, viewport) {
  const items = textContent.items;
  for (const item of items) {
    if (!item.str) continue;
    const span = document.createElement('span');
    span.textContent = item.str;

    const tx = pdfjsLib.Util.transform(viewport.transform, item.transform);
    const [a, b, c, d, e, f] = tx;

    const angle = Math.atan2(b, a);
    const scaleX = Math.hypot(a, b);
    const scaleY = Math.hypot(c, d);

    span.style.left = e + 'px';
    span.style.top = (f - item.height * currentScale) + 'px';
    span.style.fontSize = (scaleY) + 'px';
    span.style.fontFamily = item.fontName || 'sans-serif';

    if (Math.abs(angle) > 0.01) {
      span.style.transform = `rotate(${angle}rad)`;
    }
    if (Math.abs(scaleX - scaleY) > 0.1) {
      span.style.transform = (span.style.transform || '') + ` scaleX(${(scaleX / scaleY).toFixed(3)})`;
    }

    container.appendChild(span);
  }

  // Highlighting: mouseup on text layer
  container.addEventListener('mouseup', () => {
    if (!highlightMode) return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;

    const range = sel.getRangeAt(0);
    const spans = container.querySelectorAll('span');
    let started = false;

    spans.forEach(span => {
      if (range.intersectsNode(span)) {
        span.classList.add('rd-highlight');
        started = true;
      }
    });

    sel.removeAllRanges();
    if (started) saveHighlights();
  });

  // Right-click to remove highlight
  container.addEventListener('contextmenu', (e) => {
    const span = e.target.closest('span.rd-highlight');
    if (span) {
      e.preventDefault();
      span.classList.remove('rd-highlight');
      saveHighlights();
    }
  });
}

// ── Highlights persistence ──
function highlightKey() {
  return 'rd-highlights-' + (currentFile || '');
}

function saveHighlights() {
  if (!currentFile) return;
  const data = {};
  renderedPages.forEach((info, pageNum) => {
    if (!info.textDiv) return;
    const indices = [];
    const spans = info.textDiv.querySelectorAll('span');
    spans.forEach((span, i) => {
      if (span.classList.contains('rd-highlight')) indices.push(i);
    });
    if (indices.length) data[pageNum] = indices;
  });
  localStorage.setItem(highlightKey(), JSON.stringify(data));
}

function restoreHighlights(pageNum, textDiv) {
  if (!currentFile) return;
  try {
    const raw = localStorage.getItem(highlightKey());
    if (!raw) return;
    const data = JSON.parse(raw);
    const indices = data[pageNum];
    if (!indices) return;
    const spans = textDiv.querySelectorAll('span');
    indices.forEach(i => {
      if (spans[i]) spans[i].classList.add('rd-highlight');
    });
  } catch { /* ignore */ }
}

// ── Navigation ──
window.prevPage = function() {
  const cur = parseInt(pageInput.value) || 1;
  if (cur > 1) goToPage(cur - 1);
};

window.nextPage = function() {
  const cur = parseInt(pageInput.value) || 1;
  if (cur < totalPages) goToPage(cur + 1);
};

window.goToPage = function(num) {
  num = Math.max(1, Math.min(totalPages, parseInt(num) || 1));
  pageInput.value = num;
  const info = renderedPages.get(num);
  if (info && info.el) {
    info.el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
};

// Track current page on scroll
viewer.addEventListener('scroll', () => {
  if (!pdfDoc) return;
  const viewerRect = viewer.getBoundingClientRect();
  const viewerMid = viewerRect.top + viewerRect.height / 3;

  for (let i = 1; i <= totalPages; i++) {
    const info = renderedPages.get(i);
    if (!info) continue;
    const rect = info.el.getBoundingClientRect();
    if (rect.top <= viewerMid && rect.bottom > viewerMid) {
      pageInput.value = i;
      break;
    }
  }
});

// ── Zoom ──
window.zoomIn = function() {
  setScale(currentScale + ZOOM_STEP);
};

window.zoomOut = function() {
  setScale(currentScale - ZOOM_STEP);
};

window.fitWidth = async function() {
  await fitWidthScale();
  updateZoomLabel();
};

async function fitWidthScale() {
  if (!pdfDoc) return;
  const page = await pdfDoc.getPage(1);
  const vp = page.getViewport({ scale: 1 });
  const availWidth = viewer.clientWidth - 32; // padding
  const newScale = availWidth / vp.width;
  setScale(Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale)));
}

function setScale(newScale) {
  newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));
  if (Math.abs(newScale - currentScale) < 0.01) return;
  currentScale = newScale;
  updateZoomLabel();
  reRenderAll();
}

function updateZoomLabel() {
  zoomLabel.textContent = Math.round((currentScale / BASE_SCALE) * 100) + '%';
}

async function reRenderAll() {
  // Re-render all currently rendered pages at new scale
  const toRerender = [];
  renderedPages.forEach((info, pageNum) => {
    if (info.rendered) {
      info.rendered = false;
      info.rendering = false;
      info.el.innerHTML = '';
      toRerender.push(pageNum);
    }
  });

  // Update placeholder sizes
  if (pdfDoc) {
    const page = await pdfDoc.getPage(1);
    const vp = page.getViewport({ scale: currentScale });
    renderedPages.forEach((info) => {
      if (!info.rendered) {
        info.el.style.width = vp.width + 'px';
        info.el.style.height = vp.height + 'px';
      }
    });
  }

  // Re-observe for lazy rendering
  if (observer) observer.disconnect();
  setupObserver();
}

// ── Highlight mode ──
window.toggleHighlightMode = function() {
  highlightMode = !highlightMode;
  document.getElementById('highlightToggle').classList.toggle('active', highlightMode);
  layout.classList.toggle('rd-highlight-mode', highlightMode);
};

// ── Fullscreen ──
window.toggleFullscreen = function() {
  isFullscreen = !isFullscreen;
  layout.classList.toggle('rd-fullscreen', isFullscreen);
  document.getElementById('fsExpandIcon').style.display = isFullscreen ? 'none' : '';
  document.getElementById('fsCollapseIcon').style.display = isFullscreen ? '' : 'none';

  // Hide nav and footer
  const nav = document.querySelector('.sb-nav');
  const footer = document.querySelector('.sb-footer');
  if (nav) nav.style.display = isFullscreen ? 'none' : '';
  if (footer) footer.style.display = isFullscreen ? 'none' : '';

  // Re-fit width after layout change
  if (pdfDoc) {
    setTimeout(() => fitWidthScale().then(updateZoomLabel), 50);
  }
};

// ── Mobile back ──
window.mobileBackToSidebar = function() {
  layout.classList.remove('rd-viewing');
};

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {
  if (!pdfDoc) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  switch(e.key) {
    case 'ArrowLeft':
    case 'ArrowUp':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); prevPage(); }
      break;
    case 'ArrowRight':
    case 'ArrowDown':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); nextPage(); }
      break;
    case '+':
    case '=':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); zoomIn(); }
      break;
    case '-':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); zoomOut(); }
      break;
    case '0':
      if (e.ctrlKey || e.metaKey) { e.preventDefault(); fitWidth(); }
      break;
    case 'f':
    case 'F':
      if (e.ctrlKey || e.metaKey) break; // Don't override browser find
      if (e.shiftKey) { e.preventDefault(); toggleFullscreen(); }
      break;
    case 'h':
      if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); toggleHighlightMode(); }
      break;
    case 'Escape':
      if (isFullscreen) { e.preventDefault(); toggleFullscreen(); }
      break;
  }
});

// ── Drag & drop PDF onto viewer ──
viewer.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.stopPropagation();
  viewer.classList.add('rd-dragover');
});

viewer.addEventListener('dragleave', (e) => {
  e.preventDefault();
  e.stopPropagation();
  viewer.classList.remove('rd-dragover');
});

viewer.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  viewer.classList.remove('rd-dragover');

  const file = e.dataTransfer.files[0];
  if (file && file.type === 'application/pdf') {
    // Reuse the local file handler by simulating a file input
    const input = document.getElementById('localPdfInput');
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    handleLocalFile(input);
  }
});

// ── Auto-load first PDF ──
document.addEventListener('DOMContentLoaded', () => {
  const first = document.querySelector('.rd-book-item');
  if (first) {
    loadPDF(first.dataset.file, first);
  }
});
