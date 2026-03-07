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

  // Cleanup previous
  cleanup();

  try {
    const url = '/api/reader/file/' + encodeURIComponent(filename);
    pdfDoc = await pdfjsLib.getDocument({ url, enableXfa: false }).promise;
    totalPages = pdfDoc.numPages;
    pageCountEl.textContent = totalPages;
    pageInput.value = 1;
    pageInput.max = totalPages;

    loadingEl.style.display = 'none';
    pagesContainer.style.display = '';

    // Create placeholders for all pages
    createPagePlaceholders();

    // Set up lazy rendering
    setupObserver();

    // Auto fit-width on load
    await fitWidthScale();

    updateZoomLabel();
  } catch (err) {
    loadingEl.innerHTML = '<p style="color:var(--sb-text-muted)">Failed to load PDF</p>';
    console.error('PDF load error:', err);
  }
};

function cleanup() {
  if (observer) observer.disconnect();
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
function setupObserver() {
  observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const pageNum = parseInt(entry.target.dataset.page);
        const info = renderedPages.get(pageNum);
        if (info && !info.rendered && !info.rendering) {
          renderPage(pageNum);
        }
      }
    }
  }, {
    root: viewer,
    rootMargin: '200px 0px',  // Pre-render 200px above/below viewport
    threshold: 0
  });

  renderedPages.forEach((info) => {
    observer.observe(info.el);
  });
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
    const dpr = window.devicePixelRatio || 1;
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

// ── Auto-load first PDF ──
document.addEventListener('DOMContentLoaded', () => {
  const first = document.querySelector('.rd-book-item');
  if (first) {
    loadPDF(first.dataset.file, first);
  }
});
