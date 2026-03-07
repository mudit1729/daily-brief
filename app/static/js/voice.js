/* ── Voice Coach: Sargam Keyboard + Pitch Detector ── */

// ── Swara definitions (C# = Sa) ──
const SWARAS = [
  { name: 'Sa',  freq: 277.18 },
  { name: 'Re',  freq: 311.13 },
  { name: 'Ga',  freq: 349.23 },
  { name: 'Ma',  freq: 369.99 },
  { name: 'Pa',  freq: 415.30 },
  { name: 'Dha', freq: 466.16 },
  { name: 'Ni',  freq: 523.25 },
  { name: "Sa'", freq: 554.37 },
];

// Also include lower and higher octave for detection range
const ALL_SWARAS = [
  // Lower octave
  ...SWARAS.map(s => ({ name: s.name === "Sa'" ? "Sa'" : s.name, freq: s.freq / 2, octave: 3 })),
  // Main octave
  ...SWARAS.map(s => ({ name: s.name, freq: s.freq, octave: 4 })),
  // Higher octave
  ...SWARAS.map(s => ({ name: s.name === "Sa'" ? "Sa'" : s.name, freq: s.freq * 2, octave: 5 })),
];

// ── Audio context (lazy init) ──
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

// ── Sargam Keyboard ──
const activeOscillators = new Map();

function playSwara(freq, key) {
  const ctx = getAudioCtx();
  if (ctx.state === 'suspended') ctx.resume();

  // Stop existing on this key
  stopSwara(key);

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = 'sine';
  osc.frequency.setValueAtTime(freq, ctx.currentTime);

  // Smooth envelope
  gain.gain.setValueAtTime(0, ctx.currentTime);
  gain.gain.linearRampToValueAtTime(0.3, ctx.currentTime + 0.05);

  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();

  activeOscillators.set(key, { osc, gain });
  key.classList.add('playing');
}

function stopSwara(key) {
  const entry = activeOscillators.get(key);
  if (!entry) return;

  const { osc, gain } = entry;
  const ctx = getAudioCtx();
  gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.1);
  setTimeout(() => {
    try { osc.stop(); } catch {}
    try { osc.disconnect(); gain.disconnect(); } catch {}
  }, 150);

  activeOscillators.delete(key);
  key.classList.remove('playing');
}

// Set up keyboard buttons
document.querySelectorAll('.vc-key').forEach(key => {
  const freq = parseFloat(key.dataset.freq);

  // Mouse
  key.addEventListener('mousedown', (e) => { e.preventDefault(); playSwara(freq, key); });
  key.addEventListener('mouseup', () => stopSwara(key));
  key.addEventListener('mouseleave', () => stopSwara(key));

  // Touch
  key.addEventListener('touchstart', (e) => { e.preventDefault(); playSwara(freq, key); });
  key.addEventListener('touchend', () => stopSwara(key));
  key.addEventListener('touchcancel', () => stopSwara(key));
});

// Keyboard shortcuts: 1-8 for swaras
const keyButtons = document.querySelectorAll('.vc-key');
const keyMap = { '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7 };
const heldKeys = new Set();

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.repeat) return;
  const idx = keyMap[e.key];
  if (idx !== undefined && keyButtons[idx]) {
    heldKeys.add(e.key);
    playSwara(parseFloat(keyButtons[idx].dataset.freq), keyButtons[idx]);
  }
});

document.addEventListener('keyup', (e) => {
  const idx = keyMap[e.key];
  if (idx !== undefined && keyButtons[idx]) {
    heldKeys.delete(e.key);
    stopSwara(keyButtons[idx]);
  }
});

// ── Pitch Detector ──
let micStream = null;
let analyser = null;
let micSource = null;
let detecting = false;
let animFrameId = null;

const BUFFER_SIZE = 2048;
const MIN_FREQ = 100;  // Hz — ignore below
const MAX_FREQ = 1200; // Hz — ignore above
const IN_TUNE_CENTS = 10;

// DOM refs
const detectedSwara = document.getElementById('detectedSwara');
const detectedFreq = document.getElementById('detectedFreq');
const detectedCents = document.getElementById('detectedCents');
const centsNeedle = document.getElementById('centsNeedle');
const volumeFill = document.getElementById('volumeFill');
const micBtn = document.getElementById('micBtn');
const micLabel = document.getElementById('micLabel');
const micOnIcon = document.getElementById('micOnIcon');
const micOffIcon = document.getElementById('micOffIcon');

window.toggleMic = async function() {
  if (detecting) {
    stopDetecting();
  } else {
    await startDetecting();
  }
};

async function startDetecting() {
  try {
    const ctx = getAudioCtx();
    if (ctx.state === 'suspended') await ctx.resume();

    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micSource = ctx.createMediaStreamSource(micStream);

    analyser = ctx.createAnalyser();
    analyser.fftSize = BUFFER_SIZE * 2;
    micSource.connect(analyser);

    detecting = true;
    micBtn.classList.add('active');
    micLabel.textContent = 'Stop Mic';
    micOnIcon.style.display = 'none';
    micOffIcon.style.display = '';

    detectLoop();
  } catch (err) {
    console.error('Mic access error:', err);
    detectedSwara.textContent = 'Mic blocked';
    detectedFreq.textContent = 'Allow microphone access';
  }
}

function stopDetecting() {
  detecting = false;
  if (animFrameId) cancelAnimationFrame(animFrameId);

  if (micStream) {
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
  }
  if (micSource) {
    try { micSource.disconnect(); } catch {}
    micSource = null;
  }
  analyser = null;

  micBtn.classList.remove('active');
  micLabel.textContent = 'Start Mic';
  micOnIcon.style.display = '';
  micOffIcon.style.display = 'none';

  // Reset display
  detectedSwara.textContent = '--';
  detectedSwara.classList.remove('vc-in-tune');
  detectedFreq.textContent = '-- Hz';
  detectedCents.textContent = '';
  centsNeedle.style.transform = 'rotate(0deg)';
  centsNeedle.classList.remove('vc-in-tune');
  volumeFill.style.width = '0%';
}

function detectLoop() {
  if (!detecting || !analyser) return;

  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);

  // Volume (RMS)
  let rms = 0;
  for (let i = 0; i < buf.length; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / buf.length);
  const volumePct = Math.min(100, rms * 500);
  volumeFill.style.width = volumePct + '%';

  // Only detect if there's enough signal
  if (rms > 0.01) {
    const freq = autocorrelate(buf, audioCtx.sampleRate);
    if (freq > MIN_FREQ && freq < MAX_FREQ) {
      updateDisplay(freq);
    }
  } else {
    // Quiet — fade display
    detectedSwara.textContent = '--';
    detectedSwara.classList.remove('vc-in-tune');
    detectedFreq.textContent = '-- Hz';
    detectedCents.textContent = '';
    centsNeedle.style.transform = 'rotate(0deg)';
    centsNeedle.classList.remove('vc-in-tune');
  }

  animFrameId = requestAnimationFrame(detectLoop);
}

// ── Autocorrelation pitch detection ──
function autocorrelate(buf, sampleRate) {
  const n = buf.length;

  // Find the first zero-crossing to skip the initial transient
  let r1 = 0;
  for (let i = 0; i < n / 2; i++) {
    if (Math.abs(buf[i]) < 0.2) { r1 = i; break; }
  }

  // Autocorrelation
  const minLag = Math.floor(sampleRate / MAX_FREQ);
  const maxLag = Math.floor(sampleRate / MIN_FREQ);

  let bestCorr = -1;
  let bestLag = -1;

  for (let lag = minLag; lag < maxLag && lag < n / 2; lag++) {
    let corr = 0;
    for (let i = 0; i < n / 2; i++) {
      corr += buf[i] * buf[i + lag];
    }
    if (corr > bestCorr) {
      bestCorr = corr;
      bestLag = lag;
    }
  }

  if (bestLag === -1) return -1;

  // Parabolic interpolation for sub-sample accuracy
  let corrPrev = 0, corrNext = 0;
  for (let i = 0; i < n / 2; i++) {
    if (bestLag > 0) corrPrev += buf[i] * buf[i + bestLag - 1];
    if (bestLag + 1 < n / 2) corrNext += buf[i] * buf[i + bestLag + 1];
  }

  const shift = (corrPrev - corrNext) / (2 * (corrPrev - 2 * bestCorr + corrNext));
  const refinedLag = bestLag + (isFinite(shift) ? shift : 0);

  return sampleRate / refinedLag;
}

// ── Map frequency to nearest swara ──
function findNearestSwara(freq) {
  let bestDist = Infinity;
  let bestSwara = null;
  let bestRef = 0;

  for (const s of ALL_SWARAS) {
    const cents = 1200 * Math.log2(freq / s.freq);
    const dist = Math.abs(cents);
    if (dist < bestDist) {
      bestDist = dist;
      bestSwara = s;
      bestRef = s.freq;
    }
  }

  const cents = 1200 * Math.log2(freq / bestRef);
  return { swara: bestSwara, cents };
}

function updateDisplay(freq) {
  const { swara, cents } = findNearestSwara(freq);
  if (!swara) return;

  const inTune = Math.abs(cents) < IN_TUNE_CENTS;

  // Swara name
  detectedSwara.textContent = swara.name;
  detectedSwara.classList.toggle('vc-in-tune', inTune);

  // Frequency
  detectedFreq.textContent = freq.toFixed(1) + ' Hz';

  // Cents
  const sign = cents >= 0 ? '+' : '';
  detectedCents.textContent = sign + cents.toFixed(0) + ' cents';

  // Needle rotation: -50 cents = -90°, 0 = 0°, +50 cents = +90°
  const clampedCents = Math.max(-50, Math.min(50, cents));
  const angle = (clampedCents / 50) * 90;
  centsNeedle.style.transform = `rotate(${angle}deg)`;
  centsNeedle.classList.toggle('vc-in-tune', inTune);

  // Highlight matching keyboard key
  document.querySelectorAll('.vc-key').forEach(k => k.classList.remove('vc-detected'));
  const matchKey = document.querySelector(`.vc-key[data-swara="${swara.name}"]`);
  if (matchKey && Math.abs(cents) < 30) {
    matchKey.classList.add('vc-detected');
  }
}

// ── Theme toggle ──
function syncVcTheme() {
  const t = localStorage.getItem('sb-theme');
  const dark = !t || t === 'dark';
  document.getElementById('vcSunIcon').style.display = dark ? 'none' : '';
  document.getElementById('vcMoonIcon').style.display = dark ? '' : 'none';
}
syncVcTheme();

window.toggleVcTheme = function() {
  const html = document.documentElement;
  const next = html.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-bs-theme', next);
  localStorage.setItem('sb-theme', next);
  syncVcTheme();
};

// ── Fullscreen ──
let vcFullscreen = false;
window.toggleVcFullscreen = function() {
  vcFullscreen = !vcFullscreen;
  const layout = document.getElementById('voiceLayout');
  layout.classList.toggle('vc-fullscreen', vcFullscreen);
  document.getElementById('vcExpandIcon').style.display = vcFullscreen ? 'none' : '';
  document.getElementById('vcCollapseIcon').style.display = vcFullscreen ? '' : 'none';

  const nav = document.querySelector('.sb-nav');
  const footer = document.querySelector('.sb-footer');
  if (nav) nav.style.display = vcFullscreen ? 'none' : '';
  if (footer) footer.style.display = vcFullscreen ? 'none' : '';
};
