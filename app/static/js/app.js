/**
 * Pulse Engine — Client-side progressive enhancement
 * No dependencies. Vanilla JS. ES2020+.
 */
(function () {
  'use strict';

  // ────────────────────────────────────────────────
  // 0. Welcome Note
  // ────────────────────────────────────────────────
  const welcomeNote = document.getElementById('welcomeNote');
  const dismissBtn = document.getElementById('dismissWelcome');
  if (welcomeNote && !localStorage.getItem('pulse-welcome-dismissed')) {
    welcomeNote.style.display = 'flex';
  }
  if (dismissBtn) {
    dismissBtn.addEventListener('click', () => {
      localStorage.setItem('pulse-welcome-dismissed', '1');
      welcomeNote.style.display = 'none';
    });
  }

  // ────────────────────────────────────────────────
  // 1. Theme Toggle
  // ────────────────────────────────────────────────
  const html = document.documentElement;
  const themeBtn = document.getElementById('themeToggle');
  const stored = localStorage.getItem('sb-theme');

  if (stored) {
    html.setAttribute('data-bs-theme', stored);
  }

  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const next = html.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-bs-theme', next);
      localStorage.setItem('sb-theme', next);
    });
  }

  // ────────────────────────────────────────────────
  // 2. Card Expand / Collapse
  // ────────────────────────────────────────────────
  const cards = document.querySelectorAll('.sb-card');

  // Cards start expanded — clicking header collapses them
  cards.forEach((card) => {
    const header = card.querySelector('.sb-card__header');
    if (header) {
      header.addEventListener('click', (e) => {
        if (e.target.closest('a')) return;
        card.classList.toggle('sb-card--collapsed');
      });
    }
  });

  // ────────────────────────────────────────────────
  // 3. Keyboard Navigation
  // ────────────────────────────────────────────────
  let focusIndex = -1;

  function getCards() {
    return document.querySelectorAll('.sb-card');
  }

  function focusCard(index) {
    const allCards = getCards();
    if (index < 0 || index >= allCards.length) return;
    focusIndex = index;
    allCards[focusIndex].focus();
    allCards[focusIndex].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  document.addEventListener('keydown', (e) => {
    // Skip if in input/textarea
    if (e.target.matches('input, textarea, select')) return;

    // Show keyboard hints
    document.body.classList.add('sb-keyboard-user');

    const allCards = getCards();
    const active = document.activeElement;
    const isCard = active && active.classList.contains('sb-card');

    switch (e.key) {
      case 'j':
        e.preventDefault();
        if (focusIndex < 0) focusIndex = -1;
        focusCard(Math.min(focusIndex + 1, allCards.length - 1));
        break;

      case 'k':
        e.preventDefault();
        focusCard(Math.max(focusIndex - 1, 0));
        break;

      case 'Enter':
        if (isCard) {
          e.preventDefault();
          active.classList.toggle('sb-card--collapsed');
        }
        break;

      case 'f':
        if (isCard) {
          e.preventDefault();
          const btn = active.querySelector('[data-action="follow"]');
          if (btn) btn.click();
        }
        break;

      case 'm':
        if (isCard) {
          e.preventDefault();
          const btn = active.querySelector('[data-action="mute"]');
          if (btn) btn.click();
        }
        break;

      case 'u':
        if (isCard) {
          e.preventDefault();
          const btn = active.querySelector('[data-action="upvote"]');
          if (btn) btn.click();
        }
        break;

      case 'd':
        if (isCard) {
          e.preventDefault();
          const btn = active.querySelector('[data-action="downvote"]');
          if (btn) btn.click();
        }
        break;

      case 'Escape':
        if (isCard) {
          active.classList.add('sb-card--collapsed');
        }
        break;

      case '?':
        // Show keyboard shortcuts info
        break;
    }
  });

  // Hide keyboard hints on mouse use
  document.addEventListener('mousedown', () => {
    document.body.classList.remove('sb-keyboard-user');
  });

  // ────────────────────────────────────────────────
  // 4. Feedback Actions
  // ────────────────────────────────────────────────
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.sb-action-btn[data-action]');
    if (!btn) return;

    e.preventDefault();
    const action = btn.dataset.action;
    const targetType = btn.dataset.targetType;
    const targetId = btn.dataset.targetId;

    if (!action || !targetType || !targetId) return;

    // Toggle active state
    btn.classList.toggle('is-active');

    // POST to API
    fetch('/api/feedback/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action_type: action,
        target_type: targetType,
        target_id: targetId,
      }),
    })
      .then((r) => {
        if (!r.ok) {
          btn.classList.toggle('is-active');
        }
      })
      .catch(() => {
        btn.classList.toggle('is-active');
      });
  });

  // ────────────────────────────────────────────────
  // 5. Section Nav Scroll Spy
  // ────────────────────────────────────────────────
  const sectionNavItems = document.querySelectorAll('.sb-section-nav__item');
  const sections = document.querySelectorAll('.sb-section[id]');

  if (sections.length > 0 && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            sectionNavItems.forEach((item) => {
              item.classList.toggle('is-active', item.getAttribute('href') === '#' + id);
            });
          }
        });
      },
      { rootMargin: '-20% 0px -60% 0px' }
    );

    sections.forEach((s) => observer.observe(s));
  }

  // ────────────────────────────────────────────────
  // 6. Settings Page — Tab Switching
  // ────────────────────────────────────────────────
  const tabLinks = document.querySelectorAll('.sb-settings-tabs__item');
  const tabPanels = document.querySelectorAll('.sb-tab-panel');

  tabLinks.forEach((link) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const tab = link.dataset.tab;

      tabLinks.forEach((l) => l.classList.remove('is-active'));
      link.classList.add('is-active');

      tabPanels.forEach((p) => {
        p.style.display = p.id === 'panel-' + tab ? 'block' : 'none';
      });
    });
  });

  // ────────────────────────────────────────────────
  // 7. Settings Admin Controls
  // ────────────────────────────────────────────────
  const ADMIN_KEY_STORAGE = 'sb-admin-key';
  const adminKeyInput = document.getElementById('adminApiKey');
  const saveAdminKeyBtn = document.getElementById('saveAdminKey');
  const adminKeyStatus = document.getElementById('adminKeyStatus');
  const triggerBtn = document.getElementById('triggerPipeline');
  const pipelineStatus = document.getElementById('pipelineStatus');
  const scheduleEnabled = document.getElementById('scheduleEnabled');
  const scheduleTime = document.getElementById('scheduleTime');
  const scheduleTimezone = document.getElementById('scheduleTimezone');
  const saveScheduleBtn = document.getElementById('savePipelineSchedule');
  const scheduleStatus = document.getElementById('scheduleStatus');
  const scheduleNextRun = document.getElementById('scheduleNextRun');
  const refreshSourceHealthBtn = document.getElementById('refreshSourceHealth');
  const sourceHealthStatus = document.getElementById('sourceHealthStatus');

  function setStatus(el, message, isError = false) {
    if (!el) return;
    el.textContent = message || '';
    el.style.color = isError ? 'var(--sb-negative)' : 'var(--sb-text-muted)';
  }

  function currentAdminKey() {
    const typed = adminKeyInput?.value?.trim();
    if (typed) return typed;
    const stored = localStorage.getItem(ADMIN_KEY_STORAGE);
    return (stored || '').trim();
  }

  function hasAdminKey() {
    return !!currentAdminKey();
  }

  function adminHeaders(withJson = false) {
    const headers = {};
    if (withJson) headers['Content-Type'] = 'application/json';
    const key = currentAdminKey();
    if (key) headers['X-Admin-Key'] = key;
    return headers;
  }

  function formatScheduleTime(hour, minute) {
    const hh = String(hour).padStart(2, '0');
    const mm = String(minute).padStart(2, '0');
    return hh + ':' + mm;
  }

  function updateScheduleUI(data) {
    if (!data) return;
    if (scheduleEnabled) scheduleEnabled.checked = !!data.enabled;
    if (scheduleTime) scheduleTime.value = formatScheduleTime(data.hour || 0, data.minute || 0);
    if (scheduleTimezone) scheduleTimezone.value = data.timezone || 'UTC';
    if (scheduleNextRun) {
      if (data.next_run_at) {
        const next = new Date(data.next_run_at);
        scheduleNextRun.textContent = 'Next run: ' + next.toLocaleString();
      } else {
        scheduleNextRun.textContent = 'Next run: not scheduled';
      }
    }
  }

  function updateHealthSummary(data) {
    if (!data || !data.summary) return;
    const healthy = document.getElementById('sourceHealthHealthy');
    const degraded = document.getElementById('sourceHealthDegraded');
    const cooldown = document.getElementById('sourceHealthCooldown');
    if (healthy) healthy.textContent = data.summary.healthy ?? 0;
    if (degraded) degraded.textContent = data.summary.degraded ?? 0;
    if (cooldown) cooldown.textContent = data.summary.cooldown ?? 0;
  }

  async function loadPipelineSchedule() {
    if (!scheduleEnabled && !scheduleTime && !scheduleTimezone) return;
    if (!hasAdminKey()) {
      setStatus(scheduleStatus, 'Enter admin key to load schedule');
      return;
    }

    setStatus(scheduleStatus, 'Loading schedule...');
    try {
      const response = await fetch('/api/admin/schedule/pipeline', {
        method: 'GET',
        headers: adminHeaders(),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || 'Unable to load schedule');
      updateScheduleUI(data);
      setStatus(scheduleStatus, 'Schedule loaded');
    } catch (err) {
      setStatus(scheduleStatus, err.message || 'Unable to load schedule', true);
    }
  }

  async function loadSourceHealth() {
    if (!refreshSourceHealthBtn && !sourceHealthStatus) return;
    if (!hasAdminKey()) {
      setStatus(sourceHealthStatus, 'Enter admin key to refresh');
      return;
    }

    setStatus(sourceHealthStatus, 'Refreshing...');
    try {
      const response = await fetch('/api/admin/sources/health', {
        method: 'GET',
        headers: adminHeaders(),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || 'Unable to fetch source health');
      updateHealthSummary(data);
      setStatus(sourceHealthStatus, 'Updated');
    } catch (err) {
      setStatus(sourceHealthStatus, err.message || 'Unable to fetch source health', true);
    }
  }

  if (adminKeyInput) {
    const storedKey = localStorage.getItem(ADMIN_KEY_STORAGE);
    if (storedKey) {
      adminKeyInput.value = storedKey;
      setStatus(adminKeyStatus, 'Admin key loaded');
    }
  }

  if (saveAdminKeyBtn) {
    saveAdminKeyBtn.addEventListener('click', () => {
      const key = adminKeyInput?.value?.trim();
      if (!key) {
        localStorage.removeItem(ADMIN_KEY_STORAGE);
        setStatus(adminKeyStatus, 'Key cleared');
        return;
      }
      localStorage.setItem(ADMIN_KEY_STORAGE, key);
      setStatus(adminKeyStatus, 'Key saved');
      loadPipelineSchedule();
      loadSourceHealth();
    });
  }

  if (triggerBtn) {
    triggerBtn.addEventListener('click', async () => {
      if (!hasAdminKey()) {
        setStatus(pipelineStatus, 'Enter admin key first', true);
        return;
      }
      triggerBtn.disabled = true;
      setStatus(pipelineStatus, 'Triggering...');

      try {
        const hdrs = adminHeaders();
        hdrs['Content-Type'] = 'application/json';
        const response = await fetch('/api/admin/pipeline/trigger', {
          method: 'POST',
          headers: hdrs,
          body: JSON.stringify({ force: true }),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'Failed to trigger');
        setStatus(pipelineStatus, 'Pipeline triggered for ' + data.date);
      } catch (err) {
        setStatus(pipelineStatus, err.message || 'Failed to trigger', true);
      } finally {
        setTimeout(() => {
          triggerBtn.disabled = false;
        }, 2000);
      }
    });
  }

  if (saveScheduleBtn) {
    saveScheduleBtn.addEventListener('click', async () => {
      if (!hasAdminKey()) {
        setStatus(scheduleStatus, 'Enter admin key first', true);
        return;
      }

      const payload = {
        enabled: !!scheduleEnabled?.checked,
        time: scheduleTime?.value || '05:30',
        timezone: (scheduleTimezone?.value || 'UTC').trim(),
      };

      setStatus(scheduleStatus, 'Saving...');
      try {
        const response = await fetch('/api/admin/schedule/pipeline', {
          method: 'PUT',
          headers: adminHeaders(true),
          body: JSON.stringify(payload),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || 'Failed to save schedule');
        updateScheduleUI(data);
        setStatus(scheduleStatus, 'Schedule saved');
      } catch (err) {
        setStatus(scheduleStatus, err.message || 'Failed to save schedule', true);
      }
    });
  }

  if (refreshSourceHealthBtn) {
    refreshSourceHealthBtn.addEventListener('click', () => {
      loadSourceHealth();
    });
  }

  if (hasAdminKey()) {
    loadPipelineSchedule();
    loadSourceHealth();
  } else {
    setStatus(scheduleStatus, 'Enter admin key to load schedule');
    setStatus(sourceHealthStatus, 'Enter admin key to refresh');
  }

  // ────────────────────────────────────────────────
  // 8. Insight Submission
  // ────────────────────────────────────────────────
  const insightForm = document.getElementById('insightForm');
  const insightInput = document.getElementById('insightText');

  if (insightForm) {
    insightForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const text = insightInput?.value?.trim();
      if (!text) return;

      fetch('/api/feedback/insight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
        .then((r) => r.json())
        .then(() => {
          insightInput.value = '';
          // Simple: reload to show new insight
          window.location.reload();
        })
        .catch(() => {
          // Silently fail
        });
    });
  }

  // ────────────────────────────────────────────────
  // 9. Insight Promotion
  // ────────────────────────────────────────────────
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-promote-id]');
    if (!btn) return;

    e.preventDefault();
    const id = btn.dataset.promoteId;

    fetch('/api/feedback/insight/' + id + '/promote', { method: 'POST' })
      .then((r) => {
        if (r.ok) {
          btn.textContent = 'Promoted';
          btn.disabled = true;
          btn.classList.add('is-active');
        }
      })
      .catch(() => {});
  });

  // ────────────────────────────────────────────────
  // 10. Card Stack Navigation
  // ────────────────────────────────────────────────
  const MAX_VISIBLE_BEHIND = 3; // how many queued cards peek behind the active one

  function initCardStacks() {
    document.querySelectorAll('.sb-card-stack').forEach((stack) => {
      const items = stack.querySelectorAll('.sb-card-stack__item');
      if (items.length === 0) return;

      // Set container height from the tallest card (measure the first one)
      const measure = () => {
        // Temporarily show all items at full size to measure
        const activeItem = stack.querySelector('.sb-card-stack__item.is-active');
        if (activeItem) {
          // Use the active card's height + padding for queued peek
          const h = activeItem.offsetHeight;
          stack.style.height = (h + MAX_VISIBLE_BEHIND * 12 + 8) + 'px';
        }
      };

      // Measure after images load
      const imgs = stack.querySelectorAll('img');
      let loaded = 0;
      const onLoad = () => {
        loaded++;
        if (loaded >= imgs.length) measure();
      };
      imgs.forEach((img) => {
        if (img.complete) { loaded++; } else { img.addEventListener('load', onLoad); img.addEventListener('error', onLoad); }
      });
      if (loaded >= imgs.length) measure();

      // Also measure after a short delay (fallback)
      setTimeout(measure, 300);
      setTimeout(measure, 1000);

      // Apply initial states
      updateStack(stack, 0);
    });
  }

  function updateStack(stack, currentIndex) {
    const items = Array.from(stack.querySelectorAll('.sb-card-stack__item'));
    const total = items.length;
    stack.dataset.current = currentIndex;

    // Update each item's class
    items.forEach((item, i) => {
      item.classList.remove('is-active', 'is-queued', 'is-hidden', 'is-swiped');

      if (i < currentIndex) {
        // Already swiped away
        item.classList.add('is-swiped');
      } else if (i === currentIndex) {
        // Current active card
        item.classList.add('is-active');
      } else if (i <= currentIndex + MAX_VISIBLE_BEHIND) {
        // Visible behind active
        item.classList.add('is-queued');
      } else {
        // Off-screen
        item.classList.add('is-hidden');
      }
    });

    // Update counter and button states in the parent section
    const section = stack.closest('.sb-section');
    if (section) {
      const counterCurrent = section.querySelector('.sb-stack-counter__current');
      const prevBtn = section.querySelector('.sb-stack-btn--prev');
      const nextBtn = section.querySelector('.sb-stack-btn--next');
      const resetBtn = section.querySelector('.sb-stack-btn--reset');

      if (counterCurrent) counterCurrent.textContent = currentIndex + 1;
      if (prevBtn) prevBtn.disabled = currentIndex === 0;
      if (nextBtn) nextBtn.disabled = currentIndex >= total - 1;
      if (resetBtn) resetBtn.disabled = currentIndex === 0;
    }

    // Re-measure height based on the active card
    requestAnimationFrame(() => {
      const activeItem = items[currentIndex];
      if (activeItem) {
        const h = activeItem.offsetHeight;
        stack.style.height = (h + MAX_VISIBLE_BEHIND * 12 + 8) + 'px';
      }
    });
  }

  // Button click handlers (delegated)
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('.sb-stack-btn');
    if (!btn || btn.disabled) return;

    e.preventDefault();
    const section = btn.closest('.sb-section');
    const stack = section?.querySelector('.sb-card-stack');
    if (!stack) return;

    const items = stack.querySelectorAll('.sb-card-stack__item');
    const current = parseInt(stack.dataset.current, 10) || 0;

    if (btn.classList.contains('sb-stack-btn--next')) {
      if (current < items.length - 1) {
        updateStack(stack, current + 1);
      }
    } else if (btn.classList.contains('sb-stack-btn--prev')) {
      if (current > 0) {
        updateStack(stack, current - 1);
      }
    } else if (btn.classList.contains('sb-stack-btn--reset')) {
      updateStack(stack, 0);
    }
  });

  // Initialize all stacks on the page
  initCardStacks();

  // ────────────────────────────────────────────────
  // 11. Story Accordion Toggle
  // ────────────────────────────────────────────────
  document.addEventListener('click', (e) => {
    const toggle = e.target.closest('[data-story-toggle]');
    if (!toggle) return;

    e.preventDefault();
    const content = toggle.nextElementSibling;
    if (!content || !content.hasAttribute('data-story-content')) return;

    const isOpen = toggle.classList.contains('is-open');
    toggle.classList.toggle('is-open', !isOpen);
    content.classList.toggle('is-open', !isOpen);
  });

  // ────────────────────────────────────────────────
  // 12. Time Travel (rolling date wheel)
  // ────────────────────────────────────────────────
  const ttWheel = document.querySelector('[data-time-travel-wheel]');
  let ttSelectedDate = null;
  let ttInitialized = false;

  const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

  function ttIsoDate(d) {
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  }

  function ttInitWheel() {
    if (!ttWheel || ttInitialized) return;
    ttInitialized = true;

    const baseDate = ttWheel.dataset.baseDate;
    if (!baseDate) return;

    // Generate 31 date cells (oldest on left → today on right)
    const frag = document.createDocumentFragment();
    // Add spacers so first/last cells can reach center
    const spacerL = document.createElement('div');
    spacerL.style.minWidth = '50%'; spacerL.style.flexShrink = '0';
    frag.appendChild(spacerL);

    for (let i = 30; i >= 0; i--) {
      const d = new Date(baseDate + 'T00:00:00');
      d.setDate(d.getDate() - i);

      const cell = document.createElement('div');
      cell.className = 'sb-time-travel__date-cell' + (i === 0 ? ' is-today is-selected' : '');
      cell.dataset.date = ttIsoDate(d);
      cell.dataset.daysAgo = i;

      const dayLabel = document.createElement('div');
      dayLabel.className = 'sb-time-travel__date-day';
      dayLabel.textContent = i === 0 ? 'Today' : DAYS[d.getDay()];

      const numLabel = document.createElement('div');
      numLabel.className = 'sb-time-travel__date-num';
      numLabel.textContent = MONTHS[d.getMonth()] + ' ' + d.getDate();

      cell.appendChild(dayLabel);
      cell.appendChild(numLabel);
      frag.appendChild(cell);
    }

    const spacerR = document.createElement('div');
    spacerR.style.minWidth = '50%'; spacerR.style.flexShrink = '0';
    frag.appendChild(spacerR);

    ttWheel.appendChild(frag);
    ttSelectedDate = baseDate;

    // Scroll "Today" (last real cell) to center after render — horizontal only
    requestAnimationFrame(() => {
      const todayCell = ttWheel.querySelector('.is-today');
      if (todayCell) {
        const wheelRect = ttWheel.getBoundingClientRect();
        const cellRect = todayCell.getBoundingClientRect();
        const offset = cellRect.left - wheelRect.left - (wheelRect.width / 2) + (cellRect.width / 2);
        ttWheel.scrollLeft += offset;
      }
    });

    // Detect centered cell on scroll
    let scrollTimer = null;
    ttWheel.addEventListener('scroll', () => {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(() => ttUpdateSelection(), 80);
    });

    // Click to select — horizontal scroll only (no page jump)
    ttWheel.addEventListener('click', (e) => {
      const cell = e.target.closest('.sb-time-travel__date-cell');
      if (cell) {
        const wheelRect = ttWheel.getBoundingClientRect();
        const cellRect = cell.getBoundingClientRect();
        const offset = cellRect.left - wheelRect.left - (wheelRect.width / 2) + (cellRect.width / 2);
        ttWheel.scrollTo({ left: ttWheel.scrollLeft + offset, behavior: 'smooth' });
        setTimeout(() => ttUpdateSelection(), 350);
      }
    });
  }

  function ttUpdateSelection() {
    if (!ttWheel) return;
    const cells = ttWheel.querySelectorAll('.sb-time-travel__date-cell');
    const wrapperRect = ttWheel.parentElement.getBoundingClientRect();
    const centerX = wrapperRect.left + wrapperRect.width / 2;

    let closest = null;
    let closestDist = Infinity;
    cells.forEach(cell => {
      const rect = cell.getBoundingClientRect();
      const cellCenter = rect.left + rect.width / 2;
      const dist = Math.abs(cellCenter - centerX);
      if (dist < closestDist) {
        closestDist = dist;
        closest = cell;
      }
    });

    if (closest) {
      cells.forEach(c => c.classList.remove('is-selected'));
      closest.classList.add('is-selected');
      ttSelectedDate = closest.dataset.date;
    }
  }

  document.addEventListener('click', (e) => {
    const toggleBtn = e.target.closest('[data-time-travel-toggle]');
    if (toggleBtn) {
      const panel = document.querySelector('[data-time-travel-panel]');
      if (panel) {
        panel.classList.toggle('is-open');
        if (panel.classList.contains('is-open')) ttInitWheel();
      }
      return;
    }

    const goBtn = e.target.closest('[data-time-travel-go]');
    if (goBtn) {
      if (ttSelectedDate) {
        window.location.href = '/brief/' + ttSelectedDate;
      }
      return;
    }
  });

  // ────────────────────────────────────────────────
  // 13. Live Clocks (multi-timezone)
  // ────────────────────────────────────────────────
  function updateClocks() {
    document.querySelectorAll('.sb-clock[data-tz]').forEach(el => {
      const tz = el.dataset.tz;
      const label = el.dataset.label || '';
      try {
        const time = new Date().toLocaleTimeString('en-US', {
          timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: true
        });
        el.textContent = label + ' ' + time;
      } catch (e) { /* skip invalid tz */ }
    });
  }
  updateClocks();
  setInterval(updateClocks, 30000); // update every 30s

  // ────────────────────────────────────────────────
  // 14. Deep Dive Modal
  // ────────────────────────────────────────────────
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-deep-dive]');
    if (!btn) return;

    e.preventDefault();
    e.stopPropagation();

    const clusterId = btn.dataset.deepDive;
    const card = btn.closest('.sb-card');
    const title = card ? card.querySelector('.sb-card__title')?.textContent?.trim() : 'Story';

    // Create modal
    const overlay = document.createElement('div');
    overlay.className = 'sb-deep-dive-overlay';
    overlay.innerHTML = `
      <div class="sb-deep-dive-modal">
        <button class="sb-deep-dive-modal__close">&times;</button>
        <div class="sb-deep-dive-modal__title">${title}</div>
        <div class="sb-deep-dive-modal__content">
          <div class="sb-deep-dive-modal__loading">
            <div class="sb-deep-dive-spinner"></div>
            <span>Generating deep dive analysis...</span>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    // Close on click outside or close button
    overlay.addEventListener('click', (ev) => {
      if (ev.target === overlay || ev.target.closest('.sb-deep-dive-modal__close')) {
        overlay.remove();
      }
    });

    // Close on Escape
    const escHandler = (ev) => {
      if (ev.key === 'Escape') {
        overlay.remove();
        document.removeEventListener('keydown', escHandler);
      }
    };
    document.addEventListener('keydown', escHandler);

    // Fetch deep dive
    fetch(`/api/deep-dive/${clusterId}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        const contentEl = overlay.querySelector('.sb-deep-dive-modal__content');
        if (data.error) {
          contentEl.innerHTML = `<p style="color:var(--sb-danger);">Error: ${data.error}</p>`;
        } else {
          // Simple markdown-ish rendering
          let html = data.content
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/^### (.+)$/gm, '<h4 style="margin-top:1em;margin-bottom:0.3em;">$1</h4>')
            .replace(/^## (.+)$/gm, '<h3 style="margin-top:1em;margin-bottom:0.3em;">$1</h3>')
            .replace(/^- (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
            .replace(/<\/ul>\s*<ul>/g, '')
            .replace(/\n\n/g, '<br><br>');

          const providerBadge = data.provider === 'xai'
            ? '<span class="sb-grok-take__label" style="margin-left:var(--sb-space-2);">Grok</span>'
            : '<span class="sb-badge" style="margin-left:var(--sb-space-2);">OpenAI</span>';

          contentEl.innerHTML = `
            <div style="margin-bottom:var(--sb-space-3);font-size:var(--sb-text-xs);color:var(--sb-text-muted);">
              Powered by ${providerBadge}
              <span style="margin-left:var(--sb-space-2);">${data.tokens} tokens</span>
            </div>
            <div style="font-size:var(--sb-text-sm);line-height:1.8;color:var(--sb-text-secondary);">${html}</div>
          `;
        }
      })
      .catch(err => {
        const contentEl = overlay.querySelector('.sb-deep-dive-modal__content');
        contentEl.innerHTML = `<p style="color:var(--sb-danger);">Failed to load analysis. Please try again.</p>`;
      });
  });

})();
