/**
 * Signal Brief Engine — Client-side progressive enhancement
 * No dependencies. Vanilla JS. ES2020+.
 */
(function () {
  'use strict';

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
        const response = await fetch('/api/admin/pipeline/trigger', {
          method: 'POST',
          headers: adminHeaders(),
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

})();
