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
  // 7. Pipeline Trigger
  // ────────────────────────────────────────────────
  const triggerBtn = document.getElementById('triggerPipeline');
  const pipelineStatus = document.getElementById('pipelineStatus');

  if (triggerBtn) {
    triggerBtn.addEventListener('click', () => {
      triggerBtn.disabled = true;
      if (pipelineStatus) pipelineStatus.textContent = 'Triggering...';

      fetch('/api/admin/pipeline/trigger', { method: 'POST' })
        .then((r) => r.json())
        .then((data) => {
          if (pipelineStatus) pipelineStatus.textContent = 'Pipeline triggered for ' + data.date;
          setTimeout(() => {
            triggerBtn.disabled = false;
          }, 5000);
        })
        .catch(() => {
          if (pipelineStatus) pipelineStatus.textContent = 'Failed to trigger';
          triggerBtn.disabled = false;
        });
    });
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
})();
