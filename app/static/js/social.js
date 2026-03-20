/* ── Social Media Follows ─────────────────────── */
(function () {
  'use strict';

  // ── State ──
  let timelineOffset = document.querySelectorAll('.sb-social-timeline-item').length;
  let currentView = 'top';
  let activeViewBeforeDetail = 'top';

  // ── Helpers ──

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || data.message || 'Request failed');
    return data;
  }

  function formatDate(iso) {
    if (!iso) return '';
    return iso.slice(0, 10);
  }

  function showToast(msg) {
    let toast = document.getElementById('sbSocialToast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'sbSocialToast';
      toast.style.cssText =
        'position:fixed;bottom:2rem;right:2rem;background:var(--sb-surface,' +
        '#1e293b);color:var(--sb-text,#f1f5f9);padding:.75rem 1.25rem;' +
        'border-radius:.5rem;font-size:.875rem;z-index:9999;opacity:0;' +
        'transition:opacity .3s ease;pointer-events:none;';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 3000);
  }

  // ── Markdown + Mermaid rendering ──

  function renderMarkdown(container, md) {
    container.innerHTML = marked.parse(md || '');
    const mermaidBlocks = container.querySelectorAll('pre > code.language-mermaid');
    const mermaidNodes = [];
    mermaidBlocks.forEach((code) => {
      const pre = code.parentElement;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      pre.replaceWith(div);
      mermaidNodes.push(div);
    });
    if (mermaidNodes.length > 0) {
      mermaid.run({ nodes: mermaidNodes });
    }
  }

  // ── Post card HTML ──

  function renderPostCard(post) {
    const date = formatDate(post.published_at);
    const platform = (post.platform || '').toUpperCase();
    const hint = (post.content_hint || '').slice(0, 200);
    let summaryHtml = '';
    if (post.has_summary) {
      const escaped = (post.summary_md || '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      summaryHtml =
        `<button class="sb-social-timeline-item__toggle" onclick="toggleSummary(this)">Show Summary</button>` +
        `<div class="sb-social-timeline-item__summary" style="display:none;" data-md="${escaped}"></div>`;
    } else if (hint) {
      summaryHtml = `<p class="sb-social-timeline-item__hint">${hint}</p>`;
    }

    return `
      <div class="sb-social-timeline-item" data-post-id="${post.id}">
        <div class="sb-social-timeline-item__meta">
          <span class="sb-platform-badge sb-platform--${post.platform}">${platform}</span>
          <span class="sb-social-timeline-item__channel">${post.channel_name || ''}</span>
          <time class="sb-social-timeline-item__date">${date}</time>
        </div>
        <h3 class="sb-social-timeline-item__title">
          <a href="${post.url || '#'}" target="_blank" rel="noopener">${post.title || 'Untitled'}</a>
        </h3>
        ${summaryHtml}
      </div>`;
  }

  // ── Follow / Unfollow ──

  window.followChannel = async function () {
    const input = document.getElementById('followUrl');
    const url = (input.value || '').trim();
    if (!url) { alert('Please enter a feed URL.'); return; }

    try {
      await api('/api/social/follow', {
        method: 'POST',
        body: JSON.stringify({ feed_url: url }),
      });
      input.value = '';
      location.reload();
    } catch (err) {
      alert('Error following channel: ' + err.message);
    }
  };

  window.unfollowChannel = async function (channelId, channelName) {
    if (!confirm('Unfollow ' + channelName + '?')) return;

    try {
      await api('/api/social/unfollow/' + channelId, { method: 'POST' });
      const card = document.querySelector(
        `.sb-social-channel-card[onclick*="showChannel(${channelId})"]`
      );
      if (card) {
        card.style.transition = 'opacity .3s ease, transform .3s ease';
        card.style.opacity = '0';
        card.style.transform = 'scale(0.95)';
        setTimeout(() => card.remove(), 300);
      } else {
        location.reload();
      }
    } catch (err) {
      alert('Error unfollowing: ' + err.message);
    }
  };

  // ── View switching ──

  window.switchView = function (view, btn) {
    currentView = view;
    document.querySelectorAll('.sb-social-view').forEach((el) => {
      el.style.display = 'none';
    });

    const viewMap = { top: 'viewTop', all: 'viewAll', timeline: 'viewTimeline' };
    const target = document.getElementById(viewMap[view]);
    if (target) target.style.display = '';

    document.querySelectorAll('.sb-social-tab').forEach((t) => t.classList.remove('active'));
    if (btn) btn.classList.add('active');

    // Hide channel detail overlay if visible
    const detail = document.getElementById('channelDetail');
    if (detail) detail.style.display = 'none';
  };

  // ── Channel detail overlay ──

  window.showChannel = async function (channelId) {
    activeViewBeforeDetail = currentView;
    const detail = document.getElementById('channelDetail');
    const postsContainer = document.getElementById('channelDetailPosts');
    const nameEl = document.getElementById('channelDetailName');
    const platformEl = document.getElementById('channelDetailPlatform');

    // Hide all views, show overlay
    document.querySelectorAll('.sb-social-view').forEach((el) => {
      el.style.display = 'none';
    });
    detail.style.display = '';
    postsContainer.innerHTML = '<p style="color:var(--sb-text-muted,#94a3b8)">Loading...</p>';

    try {
      const posts = await api('/api/social/channel/' + channelId + '/posts');
      if (posts.length > 0) {
        nameEl.textContent = posts[0].channel_name || 'Channel';
        platformEl.textContent = (posts[0].platform || '').toUpperCase();
        platformEl.className = 'sb-platform-badge sb-platform--' + (posts[0].platform || '');
      } else {
        nameEl.textContent = 'Channel';
        platformEl.textContent = '';
      }
      postsContainer.innerHTML = posts.length
        ? posts.map(renderPostCard).join('')
        : '<p style="color:var(--sb-text-muted,#94a3b8)">No posts yet.</p>';
    } catch (err) {
      postsContainer.innerHTML = '<p style="color:#ef4444">Failed to load posts.</p>';
    }
  };

  window.closeChannelDetail = function () {
    document.getElementById('channelDetail').style.display = 'none';
    const activeBtn = document.querySelector(`.sb-social-tab[data-view="${activeViewBeforeDetail}"]`);
    switchView(activeViewBeforeDetail, activeBtn);
  };

  // ── All Channels split view ──

  window.showChannelPosts = async function (channelId, btnEl) {
    // Mark sidebar item active
    document.querySelectorAll('.sb-social-channel-sidebar__item').forEach((el) => {
      el.classList.remove('active');
    });
    if (btnEl) btnEl.classList.add('active');

    const postList = document.getElementById('postList');
    postList.innerHTML = '<p style="color:var(--sb-text-muted,#94a3b8)">Loading...</p>';

    try {
      const posts = await api('/api/social/channel/' + channelId + '/posts');
      postList.innerHTML = posts.length
        ? posts.map(renderPostCard).join('')
        : '<p style="color:var(--sb-text-muted,#94a3b8)">No posts for this channel.</p>';
    } catch (err) {
      postList.innerHTML = '<p style="color:#ef4444">Failed to load posts.</p>';
    }
  };

  // ── Toggle summary ──

  window.toggleSummary = function (btn) {
    const summaryDiv = btn.nextElementSibling;
    if (!summaryDiv) return;

    const isHidden = summaryDiv.style.display === 'none';
    summaryDiv.style.display = isHidden ? '' : 'none';
    btn.textContent = isHidden ? 'Hide Summary' : 'Show Summary';

    if (isHidden && !summaryDiv.dataset.rendered) {
      const md = summaryDiv.dataset.md || '';
      renderMarkdown(summaryDiv, md);
      summaryDiv.dataset.rendered = '1';
    }
  };

  // ── Load more timeline posts ──

  window.loadMorePosts = async function () {
    const btn = document.getElementById('loadMoreBtn');
    if (btn) btn.textContent = 'Loading...';

    try {
      const posts = await api('/api/social/posts?limit=20&offset=' + timelineOffset + '&days=30');
      const timeline = document.getElementById('socialTimeline');
      posts.forEach((post) => {
        timeline.insertAdjacentHTML('beforeend', renderPostCard(post));
      });
      timelineOffset += posts.length;

      if (posts.length < 20 && btn) {
        btn.style.display = 'none';
      } else if (btn) {
        btn.textContent = 'Load More';
      }
    } catch (err) {
      if (btn) btn.textContent = 'Load More';
      showToast('Failed to load more posts.');
    }
  };

  // ── Fetch now ──

  window.fetchNow = async function () {
    const btn = document.querySelector('.sb-social-fetch-btn');
    if (btn) {
      btn.disabled = true;
      btn.style.opacity = '0.5';
    }

    try {
      const data = await api('/api/social/fetch', { method: 'POST' });
      const count = data.new_posts || 0;
      showToast(count > 0 ? count + ' new post' + (count !== 1 ? 's' : '') + ' fetched!' : 'No new posts.');
      if (count > 0) {
        setTimeout(() => location.reload(), 1000);
      }
    } catch (err) {
      showToast('Fetch failed: ' + err.message);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.style.opacity = '';
      }
    }
  };

  // ── Init ──

  document.addEventListener('DOMContentLoaded', () => {
    // Mermaid setup
    if (typeof mermaid !== 'undefined') {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        themeVariables: { primaryColor: '#3b82f6' },
      });
    }

    // Enter key on follow input
    const followInput = document.getElementById('followUrl');
    if (followInput) {
      followInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          followChannel();
        }
      });
    }
  });
})();
