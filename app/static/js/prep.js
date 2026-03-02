/* ── Prep: Notes Viewer ────────────────────────── */

function loadNote(filename, btn) {
  // Mark active
  document.querySelectorAll('.sb-notes-list__item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const content = document.getElementById('notesContent');
  const expandBtn = document.getElementById('notesExpandBtn');
  content.innerHTML = '<div class="sb-notes-empty"><p>Loading...</p></div>';
  // Re-attach expand button (it gets wiped by innerHTML)
  content.prepend(expandBtn);
  expandBtn.style.display = '';

  fetch('/api/prep/notes/' + encodeURIComponent(filename))
    .then(r => {
      if (!r.ok) throw new Error('Failed to load note');
      return r.json();
    })
    .then(data => {
      content.innerHTML = '<div class="md-body">' + data.html + '</div>';
      content.prepend(expandBtn);
      expandBtn.style.display = '';
    })
    .catch(err => {
      content.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
      content.prepend(expandBtn);
    });
}

function toggleNotesFullscreen() {
  const layout = document.getElementById('notesLayout');
  const expandIcon = document.getElementById('expandIcon');
  const collapseIcon = document.getElementById('collapseIcon');
  const btn = document.getElementById('notesExpandBtn');

  layout.classList.toggle('sb-notes-layout--fullscreen');

  const isFullscreen = layout.classList.contains('sb-notes-layout--fullscreen');
  expandIcon.style.display = isFullscreen ? 'none' : '';
  collapseIcon.style.display = isFullscreen ? '' : 'none';
  btn.title = isFullscreen ? 'Collapse' : 'Expand';
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
