/* ── Prep: Notes Viewer ────────────────────────── */

function loadNote(filename, btn) {
  // Mark active
  document.querySelectorAll('.sb-notes-list__item').forEach(el => el.classList.remove('active'));
  if (btn) btn.classList.add('active');

  const content = document.getElementById('notesContent');
  content.innerHTML = '<div class="sb-notes-empty"><p>Loading...</p></div>';

  fetch('/api/prep/notes/' + encodeURIComponent(filename))
    .then(r => {
      if (!r.ok) throw new Error('Failed to load note');
      return r.json();
    })
    .then(data => {
      content.innerHTML = '<div class="md-body">' + data.html + '</div>';
    })
    .catch(err => {
      content.innerHTML = '<div class="sb-notes-empty"><p>Error: ' + err.message + '</p></div>';
    });
}
