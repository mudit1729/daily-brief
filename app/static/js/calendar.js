/* ── Calendar ─────────────────────────────────── */
(function () {
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const grid = document.getElementById('calGrid');
  const label = document.getElementById('calMonthLabel');
  const modal = document.getElementById('eventModal');
  const form = document.getElementById('eventForm');

  let currentYear, currentMonth; // 0-indexed month
  let events = [];

  function init() {
    const now = new Date();
    currentYear = now.getFullYear();
    currentMonth = now.getMonth();

    document.getElementById('calPrev').addEventListener('click', () => { changeMonth(-1); });
    document.getElementById('calNext').addEventListener('click', () => { changeMonth(1); });
    document.getElementById('calToday').addEventListener('click', goToday);
    document.getElementById('modalClose').addEventListener('click', closeModal);
    document.getElementById('eventCancelBtn').addEventListener('click', closeModal);
    document.getElementById('eventDeleteBtn').addEventListener('click', deleteEvent);
    form.addEventListener('submit', saveEvent);

    // Color picker
    document.querySelectorAll('.sb-color-swatch').forEach(sw => {
      sw.addEventListener('click', () => {
        document.querySelectorAll('.sb-color-swatch').forEach(s => s.classList.remove('active'));
        sw.classList.add('active');
      });
    });

    // Close modal on overlay click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });

    render();
  }

  function changeMonth(delta) {
    currentMonth += delta;
    if (currentMonth < 0) { currentMonth = 11; currentYear--; }
    if (currentMonth > 11) { currentMonth = 0; currentYear++; }
    render();
  }

  function goToday() {
    const now = new Date();
    currentYear = now.getFullYear();
    currentMonth = now.getMonth();
    render();
  }

  function render() {
    label.textContent = MONTHS[currentMonth] + ' ' + currentYear;
    fetchEvents().then(() => renderGrid());
  }

  function fetchEvents() {
    const mm = String(currentMonth + 1).padStart(2, '0');
    return fetch('/api/calendar/events?month=' + currentYear + '-' + mm)
      .then(r => r.json())
      .then(data => { events = data; })
      .catch(() => { events = []; });
  }

  function renderGrid() {
    const firstDay = new Date(currentYear, currentMonth, 1).getDay();
    const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    const today = new Date();
    const isCurrentMonth = today.getFullYear() === currentYear && today.getMonth() === currentMonth;

    let html = '';

    // Leading blanks
    for (let i = 0; i < firstDay; i++) {
      html += '<div class="sb-cal__cell sb-cal__cell--empty"></div>';
    }

    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = currentYear + '-' + String(currentMonth + 1).padStart(2, '0') + '-' + String(d).padStart(2, '0');
      const isToday = isCurrentMonth && d === today.getDate();
      const dayEvents = events.filter(e => e.event_date === dateStr);

      html += '<div class="sb-cal__cell' + (isToday ? ' sb-cal__cell--today' : '') + '" data-date="' + dateStr + '">';
      html += '<div class="sb-cal__day-num">' + d + '</div>';

      if (dayEvents.length > 0) {
        html += '<div class="sb-cal__events">';
        dayEvents.forEach(evt => {
          html += '<button class="sb-cal__event-pill" style="--pill-color:' + (evt.color || '#6366f1') + ';" '
                + 'data-event-id="' + evt.id + '" onclick="window._calEditEvent(' + evt.id + ')">'
                + escapeHtml(evt.title)
                + '</button>';
        });
        html += '</div>';
      }

      html += '</div>';
    }

    grid.innerHTML = html;

    // Click empty area to add event
    grid.querySelectorAll('.sb-cal__cell:not(.sb-cal__cell--empty)').forEach(cell => {
      cell.addEventListener('click', (e) => {
        if (e.target.closest('.sb-cal__event-pill')) return;
        openModal(cell.dataset.date);
      });
    });
  }

  function openModal(dateStr, evt) {
    document.getElementById('modalTitle').textContent = evt ? 'Edit Event' : 'Add Event';
    document.getElementById('eventId').value = evt ? evt.id : '';
    document.getElementById('eventTitleInput').value = evt ? evt.title : '';
    document.getElementById('eventDateInput').value = evt ? evt.event_date : dateStr;
    document.getElementById('eventTimeInput').value = evt ? (evt.event_time || '') : '';
    document.getElementById('eventDescInput').value = evt ? (evt.description || '') : '';
    document.getElementById('eventDeleteBtn').style.display = evt ? '' : 'none';

    // Set color
    const color = evt ? evt.color : '#6366f1';
    document.querySelectorAll('.sb-color-swatch').forEach(s => {
      s.classList.toggle('active', s.dataset.color === color);
    });

    modal.style.display = '';
    document.getElementById('eventTitleInput').focus();
  }

  function closeModal() {
    modal.style.display = 'none';
    form.reset();
  }

  function getSelectedColor() {
    const active = document.querySelector('.sb-color-swatch.active');
    return active ? active.dataset.color : '#6366f1';
  }

  function saveEvent(e) {
    e.preventDefault();
    const id = document.getElementById('eventId').value;
    const payload = {
      title: document.getElementById('eventTitleInput').value,
      event_date: document.getElementById('eventDateInput').value,
      event_time: document.getElementById('eventTimeInput').value || null,
      description: document.getElementById('eventDescInput').value,
      color: getSelectedColor(),
    };

    const url = id ? '/api/calendar/events/' + id : '/api/calendar/events';
    const method = id ? 'PUT' : 'POST';

    fetch(url, {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(r => {
        if (!r.ok) throw new Error('Save failed');
        return r.json();
      })
      .then(() => { closeModal(); render(); })
      .catch(err => alert('Error: ' + err.message));
  }

  function deleteEvent() {
    const id = document.getElementById('eventId').value;
    if (!id || !confirm('Delete this event?')) return;
    fetch('/api/calendar/events/' + id, { method: 'DELETE' })
      .then(r => {
        if (!r.ok) throw new Error('Delete failed');
        return r.json();
      })
      .then(() => { closeModal(); render(); })
      .catch(err => alert('Error: ' + err.message));
  }

  // Expose edit handler for inline onclick
  window._calEditEvent = function (id) {
    const evt = events.find(e => e.id === id);
    if (evt) openModal(null, evt);
  };

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
