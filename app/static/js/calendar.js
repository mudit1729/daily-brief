/* ── Calendar — day / week / month views ─────── */
(function () {
  const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
  const DAYS  = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];
  const DAYS_SHORT = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const HOURS = Array.from({length: 24}, (_, i) => {
    if (i === 0) return '12 AM';
    if (i < 12) return i + ' AM';
    if (i === 12) return '12 PM';
    return (i - 12) + ' PM';
  });

  // DOM refs
  const label     = document.getElementById('calLabel');
  const modal     = document.getElementById('eventModal');
  const form      = document.getElementById('eventForm');
  const monthView = document.getElementById('monthView');
  const weekView  = document.getElementById('weekView');
  const dayView   = document.getElementById('dayView');
  const calGrid   = document.getElementById('calGrid');
  const weekHeader = document.getElementById('weekHeader');
  const weekBody  = document.getElementById('weekBody');
  const dayBody   = document.getElementById('dayBody');
  const recSelect = document.getElementById('eventRecurrence');
  const recEndField = document.getElementById('recEndField');

  let currentView = 'day';
  let currentDate = new Date(); // anchor date
  let events = [];

  // Get current time in PST (America/Los_Angeles)
  function getPSTDate() {
    return new Date(new Date().toLocaleString('en-US', { timeZone: 'America/Los_Angeles' }));
  }

  function getPSTHour() {
    const pst = getPSTDate();
    return pst.getHours();
  }

  function getPSTMinute() {
    const pst = getPSTDate();
    return pst.getMinutes();
  }

  /* ── Init ─────────────────────────────────── */
  function init() {
    document.getElementById('calPrev').addEventListener('click', () => nav(-1));
    document.getElementById('calNext').addEventListener('click', () => nav(1));
    document.getElementById('calToday').addEventListener('click', goToday);
    document.getElementById('modalClose').addEventListener('click', closeModal);
    document.getElementById('eventCancelBtn').addEventListener('click', closeModal);
    document.getElementById('eventDeleteBtn').addEventListener('click', deleteEvent);
    form.addEventListener('submit', saveEvent);

    // View switcher
    document.querySelectorAll('.sb-cal__view-btn').forEach(btn => {
      btn.addEventListener('click', () => switchView(btn.dataset.view));
    });

    // Color picker
    document.querySelectorAll('.sb-color-swatch').forEach(sw => {
      sw.addEventListener('click', () => {
        document.querySelectorAll('.sb-color-swatch').forEach(s => s.classList.remove('active'));
        sw.classList.add('active');
      });
    });

    // Recurrence toggle
    recSelect.addEventListener('change', () => {
      recEndField.style.display = recSelect.value ? '' : 'none';
    });

    modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

    // Set default view to day
    switchView('day');

    // Update time indicator every minute
    setInterval(updateTimeIndicators, 60000);
  }

  /* ── Navigation ──────────────────────────── */
  function nav(dir) {
    if (currentView === 'month') {
      currentDate.setMonth(currentDate.getMonth() + dir);
    } else if (currentView === 'week') {
      currentDate.setDate(currentDate.getDate() + dir * 7);
    } else {
      currentDate.setDate(currentDate.getDate() + dir);
    }
    render();
  }

  function goToday() {
    currentDate = getPSTDate();
    render();
  }

  function switchView(view) {
    currentView = view;
    document.querySelectorAll('.sb-cal__view-btn').forEach(b => b.classList.toggle('active', b.dataset.view === view));
    monthView.style.display = view === 'month' ? '' : 'none';
    weekView.style.display = view === 'week' ? '' : 'none';
    dayView.style.display = view === 'day' ? '' : 'none';
    render();
  }

  /* ── Render dispatcher ───────────────────── */
  function render() {
    updateLabel();
    const { start, end } = getRange();
    fetchEvents(start, end).then(() => {
      if (currentView === 'month') renderMonth();
      else if (currentView === 'week') renderWeek();
      else renderDay();
    });
  }

  function updateLabel() {
    if (currentView === 'month') {
      label.textContent = MONTHS[currentDate.getMonth()] + ' ' + currentDate.getFullYear();
    } else if (currentView === 'week') {
      const ws = weekStart(currentDate);
      const we = new Date(ws); we.setDate(we.getDate() + 6);
      if (ws.getMonth() === we.getMonth()) {
        label.textContent = MONTHS[ws.getMonth()] + ' ' + ws.getDate() + '–' + we.getDate() + ', ' + ws.getFullYear();
      } else {
        label.textContent = MONTHS[ws.getMonth()].slice(0,3) + ' ' + ws.getDate() + ' – ' + MONTHS[we.getMonth()].slice(0,3) + ' ' + we.getDate() + ', ' + we.getFullYear();
      }
    } else {
      label.textContent = DAYS[currentDate.getDay()] + ', ' + MONTHS[currentDate.getMonth()] + ' ' + currentDate.getDate() + ', ' + currentDate.getFullYear();
    }
  }

  function getRange() {
    if (currentView === 'month') {
      const y = currentDate.getFullYear(), m = currentDate.getMonth();
      const first = new Date(y, m, 1);
      const startPad = new Date(first); startPad.setDate(1 - first.getDay());
      const last = new Date(y, m + 1, 0);
      const endPad = new Date(last); endPad.setDate(last.getDate() + (6 - last.getDay()));
      return { start: fmtDate(startPad), end: fmtDate(endPad) };
    } else if (currentView === 'week') {
      const ws = weekStart(currentDate);
      const we = new Date(ws); we.setDate(we.getDate() + 6);
      return { start: fmtDate(ws), end: fmtDate(we) };
    } else {
      return { start: fmtDate(currentDate), end: fmtDate(currentDate) };
    }
  }

  function fetchEvents(start, end) {
    return fetch('/api/calendar/events?start=' + start + '&end=' + end)
      .then(r => r.json())
      .then(data => { events = data; })
      .catch(() => { events = []; });
  }

  /* ── Month view ──────────────────────────── */
  function renderMonth() {
    const y = currentDate.getFullYear(), m = currentDate.getMonth();
    const firstDay = new Date(y, m, 1).getDay();
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    const today = new Date();
    const isCurrentMonth = today.getFullYear() === y && today.getMonth() === m;
    const prevMonthDays = new Date(y, m, 0).getDate();

    let html = '';

    // Leading days from previous month
    for (let i = firstDay - 1; i >= 0; i--) {
      const d = prevMonthDays - i;
      const dt = new Date(y, m - 1, d);
      const dateStr = fmtDate(dt);
      html += '<div class="sb-cal__cell sb-cal__cell--outside" data-date="' + dateStr + '">';
      html += '<div class="sb-cal__day-num">' + d + '</div>';
      html += renderEventPills(dateStr);
      html += '</div>';
    }

    // Current month days
    for (let d = 1; d <= daysInMonth; d++) {
      const dateStr = y + '-' + pad(m + 1) + '-' + pad(d);
      const isToday = isCurrentMonth && d === today.getDate();
      html += '<div class="sb-cal__cell' + (isToday ? ' sb-cal__cell--today' : '') + '" data-date="' + dateStr + '">';
      html += '<div class="sb-cal__day-num">' + d + '</div>';
      html += renderEventPills(dateStr);
      html += '</div>';
    }

    // Trailing days
    const totalCells = firstDay + daysInMonth;
    const trailing = (7 - totalCells % 7) % 7;
    for (let d = 1; d <= trailing; d++) {
      const dt = new Date(y, m + 1, d);
      const dateStr = fmtDate(dt);
      html += '<div class="sb-cal__cell sb-cal__cell--outside" data-date="' + dateStr + '">';
      html += '<div class="sb-cal__day-num">' + d + '</div>';
      html += renderEventPills(dateStr);
      html += '</div>';
    }

    calGrid.innerHTML = html;
    attachCellListeners(calGrid);
  }

  function renderEventPills(dateStr) {
    const dayEvts = events.filter(e => e.event_date === dateStr);
    if (dayEvts.length === 0) return '';
    let html = '<div class="sb-cal__events">';
    dayEvts.forEach(evt => {
      const recIcon = evt.recurrence ? ' ↻' : '';
      html += '<button class="sb-cal__event-pill" style="--pill-color:' + (evt.color || '#6366f1') + ';" '
            + 'data-event-id="' + evt.id + '" onclick="window._calEditEvent(' + evt.id + ')">'
            + escapeHtml(evt.title) + recIcon + '</button>';
    });
    return html + '</div>';
  }

  /* ── Week view ───────────────────────────── */
  function renderWeek() {
    const ws = weekStart(currentDate);
    const today = fmtDate(new Date());

    // Header with day labels
    let hdr = '<div class="sb-cal__week-gutter"></div>';
    for (let i = 0; i < 7; i++) {
      const d = new Date(ws);
      d.setDate(d.getDate() + i);
      const dateStr = fmtDate(d);
      const cls = dateStr === today ? ' sb-cal__week-day--today' : '';
      hdr += '<div class="sb-cal__week-day' + cls + '" data-date="' + dateStr + '">'
           + '<span class="sb-cal__week-day-name">' + DAYS_SHORT[d.getDay()] + '</span>'
           + '<span class="sb-cal__week-day-num">' + d.getDate() + '</span>'
           + '</div>';
    }
    weekHeader.innerHTML = hdr;

    // Time grid
    let body = '';
    for (let h = 0; h < 24; h++) {
      body += '<div class="sb-cal__week-row">';
      body += '<div class="sb-cal__week-gutter sb-cal__week-time">' + HOURS[h] + '</div>';
      for (let i = 0; i < 7; i++) {
        const d = new Date(ws);
        d.setDate(d.getDate() + i);
        const dateStr = fmtDate(d);
        body += '<div class="sb-cal__week-cell" data-date="' + dateStr + '" data-hour="' + h + '"></div>';
      }
      body += '</div>';
    }
    weekBody.innerHTML = body;

    // Place events
    for (let i = 0; i < 7; i++) {
      const d = new Date(ws);
      d.setDate(d.getDate() + i);
      const dateStr = fmtDate(d);
      const dayEvts = events.filter(e => e.event_date === dateStr);
      dayEvts.forEach(evt => {
        const hour = evt.event_time ? parseInt(evt.event_time.split(':')[0]) : 8;
        const min = evt.event_time ? parseInt(evt.event_time.split(':')[1]) : 0;
        let duration = 1; // default 1 hour
        if (evt.event_time && evt.end_time) {
          const startMin = hour * 60 + min;
          const endH = parseInt(evt.end_time.split(':')[0]);
          const endM = parseInt(evt.end_time.split(':')[1]);
          duration = Math.max(0.5, (endH * 60 + endM - startMin) / 60);
        }
        const cell = weekBody.querySelector('.sb-cal__week-cell[data-date="' + dateStr + '"][data-hour="' + hour + '"]');
        if (cell) {
          const pill = document.createElement('div');
          pill.className = 'sb-cal__time-event';
          pill.style.cssText = '--pill-color:' + (evt.color || '#6366f1')
            + ';top:' + (min / 60 * 100) + '%;height:' + (duration * 100) + '%;';
          const recIcon = evt.recurrence ? ' ↻' : '';
          pill.innerHTML = '<strong>' + escapeHtml(evt.title) + recIcon + '</strong>'
            + (evt.event_time ? '<span>' + fmtTime(evt.event_time) + (evt.end_time ? ' – ' + fmtTime(evt.end_time) : '') + '</span>' : '');
          pill.addEventListener('click', e => { e.stopPropagation(); window._calEditEvent(evt.id); });
          cell.appendChild(pill);
        }
      });
    }

    // Click to add
    weekBody.querySelectorAll('.sb-cal__week-cell').forEach(cell => {
      cell.addEventListener('click', e => {
        if (e.target.closest('.sb-cal__time-event')) return;
        openModal(cell.dataset.date, null, cell.dataset.hour);
      });
    });

    // Add current time indicator for today's column
    const todayStr = fmtDate(getPSTDate());
    addTimeIndicator(weekBody, '.sb-cal__week-cell', todayStr);

    // Auto-scroll to current hour
    scrollToCurrentTime(weekBody);
  }

  /* ── Day view ────────────────────────────── */
  function renderDay() {
    const dateStr = fmtDate(currentDate);
    const dayEvts = events.filter(e => e.event_date === dateStr);

    let html = '';
    for (let h = 0; h < 24; h++) {
      html += '<div class="sb-cal__day-row">';
      html += '<div class="sb-cal__day-time">' + HOURS[h] + '</div>';
      html += '<div class="sb-cal__day-cell" data-date="' + dateStr + '" data-hour="' + h + '"></div>';
      html += '</div>';
    }
    dayBody.innerHTML = html;

    // Place events
    dayEvts.forEach(evt => {
      const hour = evt.event_time ? parseInt(evt.event_time.split(':')[0]) : 8;
      const min = evt.event_time ? parseInt(evt.event_time.split(':')[1]) : 0;
      let duration = 1;
      if (evt.event_time && evt.end_time) {
        const startMin = hour * 60 + min;
        const endH = parseInt(evt.end_time.split(':')[0]);
        const endM = parseInt(evt.end_time.split(':')[1]);
        duration = Math.max(0.5, (endH * 60 + endM - startMin) / 60);
      }
      const cell = dayBody.querySelector('.sb-cal__day-cell[data-hour="' + hour + '"]');
      if (cell) {
        const pill = document.createElement('div');
        pill.className = 'sb-cal__time-event';
        pill.style.cssText = '--pill-color:' + (evt.color || '#6366f1')
          + ';top:' + (min / 60 * 100) + '%;height:' + (duration * 100) + '%;';
        const recIcon = evt.recurrence ? ' ↻' : '';
        pill.innerHTML = '<strong>' + escapeHtml(evt.title) + recIcon + '</strong>'
          + (evt.event_time ? '<span>' + fmtTime(evt.event_time) + (evt.end_time ? ' – ' + fmtTime(evt.end_time) : '') + '</span>' : '');
        pill.addEventListener('click', e => { e.stopPropagation(); window._calEditEvent(evt.id); });
        cell.appendChild(pill);
      }
    });

    // Click to add
    dayBody.querySelectorAll('.sb-cal__day-cell').forEach(cell => {
      cell.addEventListener('click', e => {
        if (e.target.closest('.sb-cal__time-event')) return;
        openModal(cell.dataset.date, null, cell.dataset.hour);
      });
    });

    // Add current time indicator if viewing today
    addTimeIndicator(dayBody, '.sb-cal__day-cell', dateStr);

    // Auto-scroll to current hour
    scrollToCurrentTime(dayBody);
  }

  /* ── Modal ───────────────────────────────── */
  function openModal(dateStr, evt, hour) {
    document.getElementById('modalTitle').textContent = evt ? 'Edit Event' : 'Add Event';
    document.getElementById('eventId').value = evt ? evt.id : '';
    document.getElementById('eventTitleInput').value = evt ? evt.title : '';
    document.getElementById('eventDateInput').value = evt ? evt.event_date : dateStr;
    document.getElementById('eventTimeInput').value = evt ? (evt.event_time || '') : (hour ? pad(parseInt(hour)) + ':00' : '');
    document.getElementById('eventEndTimeInput').value = evt ? (evt.end_time || '') : (hour ? pad(parseInt(hour) + 1) + ':00' : '');
    document.getElementById('eventDescInput').value = evt ? (evt.description || '') : '';
    recSelect.value = evt ? (evt.recurrence || '') : '';
    document.getElementById('eventRecEnd').value = evt ? (evt.recurrence_end || '') : '';
    recEndField.style.display = recSelect.value ? '' : 'none';
    document.getElementById('eventDeleteBtn').style.display = evt ? '' : 'none';

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
      end_time: document.getElementById('eventEndTimeInput').value || null,
      description: document.getElementById('eventDescInput').value,
      color: getSelectedColor(),
      recurrence: recSelect.value || null,
      recurrence_end: document.getElementById('eventRecEnd').value || null,
    };

    const url = id ? '/api/calendar/events/' + id : '/api/calendar/events';
    const method = id ? 'PUT' : 'POST';

    fetch(url, {
      method: method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
      .then(r => { if (!r.ok) throw new Error('Save failed'); return r.json(); })
      .then(() => { closeModal(); render(); })
      .catch(err => alert('Error: ' + err.message));
  }

  function deleteEvent() {
    const id = document.getElementById('eventId').value;
    if (!id || !confirm('Delete this event and all its recurrences?')) return;
    fetch('/api/calendar/events/' + id, { method: 'DELETE' })
      .then(r => { if (!r.ok) throw new Error('Delete failed'); return r.json(); })
      .then(() => { closeModal(); render(); })
      .catch(err => alert('Error: ' + err.message));
  }

  /* ── Time indicator ─────────────────────── */
  function addTimeIndicator(container, cellSelector, todayStr) {
    const pstNow = getPSTDate();
    const today = fmtDate(pstNow);
    if (todayStr !== today && cellSelector === '.sb-cal__day-cell') return;

    const hour = getPSTHour();
    const min = getPSTMinute();

    // For day view, find the cell for the current hour
    if (cellSelector === '.sb-cal__day-cell') {
      const cell = container.querySelector(cellSelector + '[data-hour="' + hour + '"]');
      if (cell) {
        cell.style.position = 'relative';
        const line = document.createElement('div');
        line.className = 'sb-cal__now-line';
        line.style.top = (min / 60 * 100) + '%';
        cell.appendChild(line);
      }
    } else {
      // Week view — add indicator to today's column
      const cell = container.querySelector(cellSelector + '[data-date="' + today + '"][data-hour="' + hour + '"]');
      if (cell) {
        cell.style.position = 'relative';
        const line = document.createElement('div');
        line.className = 'sb-cal__now-line';
        line.style.top = (min / 60 * 100) + '%';
        cell.appendChild(line);
      }
    }
  }

  function scrollToCurrentTime(container) {
    const hour = getPSTHour();
    // Scroll to 1 hour before current time for context
    const scrollHour = Math.max(0, hour - 1);
    const rows = container.querySelectorAll('.sb-cal__day-row, .sb-cal__week-row');
    if (rows.length > scrollHour) {
      rows[scrollHour].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function updateTimeIndicators() {
    // Remove existing indicators and re-add
    document.querySelectorAll('.sb-cal__now-line').forEach(el => el.remove());
    if (currentView === 'day') {
      const dateStr = fmtDate(currentDate);
      addTimeIndicator(dayBody, '.sb-cal__day-cell', dateStr);
    } else if (currentView === 'week') {
      const todayStr = fmtDate(getPSTDate());
      addTimeIndicator(weekBody, '.sb-cal__week-cell', todayStr);
    }
  }

  /* ── Helpers ─────────────────────────────── */
  function attachCellListeners(container) {
    container.querySelectorAll('.sb-cal__cell:not(.sb-cal__cell--empty)').forEach(cell => {
      cell.addEventListener('click', e => {
        if (e.target.closest('.sb-cal__event-pill')) return;
        openModal(cell.dataset.date);
      });
    });
  }

  window._calEditEvent = function (id) {
    const evt = events.find(e => e.id === id);
    if (evt) openModal(null, evt);
  };

  function weekStart(d) {
    const ws = new Date(d);
    ws.setDate(ws.getDate() - ws.getDay());
    ws.setHours(0, 0, 0, 0);
    return ws;
  }

  function fmtDate(d) {
    if (typeof d === 'string') return d;
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
  }

  function fmtTime(t) {
    if (!t) return '';
    const [h, m] = t.split(':').map(Number);
    const ampm = h >= 12 ? 'PM' : 'AM';
    const hr = h % 12 || 12;
    return hr + ':' + pad(m) + ' ' + ampm;
  }

  function pad(n) { return String(n).padStart(2, '0'); }

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
