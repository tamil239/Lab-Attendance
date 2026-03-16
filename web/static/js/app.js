/* ==========================================================
   Lab Attendance Dashboard — Client-Side Logic
   ========================================================== */

const API = '';  // same origin

// ── DOM ready ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initFilters();
    loadDashboard();
    loadStudents();

    // Auto-refresh every 30 seconds
    setInterval(loadDashboard, 30_000);
});

// ── Navigation ───────────────────────────────────────────────
function initNavigation() {
    const links = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');
    const sidebar = document.getElementById('sidebar');
    const toggle = document.getElementById('menuToggle');

    links.forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            const target = link.dataset.page;

            links.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            pages.forEach(p => {
                p.classList.toggle('active', p.id === `page-${target}`);
            });

            // Close sidebar on mobile
            sidebar.classList.remove('open');

            // Load page-specific data
            if (target === 'dashboard') loadDashboard();
            if (target === 'students') loadStudents();
            if (target === 'analytics') loadAnalytics();
        });
    });

    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
}

// ── Filter bar ───────────────────────────────────────────────
function initFilters() {
    const today = new Date().toISOString().split('T')[0];
    const weekAgo = new Date(Date.now() - 7 * 864e5).toISOString().split('T')[0];

    document.getElementById('filterStart').value = weekAgo;
    document.getElementById('filterEnd').value = today;

    document.getElementById('filterBtn').addEventListener('click', loadRecords);
    document.getElementById('exportRangeBtn').addEventListener('click', exportRange);

    const daysSelect = document.getElementById('analyticsDays');
    daysSelect.addEventListener('change', loadAnalytics);
}

// ── Dashboard ────────────────────────────────────────────────
async function loadDashboard() {
    try {
        const [stats, today] = await Promise.all([
            fetchJSON('/api/stats'),
            fetchJSON('/api/attendance/today'),
        ]);

        document.getElementById('totalStudents').textContent = stats.total_students;
        document.getElementById('todayPresent').textContent = stats.today_present;
        document.getElementById('attendanceRate').textContent = stats.attendance_rate + '%';
        document.getElementById('avgConfidence').textContent =
            (stats.avg_confidence * 100).toFixed(1) + '%';

        document.getElementById('dashboardTime').textContent =
            new Date().toLocaleString();

        renderTable('todayBody', 'todayEmpty', today, row => `
            <td>${esc(row.student_id)}</td>
            <td>${esc(row.name)}</td>
            <td>${formatTime(row.timestamp)}</td>
            <td>${(row.confidence * 100).toFixed(1)}%</td>
            <td>${badge(row.id_card_verified)}</td>
        `);
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

// ── Records ──────────────────────────────────────────────────
async function loadRecords() {
    const start = document.getElementById('filterStart').value;
    const end = document.getElementById('filterEnd').value;
    if (!start || !end) return;

    try {
        const data = await fetchJSON(`/api/attendance?start=${start}&end=${end}`);
        renderTable('recordsBody', 'recordsEmpty', data, row => `
            <td>${esc(row.student_id)}</td>
            <td>${esc(row.name)}</td>
            <td>${formatTime(row.timestamp)}</td>
            <td>${(row.confidence * 100).toFixed(1)}%</td>
            <td>${badge(row.id_card_verified)}</td>
        `);
    } catch (err) {
        console.error('Records load error:', err);
    }
}

// ── Students ─────────────────────────────────────────────────
async function loadStudents() {
    try {
        const data = await fetchJSON('/api/students');
        renderTable('studentsBody', 'studentsEmpty', data, row => `
            <td>${esc(row.student_id)}</td>
            <td>${esc(row.name)}</td>
            <td>${esc(row.email || '—')}</td>
            <td>${formatTime(row.registration_date)}</td>
        `);
    } catch (err) {
        console.error('Students load error:', err);
    }
}

// ── Analytics (Chart.js) ─────────────────────────────────────
let dailyChart = null;
let hourlyChart = null;

async function loadAnalytics() {
    const days = document.getElementById('analyticsDays').value;

    try {
        const [daily, hourly] = await Promise.all([
            fetchJSON(`/api/analytics/daily?days=${days}`),
            fetchJSON('/api/analytics/hourly'),
        ]);

        // ── Daily trend line chart ──
        const dailyCtx = document.getElementById('dailyChart').getContext('2d');
        if (dailyChart) dailyChart.destroy();

        dailyChart = new Chart(dailyCtx, {
            type: 'line',
            data: {
                labels: daily.map(d => d.date),
                datasets: [{
                    label: 'Students Present',
                    data: daily.map(d => d.count),
                    borderColor: '#a78bfa',
                    backgroundColor: 'rgba(167,139,250,0.15)',
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#a78bfa',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                }],
            },
            options: chartOptions('Students'),
        });

        // ── Hourly bar chart ──
        const hourlyCtx = document.getElementById('hourlyChart').getContext('2d');
        if (hourlyChart) hourlyChart.destroy();

        // Fill in missing hours (0-23)
        const hourMap = Object.fromEntries(hourly.map(h => [h.hour, h.count]));
        const hours = Array.from({ length: 24 }, (_, i) => i);

        hourlyChart = new Chart(hourlyCtx, {
            type: 'bar',
            data: {
                labels: hours.map(h => `${h}:00`),
                datasets: [{
                    label: 'Check-ins',
                    data: hours.map(h => hourMap[h] || 0),
                    backgroundColor: 'rgba(96,165,250,0.55)',
                    borderColor: '#60a5fa',
                    borderWidth: 1,
                    borderRadius: 6,
                }],
            },
            options: chartOptions('Check-ins'),
        });
    } catch (err) {
        console.error('Analytics load error:', err);
    }
}

function chartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
        },
        scales: {
            x: {
                ticks: { color: '#94a3b8', font: { size: 11 } },
                grid: { color: 'rgba(148,163,184,0.06)' },
            },
            y: {
                beginAtZero: true,
                ticks: { color: '#94a3b8', font: { size: 11 }, precision: 0 },
                grid: { color: 'rgba(148,163,184,0.06)' },
                title: { display: true, text: yLabel, color: '#94a3b8' },
            },
        },
    };
}

// ── Export helpers ────────────────────────────────────────────
function exportCSV() {
    const today = new Date().toISOString().split('T')[0];
    window.open(`/api/attendance/export?start=${today}&end=${today}`, '_blank');
}

function exportRange() {
    const start = document.getElementById('filterStart').value;
    const end = document.getElementById('filterEnd').value;
    if (start && end) {
        window.open(`/api/attendance/export?start=${start}&end=${end}`, '_blank');
    }
}

// ── Utility functions ────────────────────────────────────────
async function fetchJSON(url) {
    const res = await fetch(API + url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

function renderTable(bodyId, emptyId, rows, rowFn) {
    const tbody = document.getElementById(bodyId);
    const empty = document.getElementById(emptyId);

    if (!rows || rows.length === 0) {
        tbody.innerHTML = '';
        empty.style.display = 'block';
        return;
    }

    empty.style.display = 'none';
    tbody.innerHTML = rows.map(r => `<tr>${rowFn(r)}</tr>`).join('');
}

function formatTime(ts) {
    if (!ts) return '—';
    const d = new Date(ts);
    return isNaN(d) ? ts : d.toLocaleString();
}

function badge(val) {
    const hasCard = (val & 1) === 1;
    const hasTag = (val & 2) === 2;

    // We render three small indicators: FACE (always blue/green), TAG, CARD
    return `
        <div class="status-bar">
            <span class="status-item detected" title="Face Recognized">FACE</span>
            <span class="status-item ${hasTag ? 'detected' : 'missing'}" title="Tag/Lanyard">${hasTag ? 'TAG' : 'NO TAG'}</span>
            <span class="status-item ${hasCard ? 'detected' : 'missing'}" title="ID Card">${hasCard ? 'CARD' : 'NO CARD'}</span>
        </div>
    `;
}

function esc(str) {
    if (!str) return '';
    const el = document.createElement('span');
    el.textContent = str;
    return el.innerHTML;
}
