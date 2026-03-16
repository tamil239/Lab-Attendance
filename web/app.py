"""
Flask REST API + Dashboard Server
Serves the attendance dashboard and provides JSON endpoints.
"""

import os
import sys
import csv
import io
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.database import AttendanceDatabase
from src.utils import load_config

# ── App setup ─────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
    static_folder=os.path.join(os.path.dirname(__file__), 'static'),
)
CORS(app)

config = load_config(os.path.join(PROJECT_ROOT, 'config.yaml'))
db = AttendanceDatabase(
    os.path.join(PROJECT_ROOT, config.get('database', {}).get('path', 'attendance.db'))
)

# ── Pages ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ── API endpoints ─────────────────────────────────────────────────────

@app.route('/api/stats')
def api_stats():
    """Summary statistics for the dashboard cards."""
    return jsonify(db.get_statistics())


@app.route('/api/attendance/today')
def api_today():
    """Today's attendance records."""
    return jsonify(db.get_todays_attendance())


@app.route('/api/attendance')
def api_attendance():
    """Attendance for a date range.  ?start=YYYY-MM-DD&end=YYYY-MM-DD"""
    start = request.args.get('start', datetime.now().strftime('%Y-%m-%d'))
    end = request.args.get('end', start)
    return jsonify(db.get_attendance_range(start, end))


@app.route('/api/students')
def api_students():
    """List of registered students."""
    return jsonify(db.get_all_students())


@app.route('/api/analytics/daily')
def api_daily():
    """Daily attendance counts for the chart.  ?days=7"""
    days = int(request.args.get('days', 7))
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days - 1)).strftime('%Y-%m-%d')
    return jsonify(db.get_daily_counts(start, end))


@app.route('/api/analytics/hourly')
def api_hourly():
    """Hourly attendance distribution."""
    date = request.args.get('date')
    return jsonify(db.get_hourly_counts(date))


@app.route('/api/logs')
def api_logs():
    """Recent system logs."""
    limit = int(request.args.get('limit', 50))
    return jsonify(db.get_recent_logs(limit))


@app.route('/api/attendance/export')
def api_export():
    """Export attendance as a CSV download."""
    start = request.args.get('start', datetime.now().strftime('%Y-%m-%d'))
    end = request.args.get('end', start)
    records = db.get_attendance_range(start, end)

    si = io.StringIO()
    writer = csv.DictWriter(si, fieldnames=[
        'student_id', 'name', 'timestamp', 'confidence', 'id_card_verified',
    ])
    writer.writeheader()
    writer.writerows(records)

    output = si.getvalue()
    filename = f"attendance_{start}_to_{end}.csv"

    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'},
    )


# ── Run ───────────────────────────────────────────────────────────────

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask development server."""
    web_cfg = config.get('web', {})
    app.run(
        host=web_cfg.get('host', host),
        port=web_cfg.get('port', port),
        debug=web_cfg.get('debug', debug),
    )


if __name__ == '__main__':
    start_server(debug=True)
