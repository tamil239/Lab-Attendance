import sqlite3
import numpy as np
import pickle
from datetime import datetime
import logging
import sys

import logging
import sys

# ── Numpy Shim ────────────────────────────────────────────────────────
def _apply_numpy_shim():
    """Shim for unpickling data from different numpy versions (e.g., Numpy 2.0 -> 1.x)"""
    try:
        import numpy as np
        if not hasattr(np, '_core'):
            # On Numpy 1.x, map Numpy 2.x's _core to core
            import numpy.core as core
            sys.modules['numpy._core'] = core
            if hasattr(core, 'multiarray'):
                sys.modules['numpy._core.multiarray'] = core.multiarray
    except ImportError:
        pass
    except Exception:
        pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttendanceDatabase:
    def __init__(self, db_path='attendance.db'):
        self.db_path = db_path
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                face_embedding BLOB,
                registration_date DATETIME DEFAULT (datetime('now', 'localtime')),
                active BOOLEAN DEFAULT 1
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                confidence FLOAT,
                id_card_verified BOOLEAN,
                FOREIGN KEY(student_id) REFERENCES students(student_id)
            )
        ''')

        # Index for fast daily duplicate checks
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_attendance_daily
            ON attendance(student_id, timestamp)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT (datetime('now', 'localtime')),
                event_type TEXT,
                message TEXT,
                student_id TEXT
            )
        ''')

        self.conn.commit()
        logger.info("Database tables created/verified")

    # ------------------------------------------------------------------
    # Student CRUD
    # ------------------------------------------------------------------

    def add_student(self, student_id, name, email, face_embedding):
        """Add new student with face embedding"""
        cursor = self.conn.cursor()
        embedding_blob = pickle.dumps(face_embedding)

        try:
            cursor.execute('''
                INSERT INTO students (student_id, name, email, face_embedding)
                VALUES (?, ?, ?, ?)
            ''', (student_id, name, email, embedding_blob))
            self.conn.commit()
            logger.info(f"Student {name} (ID: {student_id}) added successfully")
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Student ID {student_id} already exists")
            return False

    def get_all_students(self):
        """Get all active students"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT student_id, name, email, registration_date
            FROM students WHERE active = 1
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def get_student_embeddings(self):
        """Retrieve all student embeddings for recognition"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT student_id, name, face_embedding FROM students WHERE active = 1'
        )

        _apply_numpy_shim()
        students = []
        for row in cursor.fetchall():
            if row['face_embedding']:
                embedding = pickle.loads(row['face_embedding'])
                students.append({
                    'student_id': row['student_id'],
                    'name': row['name'],
                    'embedding': embedding,
                })
        return students

    def delete_student(self, student_id):
        """Soft-delete a student"""
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE students SET active = 0 WHERE student_id = ?', (student_id,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------

    def mark_attendance(self, student_id, confidence, id_card_verified):
        """Mark attendance for a student (one entry per day)"""
        cursor = self.conn.cursor()

        # Check for duplicate today
        cursor.execute('''
            SELECT id FROM attendance
            WHERE student_id = ? AND DATE(timestamp) = DATE('now', 'localtime')
        ''', (student_id,))

        existing = cursor.fetchone()
        if existing:
            # Update the existing record's timestamp to the current local time 
            # for testing/feedback purposes, without creating multiple rows.
            cursor.execute('''
                UPDATE attendance 
                SET timestamp = datetime('now', 'localtime'),
                    confidence = ?,
                    id_card_verified = ?
                WHERE id = ?
            ''', (confidence, id_card_verified, existing['id']))
            self.conn.commit()
            return True

        cursor.execute('''
            INSERT INTO attendance (student_id, confidence, id_card_verified, timestamp)
            VALUES (?, ?, ?, datetime('now', 'localtime'))
        ''', (student_id, confidence, id_card_verified))
        self.conn.commit()
        self.log_event('attendance', f"Attendance marked for {student_id}", student_id)
        return True

    def get_todays_attendance(self):
        """Get today's attendance records"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT s.student_id, s.name, a.timestamp, a.confidence, a.id_card_verified
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE DATE(a.timestamp) = DATE('now', 'localtime')
            ORDER BY a.timestamp DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_by_date(self, date_str):
        """Get attendance for a specific date (YYYY-MM-DD)"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT s.student_id, s.name, a.timestamp, a.confidence, a.id_card_verified
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE DATE(a.timestamp) = ?
            ORDER BY a.timestamp DESC
        ''', (date_str,))
        return [dict(row) for row in cursor.fetchall()]

    def get_attendance_range(self, start_date, end_date):
        """Get attendance between two dates inclusive"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT s.student_id, s.name, a.timestamp, a.confidence, a.id_card_verified
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE DATE(a.timestamp) BETWEEN ? AND ?
            ORDER BY a.timestamp DESC
        ''', (start_date, end_date))
        return [dict(row) for row in cursor.fetchall()]

    def get_daily_counts(self, start_date, end_date):
        """Get daily unique-student attendance counts"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(DISTINCT student_id) as count
            FROM attendance
            WHERE DATE(timestamp) BETWEEN ? AND ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_date, end_date))
        return [dict(row) for row in cursor.fetchall()]

    def get_hourly_counts(self, date_str=None):
        """Get attendance counts by hour"""
        cursor = self.conn.cursor()
        if date_str:
            cursor.execute('''
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                       COUNT(*) as count
                FROM attendance
                WHERE DATE(timestamp) = ?
                GROUP BY hour ORDER BY hour
            ''', (date_str,))
        else:
            cursor.execute('''
                SELECT CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                       COUNT(*) as count
                FROM attendance
                GROUP BY hour ORDER BY hour
            ''')
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self):
        """Get summary statistics"""
        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) as c FROM students WHERE active = 1')
        total_students = cursor.fetchone()['c']

        cursor.execute('''
            SELECT COUNT(DISTINCT student_id) as c
            FROM attendance
            WHERE DATE(timestamp) = DATE('now', 'localtime')
        ''')
        today_present = cursor.fetchone()['c']

        cursor.execute('''
            SELECT AVG(confidence) as c
            FROM attendance
            WHERE DATE(timestamp) = DATE('now', 'localtime')
        ''')
        avg_confidence = cursor.fetchone()['c'] or 0

        cursor.execute('SELECT COUNT(*) as c FROM attendance')
        total_records = cursor.fetchone()['c']

        return {
            'total_students': total_students,
            'today_present': today_present,
            'attendance_rate': round(
                (today_present / total_students * 100) if total_students > 0 else 0, 1
            ),
            'avg_confidence': round(avg_confidence, 4),
            'total_records': total_records,
        }

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def log_event(self, event_type, message, student_id=None):
        """Log system events"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO logs (event_type, message, student_id)
            VALUES (?, ?, ?)
        ''', (event_type, message, student_id))
        self.conn.commit()

    def get_recent_logs(self, limit=50):
        """Get recent log entries"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM logs ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close()
