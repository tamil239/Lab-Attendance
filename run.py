#!/usr/bin/env python3
"""
Lab Attendance System — Main Entry Point
=========================================
Modes:
  collect    – Standalone ID/Tag dataset collection tool
  initdb     – Create / reset the database
"""

import argparse
import os
import sys
import time
# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def cmd_register(args):
    from src.registration import main as register_main
    register_main()


def cmd_attendance(args):
    from src.attendance_system import main as attendance_main
    attendance_main()


def cmd_dashboard(args):
    from web.app import start_server
    print(f"\n  🌐  Dashboard → http://localhost:{args.port}\n")
    start_server(host='0.0.0.0', port=args.port, debug=args.debug)


def cmd_train(args):
    print("\n  Training Modes:")
    print("  1. Train ID Card Detector (YOLOv8)")
    print("  2. Train Face Recognizer  (FaceNet)")

    choice = input("\n  Choose (1/2): ").strip()

    if choice == '1':
        from src.train_id_card import train_id_card_detector, export_best_model
        train_id_card_detector(device=args.device)
        export_best_model()
    elif choice == '2':
        from src.train_face import train_face_recognition
        train_face_recognition()
    else:
        print("  Invalid choice.")


def cmd_initdb(args):
    from src.database import AttendanceDatabase
    db = AttendanceDatabase(args.db)
    stats = db.get_statistics()
    db.close()
    print(f"\n  ✅ Database initialised → {args.db}")
    print(f"     Students : {stats['total_students']}")
    print(f"     Records  : {stats['total_records']}\n")


def cmd_collect(args):
    from src.id_collection import IDCollector
    collector = IDCollector()
    collector.run_collection()


def main():
    parser = argparse.ArgumentParser(
        description='Lab Attendance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='mode', help='Operation mode')

    # ── register ──
    sub.add_parser('register', help='Register new students via webcam')

    # ── attendance ──
    sub.add_parser('attendance', help='Start real-time attendance system')

    # ── dashboard ──
    p_dash = sub.add_parser('dashboard', help='Launch web dashboard')
    p_dash.add_argument('--port', type=int, default=5000)
    p_dash.add_argument('--debug', action='store_true')

    # ── train ──
    p_train = sub.add_parser('train', help='Train models')
    p_train.add_argument('--device', default='cpu', help="'cpu' or '0' for GPU")

    # ── initdb ──
    p_db = sub.add_parser('initdb', help='Initialise the database')
    p_db.add_argument('--db', default='attendance.db', help='Database path')

    # ── collect ──
    sub.add_parser('collect', help='Standalone ID/Tag dataset collection tool')

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    dispatch = {
        'register': cmd_register,
        'attendance': cmd_attendance,
        'dashboard': cmd_dashboard,
        'train': cmd_train,
        'initdb': cmd_initdb,
        'collect': cmd_collect,
    }

    dispatch[args.mode](args)


if __name__ == '__main__':
    main()
