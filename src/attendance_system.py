"""
Main Attendance System
Real-time face recognition + ID card detection using YOLOv8 and FaceNet.
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import sys
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import AttendanceDatabase
from src.utils import (
    load_config, cosine_similarity, draw_detections, non_max_suppression,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttendanceSystem:
    """Combines YOLOv8 face/ID-card detection with FaceNet recognition."""

    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.cap = None

        # Database
        self.db = AttendanceDatabase(
            self.config.get('database', {}).get('path', 'attendance.db')
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # ------- Models -------
        models_cfg = self.config.get('models', {})

        # Face detector (YOLOv8) - Not needed anymore as unified model handles it
        # self.face_detector = YOLO('yolov8n.pt')

        # Directories
        self.proof_dir = 'attendance_records'
        os.makedirs(self.proof_dir, exist_ok=True)
        self.last_frame = None
        self.last_db_mark = {} # student_id -> last_marked_time (2s throttle)
        self.last_seen_time = {} # student_id -> status_bar_time

        # ID card detector (YOLOv8, fine-tuned or base)
        id_model = models_cfg.get('id_card_detector', 'models/id_card_detector.pt')
        if os.path.exists(id_model):
            self.id_card_detector = YOLO(id_model)
            logger.info(f"ID card detector loaded: {id_model}")
        else:
            self.id_card_detector = YOLO('yolov8n.pt')
            logger.warning("Using base YOLOv8 for ID-card detection (not fine-tuned)")

        # FaceNet
        from facenet_pytorch import InceptionResnetV1
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device)
        self.facenet.eval()
        logger.info("FaceNet model loaded")

        # Load student embeddings
        self.students = self.db.get_student_embeddings()
        logger.info(f"Loaded {len(self.students)} student embeddings")

        # Thresholds
        recog_cfg = self.config.get('recognition', {})
        self.threshold = recog_cfg.get('similarity_threshold', 0.55)

        # Camera config
        self.cam_cfg = self.config.get('camera', {})

        # Session stats
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'recognitions': 0,
            'id_cards_detected': 0,
            'attendance_marked': 0,
        }

        logger.info("Attendance system initialised ✓")

    def _open_camera(self):
        """Internal method to open the camera."""
        if self.cap is not None and self.cap.isOpened():
            return

        dev_id = self.cam_cfg.get('device_id', 0)
        self.cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            # Fallback to default if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(dev_id)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.get('width', 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.get('height', 720))
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_cfg.get('fps', 30))

        if not self.cap.isOpened():
            logger.error("Could not open camera.")

    def _close_camera(self):
        """Internal method to close the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def __enter__(self):
        self._open_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

        # Session stats
        self.stats = {
            'total_frames': 0,
            'faces_detected': 0,
            'recognitions': 0,
            'id_cards_detected': 0,
            'attendance_marked': 0,
        }

        logger.info("Attendance system initialised ✓")

    # ------------------------------------------------------------------
    # Embedding / recognition
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_face_embedding(self, face_bgr):
        """Extract a 512-d FaceNet embedding from a BGR face crop."""
        face = cv2.resize(face_bgr, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(face).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(self.device)
        return self.facenet(tensor).cpu().numpy().flatten()

    def recognize_face(self, embedding):
        """Return (student_dict | None, similarity)."""
        if not self.students:
            return None, 0.0

        best, best_sim = None, 0.0
        for s in self.students:
            sim = cosine_similarity(embedding, s['embedding'])
            if sim > best_sim:
                best_sim = sim
                best = s

        if best_sim >= self.threshold:
            return best, best_sim
        return None, best_sim

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame):
        """Detect faces + ID cards using the unified YOLOv8 model with tight face crops."""
        self.last_frame = frame.copy()
        detections = []

        # 1. Detection Stage (Unified Model)
        results = self.id_card_detector(frame, conf=0.25, verbose=False)
        names = self.id_card_detector.names
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            raw_label = names.get(cls_idx, 'id_obj')
            
            # Rewrite label: 'lanyard' -> 'tag'
            label = 'tag' if raw_label == 'lanyard' else raw_label

            # --- Handle Face Recognition (Tight Crop) ---
            if label == 'face':
                # Tighten the face crop by 15% to remove background noise/hair
                w, h = x2 - x1, y2 - y1
                margin_x, margin_y = int(w * 0.15), int(h * 0.15)
                tx1, ty1 = max(0, x1 + margin_x), max(0, y1 + margin_y)
                tx2, ty2 = min(frame.shape[1], x2 - margin_x), min(frame.shape[0], y2 - margin_y)
                
                face_crop = frame[ty1:ty2, tx1:tx2]
                if face_crop.size == 0: continue

                emb = self.extract_face_embedding(face_crop)
                match, sim = self.recognize_face(emb)

                det = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'type': 'face',
                    'label': 'face'
                }

                if match:
                    det.update({
                        'name': match['name'],
                        'student_id': match['student_id'],
                        'similarity': sim,
                    })
                    self.stats['recognitions'] += 1
                else:
                    det.update({'name': 'Unknown', 'similarity': sim})

                detections.append(det)
                self.stats['faces_detected'] += 1

            # --- Handle ID Card and Tag ---
            elif label in ['idcard', 'tag', 'id_card']:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'label': 'tag' if label in ['lanyard', 'tag'] else 'idcard',
                    'type': 'id_obj',
                })
                if label == 'idcard' or label == 'id_card':
                    self.stats['id_cards_detected'] += 1

        return non_max_suppression(detections)

    def is_wearing_id(self, face_det, detections):
        """
        Check if an ID card and Tag are associated with this SPECIFIC face.
        Uses a tight horizontal center-lock and closest-face logic.
        """
        fx1, fy1, fx2, fy2 = face_det['bbox']
        face_center_x = (fx1 + fx2) / 2
        face_bottom_y = fy2
        
        # Relaxed margin (1.2x face width) to handle off-center lanyards
        margin = (fx2 - fx1) * 1.2 
        
        # Get all other faces in the frame for "Closest Face" check
        other_faces = [d for d in detections if d.get('type') == 'face' and d['bbox'] != face_det['bbox']]
        
        # Look for cards/tags in the face's vertical corridor
        matching_cards = []
        matching_tags = []
        
        face_height = fy2 - fy1
        
        for d in detections:
            if d.get('type') != 'id_obj': continue
            dx1, dy1, dx2, dy2 = d['bbox']
            d_center_x = (dx1 + dx2) / 2
            d_top_y = dy1
            
            # --- Strict Spatial Guarding ---
            # 1. Horizontal Alignment: Must be within the margin
            dist_x = abs(d_center_x - face_center_x)
            is_aligned = dist_x < margin
            
            # 2. Vertical Proximity: Allow overlap (lanyards can be high) and increase range
            dist_y = d_top_y - face_bottom_y
            # Allow overlap up to 1 face height, and range up to 5 face heights
            is_proximate = (-1.0 * face_height) <= dist_y < (face_height * 5.0)
            
            # 3. Closest Face Check (2D distance with horizontal penalty)
            # We penalty horizontal distance because lanyards are primarily vertical
            h_penalty = 2.0
            dist_to_this_2d = np.sqrt((dist_x * h_penalty)**2 + (max(0, dist_y))**2)
            
            is_closest = True
            for other in other_faces:
                ox1, oy1, ox2, oy2 = other['bbox']
                other_face_center_x = (ox1 + ox2) / 2
                other_face_bottom_y = oy2
                
                odist_x = abs(d_center_x - other_face_center_x)
                odist_y = d_top_y - other_face_bottom_y
                dist_to_other_2d = np.sqrt((odist_x * h_penalty)**2 + (max(0, odist_y))**2)
                
                if dist_to_other_2d < dist_to_this_2d:
                    is_closest = False
                    break
            
            if is_aligned and is_proximate and is_closest:
                label_lower = d.get('label', '').lower()
                if label_lower in ['id_card', 'idcard', 'card']:
                    matching_cards.append(d)
                elif label_lower == 'tag':
                    matching_tags.append(d)
            else:
                # ENHANCED Debug logging for spatial failures
                logger.debug(f"Spatial Fail: {face_det.get('name', 'Unknown')} -> {d['label']} | "
                             f"Aligned({is_aligned}): dx={dist_x:.1f} marg={margin:.1f} | "
                             f"Proximate({is_proximate}): dy={dist_y:.1f} lim=[-{face_height:.1f}, {face_height*5.0:.1f}] | "
                             f"Closest({is_closest})")
                    
        # Decoupled Verification: Return both independently
        has_tag = len(matching_tags) > 0
        has_card = len(matching_cards) > 0
        
        # Add owner tag if any match
        if has_tag or has_card:
            owner_name = face_det.get('name', 'Unknown')
            for obj in matching_cards + matching_tags:
                obj['owner'] = owner_name
                
        return has_tag, has_card

    # ------------------------------------------------------------------
    # Attendance logic
    # ------------------------------------------------------------------

    def _try_mark(self, det, all_detections):
        """Mark attendance for recognized faces with status tracking (0:None, 1:Tag, 2:Tag+Card)."""
        if det.get('type') != 'face' or 'student_id' not in det:
            return False

        # Determine verification bits (Bit 0: Card, Bit 1: Tag)
        has_tag, has_card = self.is_wearing_id(det, all_detections)
        
        # Encode: 0:None, 1:Card, 2:Tag, 3:Both
        v_level = (2 if has_tag else 0) | (1 if has_card else 0)

        now = datetime.now()
        
        # Database Throttle: Update DB every 2 seconds to stay responsive without spamming
        last_t = self.last_db_mark.get(det['student_id'])
        if last_t and (now - last_t).total_seconds() < 2:
            return False

        # Mark in Database (v_level 0, 1, or 2)
        ok = self.db.mark_attendance(
            det['student_id'], det['similarity'], v_level
        )
        
        if ok:
            self.last_db_mark[det['student_id']] = now
            self.stats['attendance_marked'] += 1
            
            # Save proof image
            ts = now.strftime("%Y%m%d_%H%M%S")
            proof_path = os.path.join(self.proof_dir, f"{det['name']}_{ts}.jpg")
            cv2.imwrite(proof_path, self.last_frame) 
            
            status_map = {0: "Face only", 1: "Face+Card", 2: "Face+Tag", 3: "Face+Tag+Card"}
            logger.info(f"✓ Attendance marked for {det['name']} ({status_map[v_level]}) at {now.strftime('%H:%M:%S')}")
        return ok

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start the real-time attendance loop.  Press 'q' to quit."""
        self._open_camera()
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera not available.")
            return

        logger.info("Starting attendance system — press 'q' to quit, 's' for stats")

        frame_count = 0
        process_every = 2  # skip every other frame for speed
        detections = []

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Camera read failed - resource may be busy or disconnected")
                # Show error on a black frame instead of just exiting
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "CAMERA ERROR: Resource Busy?", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(error_frame, "Please close other camera windows.", (50, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow('Lab Attendance System', error_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'): break
                continue
            
            # Check for all-black frame (common when camera is partially locked)
            if np.mean(frame) < 1.0:
                cv2.putText(frame, "WAITING FOR CAMERA DATA...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow('Lab Attendance System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            frame_count += 1
            self.stats['total_frames'] += 1

            if frame_count % process_every == 0:
                detections = self.process_frame(frame)

            # ---- Draw on display copy ----
            display = frame.copy()

            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                if det['type'] == 'face':
                    if 'name' in det:
                        sim = det.get('similarity', 0)
                        last_t = self.last_seen_time.get(det.get('student_id', ''), '')
                        time_label = f" | Seen: {last_t}" if last_t else ""
                        label = f"{det['name']} ({sim:.2f}){time_label}"
                        color = (0, 255, 0)
                        self._try_mark(det, detections)
                    else:
                        color = (0, 0, 255)
                        label = f"Unknown ({det.get('similarity', 0):.2f})"
                else:
                    is_tag = det.get('label', '').lower() == 'tag'
                    color = (255, 105, 180) if is_tag else (255, 165, 0)
                    
                    # Add student name to ID label if verified
                    name_suffix = f" [{det['owner']}]" if 'owner' in det else ""
                    label = f"{det['label'].replace('_', ' ').title()}{name_suffix} ({det['confidence']:.2f})"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(display, (x1, y1 - th - 10),
                              (x1 + tw, y1), color, -1)
                cv2.putText(display, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Stats overlay
            for i, line in enumerate([
                f"Faces: {self.stats['faces_detected']}",
                f"ID Cards: {self.stats['id_cards_detected']}",
                f"Marked: {self.stats['attendance_marked']}",
            ]):
                cv2.putText(display, line, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Lab Attendance System', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._print_stats()

        self._print_stats()
        self.cleanup()

    # ------------------------------------------------------------------

    def _print_stats(self):
        print(f"\n{'='*40}")
        print("  SESSION STATISTICS")
        print(f"{'='*40}")
        for k, v in self.stats.items():
            print(f"  {k.replace('_', ' ').title()}: {v}")
        today = self.db.get_todays_attendance()
        print(f"\n  Today's attendance: {len(today)} students")
        for r in today[:5]:
            print(f"    {r['name']}: {r['timestamp']}")
        if len(today) > 5:
            print(f"    … and {len(today) - 5} more")
        print()

    def cleanup(self):
        self._close_camera()
        if hasattr(self, 'db'):
            self.db.close()
        logger.info("Attendance system stopped")

    def __del__(self):
        self.cleanup()


# ======================================================================

def main():
    with AttendanceSystem() as system:
        try:
            system.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")


if __name__ == '__main__':
    main()
