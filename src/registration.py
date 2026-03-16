"""
Student Registration Module
Capture face samples via webcam, extract embeddings, and store in the database.
"""

import cv2
import numpy as np
import os
import sys
import torch
import logging
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import AttendanceDatabase
from src.utils import load_config, preprocess_face

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentRegistration:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.cap = None
        self.db = AttendanceDatabase(
            self.config.get('database', {}).get('path', 'attendance.db')
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FaceNet for embedding extraction
        from facenet_pytorch import InceptionResnetV1
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(self.device)
        self.facenet.eval()

        # OpenCV face cascade (built-in, no extra downloads)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Camera config
        self.cam_cfg = self.config.get('camera', {})

    def _open_camera(self):
        """Internal method to open the camera."""
        if self.cap is not None and self.cap.isOpened():
            return
            
        dev_id = self.cam_cfg.get('device_id', 0)
        self.cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            # Fallback to default if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(dev_id)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.get('width', 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.get('height', 480))
        
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

    # ------------------------------------------------------------------

    def capture_face_samples(self, num_samples=10):
        """Capture multiple face samples from the webcam."""
        self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print("  ✗ Camera not available.")
            return None
            
        samples = []

        print(f"\n{'='*40}")
        print("  FACE REGISTRATION")
        print(f"{'='*40}")
        print(f"Look at the camera.  Press SPACE to capture, ESC to cancel.")
        print(f"Need {num_samples} samples.\n")

        while len(samples) < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw UI
            cv2.putText(
                frame,
                f"Samples: {len(samples)}/{num_samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Registration', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    samples.append(frame[y:y + h, x:x + w])
                    print(f"  ✓ Captured sample {len(samples)}/{num_samples}")
                else:
                    print("  ⚠ Ensure exactly ONE face is visible")
            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        return samples if len(samples) == num_samples else None

    def capture_id_samples(self, student_id, num_samples=5):
        """Capture ID card samples and auto-generate YOLO labels."""
        self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print("  ✗ Camera not available.")
            return []
            
        samples = []
        print(f"\n{'='*40}")
        print("  ID CARD SAMPLE COLLECTION")
        print(f"{'='*40}")
        print("Hold the ID card in front of the camera.")
        print("Press SPACE to capture, ESC to skip.\n")

        while len(samples) < num_samples:
            ret, frame = self.cap.read()
            if not ret: continue
            
            h, w, _ = frame.shape
            # Guide rectangle (the label)
            rw, rh = int(w * 0.5), int(h * 0.6)
            rx, ry = (w - rw) // 2, (h - rh) // 2
            
            # YOLO format: cls x_center y_center width height (normalized 0-1)
            xc, yc = (rx + rw/2)/w, (ry + rh/2)/h
            wn, hn = rw/w, rh/h
            
            display = frame.copy()
            cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
            cv2.putText(display, f"ID Samples: {len(samples)}/{num_samples}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow('Registration - ID Card', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: # SPACE
                # Store frame and normalized label
                samples.append({
                    'image': frame, 
                    'label': f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                })
                print(f"  ✓ Captured sample {len(samples)}")
            elif key == 27: # ESC
                break
                
        cv2.destroyAllWindows()
        return samples

    # ------------------------------------------------------------------

    def extract_embedding(self, face_samples):
        """Average embedding from multiple face samples."""
        embeddings = []

        for face in face_samples:
            preprocessed = preprocess_face(face)
            tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).unsqueeze(0).float()
            tensor = tensor.to(self.device)

            with torch.no_grad():
                emb = self.facenet(tensor)
            embeddings.append(emb.cpu().numpy().flatten())

        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)  # L2-normalise
        return avg

    # ------------------------------------------------------------------

    def register_new_student(self):
        """Interactive registration workflow."""
        print(f"\n{'='*40}")
        print("  NEW STUDENT REGISTRATION")
        print(f"{'='*40}\n")

        student_id = input("  Student ID : ").strip()
        name = input("  Full Name  : ").strip()
        email = input("  Email      : ").strip()

        if not student_id or not name:
            print("  ✗ Student ID and Name are required!")
            return False

        samples = self.capture_face_samples(num_samples=10)
        if samples is None:
            print("  ✗ Registration cancelled.")
            return False

        print("  Processing face samples …")
        embedding = self.extract_embedding(samples)

        success = self.db.add_student(student_id, name, email, embedding)

        if success:
            print(f"\n  ✅ Student '{name}' registered successfully!")
            
            # Save Face Samples
            face_dir = os.path.join('datasets', 'faces', 'train', f"{student_id}_{name}")
            os.makedirs(face_dir, exist_ok=True)
            for i, face in enumerate(samples):
                cv2.imwrite(os.path.join(face_dir, f"face_{i}.jpg"), face)
            
            # ── NEW: Capture & Save ID Samples ──
            # Release camera before blocking input
            self._close_camera()
            
            if input("\n  Capture ID card samples for training? (y/n): ").lower() == 'y':
                id_samples = self.capture_id_samples(student_id)
                if id_samples:
                    img_dir = os.path.join('datasets', 'id_cards', 'images', 'train')
                    lbl_dir = os.path.join('datasets', 'id_cards', 'labels', 'train')
                    os.makedirs(img_dir, exist_ok=True)
                    os.makedirs(lbl_dir, exist_ok=True)
                    
                    for i, s in enumerate(id_samples):
                        ts = int(datetime.now().timestamp())
                        fname = f"id_{student_id}_{ts}_{i}"
                        # Save Image
                        cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), s['image'])
                        # Save Label
                        with open(os.path.join(lbl_dir, f"{fname}.txt"), 'w') as f:
                            f.write(s['label'])
                            
                    print(f"  ✓ {len(id_samples)} ID samples & labels saved for training.")
        else:
            print(f"  ✗ Registration failed — ID '{student_id}' already exists.")

        return success

    # ------------------------------------------------------------------

    def test_recognition(self):
        """Quick test loop to verify face recognition against the DB."""
        print("\n  Test Recognition — press 'q' to quit\n")

        students = self.db.get_student_embeddings()
        if not students:
            print("  No students registered yet.")
        self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print("  ✗ Camera not available.")
            return

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                emb = self.extract_embedding([face])

                best_name, best_sim = 'Unknown', 0.0
                for s in students:
                    sim = float(np.dot(emb, s['embedding']))
                    if sim > best_sim:
                        best_sim = sim
                        best_name = s['name']

                threshold = self.config.get('recognition', {}).get(
                    'similarity_threshold', 0.65
                )
                if best_sim < threshold:
                    best_name = 'Unknown'

                color = (0, 255, 0) if best_name != 'Unknown' else (0, 0, 255)
                label = f"{best_name} ({best_sim:.2f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow('Test Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    # ------------------------------------------------------------------

    def cleanup(self):
        self._close_camera()
        if hasattr(self, 'db'):
            self.db.close()

    def __del__(self):
        self.cleanup()


# ======================================================================

def main():
    with StudentRegistration() as reg:
        try:
            while True:
                print(f"\n{'='*40}")
                print("  STUDENT REGISTRATION SYSTEM")
                print(f"{'='*40}")
                print("  1. Register New Student")
                print("  2. Test Recognition")
                print("  3. List All Students")
                print("  4. Exit")

                choice = input("\n  Choose (1-4): ").strip()

                if choice == '1':
                    reg.register_new_student()
                elif choice == '2':
                    reg.test_recognition()
                elif choice == '3':
                    students = reg.db.get_all_students()
                    print("\n  Registered Students:")
                    if not students:
                        print("    (none)")
                    for s in students:
                        print(f"    {s['student_id']}: {s['name']} ({s['email']})")
                elif choice == '4':
                    print("  Goodbye!")
                    break
                else:
                    print("  Invalid choice.")
        except KeyboardInterrupt:
            print("\n  Interrupted.")


if __name__ == '__main__':
    main()
