"""
Dedicated ID Collection Tool
Capture and auto-label ID card and tag samples for the general dataset.
"""

import cv2
import os
import sys
import logging
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IDCollector:
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        
        # Camera config
        self.cam_cfg = self.config.get('camera', {})
        self.cap = None

    def _open_camera(self):
        """Internal method to open the camera."""
        if self.cap is not None and self.cap.isOpened():
            return
            
        dev_id = self.cam_cfg.get('device_id', 0)
        self.cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev_id)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.get('width', 1280))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.get('height', 720))

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
        self._close_camera()

    def run_collection(self):
        """Interactive collection loop."""
        self._open_camera()
        if not self.cap or not self.cap.isOpened():
            print("  ✗ Camera not available.")
            return

        print(f"\n{'='*40}")
        print("  DEDICATED ID DATASET COLLECTION")
        print(f"{'='*40}")
        print("1. Position the TAG in the TOP (green) box.")
        print("2. Position the CARD in the BOTTOM (yellow) box.")
        print("Press SPACE to capture, ESC to exit.\n")

        # Directories
        img_dir = os.path.join('datasets', 'id_cards', 'images', 'train')
        lbl_dir = os.path.join('datasets', 'id_cards', 'labels', 'train')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            
            h, w, _ = frame.shape
            
            # --- Box 1: Tag (Lanyard) ---
            tw, th = int(w * 0.5), int(h * 0.3)
            tx, ty = (w - tw) // 2, int(h * 0.1)
            
            # --- Box 2: ID Card ---
            cw, ch = int(w * 0.3), int(h * 0.2)
            cx, cy = (w - cw) // 2, int(h * 0.55)

            # YOLO format labels (normalized)
            # Tag (class 1)
            tx_c, ty_c = (tx + tw/2)/w, (ty + th/2)/h
            tw_n, th_n = tw/w, th/h
            tag_label = f"1 {tx_c:.6f} {ty_c:.6f} {tw_n:.6f} {th_n:.6f}"

            # Card (class 0)
            cx_c, cy_c = (cx + cw/2)/w, (cy + ch/2)/h
            cw_n, ch_n = cw/w, ch/h
            card_label = f"0 {cx_c:.6f} {cy_c:.6f} {cw_n:.6f} {ch_n:.6f}"
            
            # UI
            display = frame.copy()
            # Tag box
            cv2.rectangle(display, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)
            cv2.putText(display, "TAG AREA", (tx, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Card box
            cv2.rectangle(display, (cx, cy), (cx+cw, cy+ch), (0, 255, 255), 2)
            cv2.putText(display, "CARD AREA", (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(display, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('ID Dataset Collection', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # SPACE
                ts = int(datetime.now().timestamp())
                fname = f"gen_id_{ts}_{count}"
                
                # Save Image
                cv2.imwrite(os.path.join(img_dir, f"{fname}.jpg"), frame)
                
                # Save Labels
                with open(os.path.join(lbl_dir, f"{fname}.txt"), 'w') as f:
                    f.write(tag_label + "\n")
                    f.write(card_label + "\n")
                
                count += 1
                print(f"  ✓ Captured sample {count}")
                
            elif key == 27: # ESC
                break

        self._close_camera()
        print(f"\nFinished. Collected {count} new samples in {img_dir}")

if __name__ == '__main__':
    with IDCollector() as collector:
        collector.run_collection()
