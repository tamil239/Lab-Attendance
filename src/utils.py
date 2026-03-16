import cv2
import numpy as np
import yaml
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def preprocess_face(face_image, target_size=(160, 160)):
    """Preprocess face image for FaceNet recognition"""
    face_resized = cv2.resize(face_image, target_size)

    if len(face_resized.shape) == 2:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
    elif face_resized.shape[2] == 4:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGRA2RGB)
    elif face_resized.shape[2] == 3:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1] (FaceNet standard)
    face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0
    return face_normalized


def draw_detections(frame, detections, colors=None):
    """Draw bounding boxes and labels on frame"""
    if colors is None:
        colors = {
            'face': (0, 255, 0),
            'id_card': (255, 165, 0),
            'unknown': (0, 0, 255),
        }

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det.get('label', 'unknown')
        confidence = det.get('confidence', 0)

        if 'name' in det and det['name'] != 'Unknown':
            color = colors['face']
            text = f"{det['name']}: {confidence:.2f}"
        elif label == 'id_card':
            color = colors['id_card']
            text = f"ID Card: {confidence:.2f}"
        else:
            color = colors['unknown']
            text = f"Unknown: {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def non_max_suppression(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if not detections:
        return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        detections = [
            d for d in detections
            if calculate_iou(best['bbox'], d['bbox']) < iou_threshold
        ]

    return keep


def setup_logging(log_file='logs/attendance.log'):
    """Setup logging to file + console"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def get_timestamp():
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
