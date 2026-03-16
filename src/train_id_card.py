import os

# Create a clean environment (Disable MLflow/Comet to avoid Windows path errors)
os.environ["ULTRALYTICS_MLFLOW"] = "False"
os.environ["COMET_MODE"] = "DISABLED"

from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import yaml
import shutil
import logging

# Hard-disable external tracking to prevent Windows URI errors
SETTINGS.update({"mlflow": False, "wandb": False, "comet": False, "clearml": False, "tensorboard": False})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_id_card_yaml(dataset_dir='datasets/id_cards'):
    """Create YAML configuration for a unified ID-card dataset merging local and Roboflow data."""
    # Class mapping:
    # 0 -> id_card (local) / idcard (roboflow)
    # 1 -> tag (local) / lanyard (roboflow)
    # 2 -> face (roboflow)
    
    class_names = ['idcard', 'tag', 'face']
    
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names,
    }

    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"Created unified dataset YAML → {yaml_path}")
    return yaml_path


def merge_roboflow_dataset(roboflow_dir='datasets/roboflow_id_dataset', target_dir='datasets/id_cards'):
    """Merge Roboflow dataset into local dataset folders."""
    logger.info("Merging Roboflow dataset into main training folders...")
    
    # Map Roboflow classes (0: face, 1: idcard, 2: lanyard) to Unified (0: idcard, 1: tag, 2: face)
    # Roboflow: face=0, idcard=1, lanyard=2
    # Unified: idcard=0, tag=1, face=2
    mapping = {0: 2, 1: 0, 2: 1}
    
    for split in ['train', 'valid', 'test']:
        target_split = 'val' if split == 'valid' else split
        
        src_img_dir = os.path.join(roboflow_dir, split, 'images')
        src_lbl_dir = os.path.join(roboflow_dir, split, 'labels')
        
        dst_img_dir = os.path.join(target_dir, 'images', target_split)
        dst_lbl_dir = os.path.join(target_dir, 'labels', target_split)
        
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        if not os.path.exists(src_img_dir): continue
        
        for fname in os.listdir(src_img_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                # Copy image
                shutil.copy(os.path.join(src_img_dir, fname), os.path.join(dst_img_dir, f"rf_{fname}"))
                
                # Copy and remap labels
                lbl_fname = os.path.splitext(fname)[0] + '.txt'
                src_lbl_path = os.path.join(src_lbl_dir, lbl_fname)
                if os.path.exists(src_lbl_path):
                    with open(src_lbl_path, 'r') as f_src:
                        lines = f_src.readlines()
                    
                    with open(os.path.join(dst_lbl_dir, f"rf_{lbl_fname}"), 'w') as f_dst:
                        for line in lines:
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                new_cls_id = mapping.get(cls_id, cls_id)
                                f_dst.write(f"{new_cls_id} {' '.join(parts[1:])}\n")
    
    logger.info("Merging complete.")


def train_id_card_detector(
    dataset_dir='datasets/id_cards',
    base_model='yolov8n.pt',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu',
):
    """Train YOLOv8 for ID card detection with merged datasets"""
    
    # Merge Roboflow data if it exists
    rb_dir = 'datasets/roboflow_id_dataset'
    if os.path.exists(rb_dir):
        merge_roboflow_dataset(rb_dir, dataset_dir)

    data_yaml = create_id_card_yaml(dataset_dir)
    model = YOLO(base_model)
    
    # Force reset all callbacks to prevent MLflow/WandB errors on Windows
    model.reset_callbacks()

    args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'patience': 50,
        'device': device,
        'workers': 4,
        'optimizer': 'Adam',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'cos_lr': True,
        'close_mosaic': 10,
        'augment': True,
        'cache': True,
        'exist_ok': True,
        'project': 'id_card_training',
        'name': 'exp',
        'seed': 42,
        # Structural focus / color-blind training
        'hsv_s': 0.0,      # Set saturation to zero (grayscale effect)
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
    }

    logger.info("Starting ID card detector training (Structural Focus) …")
    results = model.train(**args)
    logger.info("Training completed!")
    return results


def validate_model(
    model_path,
    dataset_dir='datasets/id_cards',
):
    """Validate the trained model and print metrics"""
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    logger.info("Validation Results:")
    logger.info(f"  mAP50   : {metrics.box.map50:.4f}")
    logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
    return metrics


def export_best_model(
    train_results=None,
    src=None,
    dst='models/id_card_detector.pt',
):
    """Copy best weights to the models/ directory"""
    if train_results and hasattr(train_results, 'save_dir'):
        src = os.path.join(train_results.save_dir, 'weights', 'best.pt')
    
    if src is None or not os.path.exists(src):
        # Fallback to absolute system path if local detection fails
        sys_path = os.path.expanduser('~/runs/detect/id_card_training/exp/weights/best.pt')
        if os.path.exists(sys_path):
            src = sys_path
        else:
            logger.error(f"Could not find model at {src} or {sys_path}")
            return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy(src, dst)
    logger.info(f"Model saved → {dst}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train YOLOv8 ID-card detector')
    parser.add_argument('--dataset', default='datasets/id_cards')
    parser.add_argument('--base-model', default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cpu',
                        help="'cpu' or '0' for first GPU")
    args = parser.parse_args()

    results = train_id_card_detector(
        dataset_dir=args.dataset,
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )
    # Use the actual save directory from results for validation and export
    best_pt = os.path.join(results.save_dir, 'weights', 'best.pt')
    validate_model(model_path=best_pt, dataset_dir=args.dataset)
    export_best_model(train_results=results)
