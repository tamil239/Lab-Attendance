"""
Data Preparation Utilities
Organize face images and ID card datasets into train/val/test splits.
"""

import os
import shutil
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split a flat directory of images (or per-class subdirectories) into
    train / val / test folders.

    Parameters
    ----------
    source_dir : str
        Path containing images or class sub-folders.
    output_dir : str
        Destination root; will create train/, val/, test/ inside.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.
    seed : int
        Random seed for reproducibility.
    """
    random.seed(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    source = Path(source_dir)
    output = Path(output_dir)

    for split in ('train', 'val', 'test'):
        (output / split).mkdir(parents=True, exist_ok=True)

    # Check if source has subdirectories (per-class structure)
    subdirs = [d for d in source.iterdir() if d.is_dir()]

    if subdirs:
        # Per-class split
        for class_dir in subdirs:
            images = _get_image_files(class_dir)
            if not images:
                continue
            _split_and_copy(images, output, class_dir.name, train_ratio, val_ratio)
            logger.info(f"  Class '{class_dir.name}': {len(images)} images split")
    else:
        # Flat directory
        images = _get_image_files(source)
        if images:
            _split_and_copy(images, output, None, train_ratio, val_ratio)
            logger.info(f"  {len(images)} images split (flat)")

    logger.info(f"Dataset split complete → {output}")


def _get_image_files(directory):
    """Return list of image file paths in a directory"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    return [
        f for f in Path(directory).iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ]


def _split_and_copy(files, output_root, class_name, train_ratio, val_ratio):
    """Shuffle and copy files into train/val/test sub-folders"""
    random.shuffle(files)
    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:],
    }

    for split_name, split_files in splits.items():
        if class_name:
            dest = output_root / split_name / class_name
        else:
            dest = output_root / split_name
        dest.mkdir(parents=True, exist_ok=True)

        for src_file in split_files:
            shutil.copy2(str(src_file), str(dest / src_file.name))


def create_yolo_dataset_yaml(dataset_dir, output_path, class_names, nc=None):
    """
    Create a YOLO-format data.yaml for object detection training.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing images/ and labels/ folders.
    output_path : str
        Where to write the YAML file.
    class_names : list[str]
        Class names, e.g. ['id_card'].
    nc : int or None
        Number of classes (defaults to len(class_names)).
    """
    import yaml

    if nc is None:
        nc = len(class_names)

    yaml_content = {
        'path': str(Path(dataset_dir).resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': nc,
        'names': class_names,
    }

    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"Created YOLO dataset YAML → {output_path}")
    return output_path


def prepare_face_dataset(raw_dir, output_dir, min_images_per_person=5):
    """
    Prepare a face dataset from a directory of person sub-folders.
    Filters out persons with fewer than *min_images_per_person* images,
    then splits into train/val/test.
    """
    raw = Path(raw_dir)
    persons = [d for d in raw.iterdir() if d.is_dir()]

    valid_persons = []
    for person_dir in persons:
        imgs = _get_image_files(person_dir)
        if len(imgs) >= min_images_per_person:
            valid_persons.append(person_dir)
        else:
            logger.warning(
                f"Skipping '{person_dir.name}' — only {len(imgs)} images "
                f"(need >= {min_images_per_person})"
            )

    logger.info(f"Valid persons: {len(valid_persons)} / {len(persons)}")

    # Copy valid persons to a staging area, then split
    staging = Path(output_dir) / '_staging'
    staging.mkdir(parents=True, exist_ok=True)

    for pdir in valid_persons:
        dest = staging / pdir.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(pdir), str(dest))

    split_dataset(str(staging), output_dir)

    # Clean up staging
    shutil.rmtree(str(staging), ignore_errors=True)
    logger.info("Face dataset preparation complete")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--mode', choices=['faces', 'id_cards', 'split'],
                        required=True,
                        help='Type of dataset to prepare')
    parser.add_argument('--source', required=True, help='Source directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--min-images', type=int, default=5,
                        help='Min images per person (faces mode)')
    args = parser.parse_args()

    if args.mode == 'faces':
        prepare_face_dataset(args.source, args.output, args.min_images)
    elif args.mode == 'id_cards':
        create_yolo_dataset_yaml(
            args.source, os.path.join(args.output, 'data.yaml'), ['id_card']
        )
    elif args.mode == 'split':
        split_dataset(args.source, args.output)
