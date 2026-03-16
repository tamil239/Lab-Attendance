"""
Face Recognition Training Script
Fine-tunes FaceNet (InceptionResnetV1) for face recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """PyTorch dataset for face images organised in per-person sub-folders."""

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    def __init__(self, data_dir, target_size=(160, 160)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.images = []
        self.labels = []
        self.label_map = {}
        self._load_data()

    def _load_data(self):
        label_idx = 0
        for person_name in sorted(os.listdir(self.data_dir)):
            person_dir = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            self.label_map[person_name] = label_idx

            for img_file in os.listdir(person_dir):
                if os.path.splitext(img_file)[1].lower() in self.EXTENSIONS:
                    self.images.append(os.path.join(person_dir, img_file))
                    self.labels.append(label_idx)

            label_idx += 1

        logger.info(
            f"Loaded {len(self.images)} images for "
            f"{len(self.label_map)} persons from {self.data_dir}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        # HWC → CHW, normalise to [-1, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0

        return tensor, self.labels[idx]


class FaceRecognizer:
    """Wrapper around FaceNet for training & embedding extraction."""

    def __init__(self, num_classes, device='cpu'):
        from facenet_pytorch import InceptionResnetV1

        self.device = device
        self.num_classes = num_classes

        # Pre-trained backbone
        self.backbone = InceptionResnetV1(pretrained='vggface2').to(device)

        # Freeze everything except last linear layer
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.last_linear.parameters():
            param.requires_grad = True

        # Classification head for fine-tuning
        self.classifier = nn.Linear(512, num_classes).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([
            {'params': self.backbone.last_linear.parameters(), 'lr': 1e-3},
            {'params': self.classifier.parameters(), 'lr': 1e-3},
        ])

    def forward(self, x):
        embeddings = self.backbone(x)
        return self.classifier(embeddings)

    def train_epoch(self, dataloader):
        self.backbone.train()
        self.classifier.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc='Training', leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.forward(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(dataloader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self, dataloader):
        self.backbone.eval()
        self.classifier.eval()

        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.forward(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

    @torch.no_grad()
    def extract_embedding(self, face_image):
        """Extract a 512-d embedding from a BGR face image (numpy)."""
        self.backbone.eval()

        if isinstance(face_image, np.ndarray):
            face = cv2.resize(face_image, (160, 160))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(face).permute(2, 0, 1).float()
            tensor = (tensor - 127.5) / 128.0
            tensor = tensor.unsqueeze(0).to(self.device)
        else:
            tensor = face_image.to(self.device)

        embedding = self.backbone(tensor)
        return embedding.cpu().numpy().flatten()


def train_face_recognition(
    train_dir='datasets/faces/train',
    val_dir='datasets/faces/val',
    save_path='models/face_recognizer.pth',
    epochs=50,
    batch_size=32,
):
    """Main training loop."""

    train_dataset = FaceDataset(train_dir)
    val_dataset = FaceDataset(val_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceRecognizer(num_classes=len(train_dataset.label_map), device=device)

    logger.info(f"Device: {device} | Classes: {len(train_dataset.label_map)}")

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = model.train_epoch(train_loader)
        val_acc = model.validate(val_loader)

        logger.info(
            f"Epoch {epoch}/{epochs}  "
            f"Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.1f}%  "
            f"Val Acc: {val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'backbone_state': model.backbone.state_dict(),
                'classifier_state': model.classifier.state_dict(),
                'label_map': train_dataset.label_map,
                'accuracy': best_acc,
            }, save_path)
            logger.info(f"  ✓ Best model saved ({best_acc:.1f}%)")

    logger.info("Training complete!")
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train face recognizer')
    parser.add_argument('--train-dir', default='datasets/faces/train')
    parser.add_argument('--val-dir', default='datasets/faces/val')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--save', default='models/face_recognizer.pth')
    args = parser.parse_args()

    train_face_recognition(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch,
    )
