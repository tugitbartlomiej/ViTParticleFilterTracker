"""
EL2N (Error L2-Norm) Scorer for Dataset Selection.

Computes difficulty scores for samples based on prediction error magnitude.
Higher EL2N = harder sample (model is more uncertain).

Reference: "Deep Learning on a Data Diet" (Paul et al., 2021)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleProxyModel(nn.Module):
    """Simple CNN proxy model for EL2N computation."""

    def __init__(self, num_classes: int = 2, input_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImageDataset(Dataset):
    """Simple image dataset for proxy training."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import cv2
        from PIL import Image

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label, idx


class EL2NScorer:
    """Computes EL2N difficulty scores for dataset samples."""

    def __init__(self,
                 num_classes: int = 2,
                 proxy_epochs: int = 20,
                 batch_size: int = 16,
                 learning_rate: float = 0.001,
                 device: str = "cuda",
                 aggregation: str = "mean"):
        """
        Initialize EL2N Scorer.

        Args:
            num_classes: Number of classes
            proxy_epochs: Number of epochs to train proxy model
            batch_size: Batch size for training
            learning_rate: Learning rate for proxy training
            device: Device to run on ('cuda' or 'cpu')
            aggregation: How to aggregate per-sample scores ('mean' or 'max')
        """
        self.num_classes = num_classes
        self.proxy_epochs = proxy_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else "cpu"
        self.aggregation = aggregation

        self.proxy_model = None
        self.scores = {}

        # Transform for preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def train_proxy_model(self,
                          image_paths: List[str],
                          labels: List[int],
                          show_progress: bool = True) -> nn.Module:
        """
        Train a simple proxy model on the dataset.

        Args:
            image_paths: List of image paths
            labels: List of labels (0 or 1 for binary)
            show_progress: Whether to show progress

        Returns:
            Trained proxy model
        """
        logger.info(f"Training proxy model for {self.proxy_epochs} epochs...")

        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, labels, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=True, num_workers=0)

        # Initialize model
        self.proxy_model = SimpleProxyModel(num_classes=self.num_classes)
        self.proxy_model.to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.proxy_model.parameters(),
                                     lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.proxy_model.train()
        for epoch in range(self.proxy_epochs):
            total_loss = 0
            correct = 0
            total = 0

            iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.proxy_epochs}",
                           disable=not show_progress)

            for images, labels_batch, _ in iterator:
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.proxy_model(images)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels_batch.size(0)
                correct += predicted.eq(labels_batch).sum().item()

            acc = 100. * correct / total
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

        return self.proxy_model

    def compute_el2n_scores(self,
                            image_paths: List[str],
                            labels: List[int],
                            show_progress: bool = True) -> Dict[str, float]:
        """
        Compute EL2N scores for all samples.

        EL2N Score = ||softmax(logits) - one_hot(label)||_2

        Args:
            image_paths: List of image paths
            labels: List of labels
            show_progress: Whether to show progress

        Returns:
            Dictionary mapping image paths to EL2N scores
        """
        if self.proxy_model is None:
            self.train_proxy_model(image_paths, labels, show_progress)

        logger.info("Computing EL2N scores...")

        # Create dataset
        dataset = ImageDataset(image_paths, labels, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=0)

        self.proxy_model.eval()
        self.scores = {}

        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Computing EL2N",
                           disable=not show_progress)

            for images, labels_batch, indices in iterator:
                images = images.to(self.device)
                labels_batch = labels_batch.to(self.device)

                # Get predictions
                outputs = self.proxy_model(images)
                probs = F.softmax(outputs, dim=1)

                # Create one-hot targets
                one_hot = F.one_hot(labels_batch, num_classes=self.num_classes).float()

                # Compute EL2N: ||probs - one_hot||_2
                el2n = torch.norm(probs - one_hot, p=2, dim=1)

                # Store scores
                for i, idx in enumerate(indices):
                    path = image_paths[idx]
                    self.scores[path] = float(el2n[i].cpu())

        logger.info(f"Computed EL2N scores for {len(self.scores)} samples")
        return self.scores

    def compute_el2n_for_detection(self,
                                   predictions: Dict[str, Dict],
                                   targets: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute EL2N-like scores for object detection.

        For detection, we use:
        - Classification confidence as proxy for difficulty
        - Localization error (IoU) as additional signal

        Args:
            predictions: Dict of image_path -> {'boxes': [...], 'scores': [...], 'labels': [...]}
            targets: Dict of image_path -> {'boxes': [...], 'labels': [...]}

        Returns:
            Dictionary mapping image paths to difficulty scores
        """
        scores = {}

        for image_path in predictions.keys():
            pred = predictions[image_path]
            target = targets.get(image_path, {'boxes': [], 'labels': []})

            if not pred['scores']:
                # No predictions = potentially hard sample
                scores[image_path] = 1.0
                continue

            # Use inverse of max confidence as difficulty
            max_conf = max(pred['scores']) if pred['scores'] else 0
            conf_difficulty = 1 - max_conf

            # Compute average confidence
            avg_conf = np.mean(pred['scores']) if pred['scores'] else 0
            avg_difficulty = 1 - avg_conf

            # Combined score
            if self.aggregation == "max":
                scores[image_path] = float(max(conf_difficulty, avg_difficulty))
            else:
                scores[image_path] = float((conf_difficulty + avg_difficulty) / 2)

        return scores

    def rank_by_difficulty(self,
                           scores: Optional[Dict[str, float]] = None,
                           keep_hard: bool = True,
                           top_k: Optional[int] = None) -> List[str]:
        """
        Rank samples by difficulty.

        Args:
            scores: Difficulty scores (uses self.scores if None)
            keep_hard: If True, return hardest samples first
            top_k: Return only top-k samples

        Returns:
            List of image paths sorted by difficulty
        """
        if scores is None:
            scores = self.scores

        if not scores:
            logger.warning("No scores available")
            return []

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: x[1],
                             reverse=keep_hard)

        paths = [item[0] for item in sorted_items]

        if top_k is not None:
            paths = paths[:top_k]

        return paths

    def get_score_statistics(self,
                            scores: Optional[Dict[str, float]] = None) -> Dict:
        """Get statistics about the scores."""
        if scores is None:
            scores = self.scores

        if not scores:
            return {}

        values = list(scores.values())
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }

    def save_scores(self, output_path: str):
        """Save scores to JSON file."""
        with open(output_path, 'w') as f:
            json.dump({
                'scores': self.scores,
                'statistics': self.get_score_statistics()
            }, f, indent=2)
        logger.info(f"Saved scores to {output_path}")

    def load_scores(self, input_path: str):
        """Load scores from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        self.scores = data.get('scores', {})
        logger.info(f"Loaded {len(self.scores)} scores from {input_path}")


def compute_proxy_el2n_from_features(features: np.ndarray,
                                     labels: np.ndarray,
                                     num_epochs: int = 20) -> np.ndarray:
    """
    Compute EL2N scores using features directly (no image loading).

    Args:
        features: Feature matrix (N x D)
        labels: Label array (N,)
        num_epochs: Training epochs

    Returns:
        EL2N scores array (N,)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_classes = len(np.unique(labels))

    # Handle single-class case (unsupervised mode)
    if n_classes < 2:
        # Use feature variance as proxy for difficulty
        # Samples far from mean are "harder"
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        distances = np.linalg.norm(features_norm, axis=1)
        # Normalize to [0, 1] range
        el2n = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)
        return el2n

    # Normalize features
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Train logistic regression
    clf = LogisticRegression(max_iter=num_epochs * 100, random_state=42)
    clf.fit(features_norm, labels)

    # Get probabilities
    probs = clf.predict_proba(features_norm)

    # Compute EL2N
    one_hot = np.eye(n_classes)[labels]
    el2n = np.linalg.norm(probs - one_hot, axis=1)

    return el2n


if __name__ == "__main__":
    # Test the EL2N scorer
    print("Testing EL2NScorer...")

    # Test with synthetic features
    n_samples = 100
    n_features = 768
    n_classes = 2

    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, n_classes, n_samples)

    # Compute EL2N from features
    scores = compute_proxy_el2n_from_features(features, labels)
    print(f"EL2N scores shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"Mean score: {scores.mean():.3f}")

    # Test scorer class
    scorer = EL2NScorer(num_classes=2, device="cpu")
    stats = scorer.get_score_statistics({'a': 0.5, 'b': 0.8, 'c': 0.3})
    print(f"Statistics: {stats}")

    print("\nEL2NScorer test completed!")
