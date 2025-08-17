"""
    ResNet50 + LSTM Baseline Training Script for Wildfire Smoke Detection 

    - Loads sequential image data from folders (with YOLO-format bounding box labels).
    - Crops frames around the median bounding box, resizes, normalizes, and applies transforms.
    - Uses a pretrained ResNet50 (frozen) as a frame-level feature extractor.
    - Feeds extracted features into an LSTM to model temporal dependencies.
    - Outputs binary classification (fire vs. no fire) using a linear layer.
    - Tracks accuracy, precision, recall with torchmetrics.
    - Implements a Lightning DataModule for train/val splits.
    - Saves best checkpoints (based on validation accuracy).
    - Logs metrics and training progress to Weights & Biases (wandb).
"""

import os
import glob
import random
import argparse
from datetime import datetime

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import random
import numpy as np
from torchmetrics import Accuracy, Precision, Recall
import wandb
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as T
from custom_tf import apply_transform_list

# Optional: set this if debugging CUDA launches
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define default image transformations
DEFAULT_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Import custom transform list if available
try:
    from custom_tf import apply_transform_list as custom_apply_transforms
except ImportError:
    custom_apply_transforms = None


def apply_transform_list(images, transform=None):
    """
    Apply a fixed sequence of transformations to a list of PIL Images.

    Args:
        images (List[PIL.Image]): List of images to transform.
        transform (torchvision.transforms.Compose, optional): Transform pipeline. 
            If None, uses DEFAULT_TRANSFORMS.

    Returns:
        List[torch.Tensor]: Transformed image tensors.
    """
    pipeline = transform or DEFAULT_TRANSFORMS
    # seed RNGs for reproducibility per batch
    seed = np.random.randint(0, 2**31 - 1)
    random.seed(seed)
    torch.manual_seed(seed)

    return [pipeline(img) for img in images]


class FireSeriesDataset(Dataset):
    """
    PyTorch Dataset for temporal sequences of fire images.

    Each sample is a sequence of cropped images centered on the median bounding box
    across all frames in a folder, resized and normalized.
    """

    def __init__(self, root_dir, img_size=112, transform=None):
        """
        Args:
            root_dir (str): Path to the parent folder containing subfolders of image sequences.
            img_size (int): Size to which each cropped image will be resized.
            transform (callable, optional): A function/transform to apply to each frame.
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform or DEFAULT_TRANSFORMS

        # Gather all sequence folders
        self.sequence_paths = glob.glob(os.path.join(root_dir, "**"), recursive=True)
        random.shuffle(self.sequence_paths)

    def __len__(self):
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        # Load image files for this sequence
        seq_path = self.sequence_paths[idx]
        image_files = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
        if not image_files:
            raise FileNotFoundError(f"No .jpg files in {seq_path}")

        # Read all label files to compute median bounding box
        labels = []
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            label_name = img_name.replace(".jpg", ".txt")
            label_path = os.path.join(seq_path, "labels", label_name)
            with open(label_path, "r") as lf:
                line = lf.readline().strip().split()[1:5]
            labels.append(np.array(line, dtype=float))
        labels = np.stack(labels)

        # Calculate center and size of the bounding box
        xc, yc = np.median(labels[:, :2], axis=0)
        wb, hb = np.max(labels[:, 2:], axis=0)

        # Load frames and determine crop coordinates
        frames = [Image.open(f) for f in image_files]
        w, h = frames[0].size
        crop_dim = max(wb * h, hb * h, self.img_size)
        x0 = int(xc * w - crop_dim / 2)
        y0 = int(yc * h - crop_dim / 2)
        x1, y1 = x0 + crop_dim, y0 + crop_dim

        # Crop, resize, and transform each frame
        processed = []
        for img in frames:
            cropped = img.crop((x0, y0, x1, y1))
            resized = cropped.resize((self.img_size, self.img_size))
            processed.append(resized)

        # Apply transformations
        if custom_apply_transforms:
            tensors = custom_apply_transforms(processed)
        else:
            tensors = apply_transform_list(processed, self.transform)

        # Stack into tensor shape (T, C, H, W)
        sequence_tensor = torch.stack(tensors)

        # Label extracted from parent folder name (assumes numeric)
        label = int(os.path.basename(os.path.dirname(seq_path)))
        return sequence_tensor, label


class FireDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for train/val/test splits.
    """

    def __init__(self, data_dir, batch_size=16, img_size=112, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = FireSeriesDataset(
            os.path.join(self.data_dir, 'train'),
            img_size=self.img_size
        )
        self.val_dataset = FireSeriesDataset(
            os.path.join(self.data_dir, 'val'),
            img_size=self.img_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class FireClassifier(pl.LightningModule):
    """
    LightningModule combining a frozen ResNet50 feature extractor with an LSTM classifier.
    """

    def __init__(self, learning_rate=1e-5):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained ResNet50 and freeze its weights
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # LSTM on extracted features
        self.lstm = nn.LSTM(
            input_size=resnet.fc.in_features,
            hidden_size=256,
            batch_first=True,
            dropout=0.2
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(256, 1)

        # Binary classification metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.train_precision = Precision(task='binary')
        self.val_precision = Precision(task='binary')
        self.train_recall = Recall(task='binary')
        self.val_recall = Recall(task='binary')

    def forward(self, x):
        """Forward pass returns raw logits."""
        b, t, c, h, w = x.shape
        features = x.view(b*t, c, h, w)
        feats = self.feature_extractor(features).view(b, t, -1)
        seq_out, _ = self.lstm(feats)
        last = seq_out[:, -1]
        out = self.dropout(last)
        return self.classifier(out).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc(preds, y))
        self.log('train_precision', self.train_precision(preds, y))
        self.log('train_recall', self.train_recall(preds, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.sigmoid(logits)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc(preds, y))
        self.log('val_precision', self.val_precision(preds, y))
        self.log('val_recall', self.val_recall(preds, y))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6
        )
        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser(
        description='Train a fire detection sequence model using PyTorch Lightning.'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory containing train/ and val/ subfolders')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and validation')
    parser.add_argument('--img_size', type=int, default=112,
                        help='Size to resize each frame')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader worker processes')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Initial learning rate')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum number of epochs to train')
    parser.add_argument('--wandb_project', type=str, default='fire_detection',
                        help='Weights & Biases project name')
    args = parser.parse_args()

    pl.seed_everything(42)

    # Prepare data
    dm = FireDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    # Initialize model
    model = FireClassifier(learning_rate=args.learning_rate)

    # Callbacks and logger
    checkpoint_cb = ModelCheckpoint(
        monitor='val_acc', mode='max', save_top_k=1
    )
    wandb_logger = WandbLogger(project=args.wandb_project)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_cb],
        logger=wandb_logger
    )

    # Train
    trainer.fit(model, dm)
    wandb.finish()


if __name__ == '__main__':
    main()
