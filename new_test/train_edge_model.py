"""
Training script for EdgeAttentionUNet with Presence Head — HPC-ready.
=====================================================================

Key improvements over training_pytorch.py:
  1. Self-contained — no external library dependencies (train_and_validate)
  2. Presence-aware loss: Focal + Dice + λ·PresenceBCE
  3. Albumentations for joint image+mask augmentation
  4. AMP (mixed precision) for HPC GPU efficiency
  5. DDP (DistributedDataParallel) for multi-GPU
  6. Linear warmup + cosine decay scheduler
  7. Gradient accumulation + clipping
  8. TensorBoard logging with seg/presence metrics split
  9. Best-model checkpointing + early stopping
 10. Auto-detection of negative (background-only) patches from masks

Usage:
  Single GPU:
    python train_edge_model.py --config configs/config_edge_model_hpc.yaml

  Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_edge_model.py --config configs/config_edge_model_hpc.yaml

  Override config from CLI:
    python train_edge_model.py --config configs/config_edge_model_hpc.yaml --epochs 100 --lr 5e-4
"""

import argparse
import logging
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

# ── HPC fix: force multiprocessing temp dirs onto local disk ─────────────────
# On NFS-mounted home dirs, Python's multiprocessing creates pymp-* temp dirs
# on the network filesystem.  NFS keeps those inodes open, so rmtree fails with
# "Device or resource busy" when DataLoader workers exit — harmless but noisy.
# Redirect to /tmp (always local on HPC compute nodes) before any workers spawn.
_local_tmp = os.environ.get("TMPDIR", "/tmp")
os.environ.setdefault("TMPDIR", _local_tmp)
tempfile.tempdir = _local_tmp

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ═════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str, cli_overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Flatten CLI overrides into nested config
    override_map = {
        "epochs": ("training", "epochs"),
        "lr": ("training", "lr"),
        "batch_size": ("training", "batch_size"),
        "num_workers": ("training", "num_workers"),
        "save_dir": ("checkpointing", "save_dir"),
        "amp": ("training", "amp"),
    }
    for key, val in cli_overrides.items():
        if val is not None and key in override_map:
            section, param = override_map[key]
            cfg[section][param] = val

    return cfg


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET — Albumentations-based with auto negative detection
# ═════════════════════════════════════════════════════════════════════════════

class PatchSegmentationDataset(Dataset):
    """Patch-based segmentation dataset with albumentations transforms.

    Automatically detects negative (background-only) patches from masks.
    Returns (image, mask, has_object) — no extra annotation needed.
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Collect and sort to ensure alignment
        self.image_paths = sorted(self.image_dir.glob("*"))
        self.image_paths = [p for p in self.image_paths if p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}]

        logger.info(f"Dataset: {len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_dir / img_path.name

        # Load as numpy arrays (grayscale)
        import cv2
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        # Binarize mask
        mask = (mask > 127).astype(np.float32)

        # Albumentations: joint image+mask transform
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype(np.float32) / 255.0

        # To tensors: (H,W) → (1,H,W)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if image.ndim == 2:
                image = image.unsqueeze(0)
        elif image.ndim == 2:
            image = image.unsqueeze(0)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Ensure binary after augmentation
        mask = (mask > 0.5).float()

        # Auto-derive presence label — zero annotation cost
        has_object = (mask.sum() > 0).float()

        return image, mask, has_object


# ═════════════════════════════════════════════════════════════════════════════
# 2b. FAST TRAINING — Epoch Subsampling + Stratified Balancing + Progressive
# ═════════════════════════════════════════════════════════════════════════════

class EpochSubsamplingDataset:
    """Wrapper that presents a DIFFERENT random subset of patches each epoch.

    Expert-level strategy for large patch datasets (100k+):
      - Each epoch sees only `epoch_fraction` of total patches
      - Stratified: maintains pos/neg ratio in every subset
      - Progressive: fraction grows from `initial_fraction` to `max_fraction`
        over `ramp_epochs` — early epochs train fast on small subsets,
        later epochs see more data for fine-tuning
      - Reproducible per epoch (seeded by epoch number)

    With 111k patches at 10% per epoch:
      - Each epoch ≈ 11.1k patches = 347 batches (bs=32) instead of 3,469
      - 10× faster per epoch
      - Over 200 epochs the model sees ~2.2M patch presentations
        with different random subsets → excellent diversity

    Args:
        dataset: The full PatchSegmentationDataset
        initial_fraction: Starting fraction of patches per epoch (e.g., 0.10)
        max_fraction: Maximum fraction to grow to (e.g., 0.50)
        ramp_epochs: Number of epochs over which to linearly ramp from
                     initial_fraction to max_fraction
        pos_neg_ratio: Target ratio of positive:total in each subset.
                       None = maintain natural ratio from full dataset.
                       0.5 = force 50/50 balance.
        seed: Base random seed for reproducibility
    """

    def __init__(self, dataset, initial_fraction=0.10, max_fraction=0.50,
                 ramp_epochs=50, pos_neg_ratio=None, seed=42):
        self.dataset = dataset
        self.initial_fraction = initial_fraction
        self.max_fraction = max_fraction
        self.ramp_epochs = ramp_epochs
        self.pos_neg_ratio = pos_neg_ratio
        self.seed = seed

        # ── Pre-scan masks to classify pos/neg ──────────────────────────────
        # Results are cached to a .json sidecar next to the mask dir so that
        # subsequent runs (and DDP worker ranks) skip the expensive NFS scan.
        import cv2
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        cache_path = dataset.mask_dir.parent / f".pos_neg_cache_{dataset.mask_dir.name}.json"

        if cache_path.exists():
            logger.info(f"Loading pos/neg cache from {cache_path} ...")
            with open(cache_path) as f:
                cache = json.load(f)
            self.pos_indices = cache["pos"]
            self.neg_indices = cache["neg"]
            logger.info(
                f"Cache loaded: {len(self.pos_indices)} pos, {len(self.neg_indices)} neg"
            )
        else:
            logger.info(
                f"Pre-scanning {len(dataset)} masks (parallel, 16 threads) ..."
            )

            def _check(i):
                mask_path = dataset.mask_dir / dataset.image_paths[i].name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                return i, (mask is not None and mask.max() > 127)

            results = [None] * len(dataset)
            with ThreadPoolExecutor(max_workers=16) as pool:
                for future in as_completed(
                    pool.submit(_check, i) for i in range(len(dataset))
                ):
                    idx, is_pos = future.result()
                    results[idx] = is_pos

            self.pos_indices = [i for i, p in enumerate(results) if p]
            self.neg_indices = [i for i, p in enumerate(results) if not p]

            # Save cache so next run is instant
            try:
                with open(cache_path, "w") as f:
                    json.dump({"pos": self.pos_indices, "neg": self.neg_indices}, f)
                logger.info(f"Pos/neg cache saved → {cache_path}")
            except OSError as e:
                logger.warning(f"Could not save pos/neg cache: {e}")

        n_pos = len(self.pos_indices)
        n_neg = len(self.neg_indices)
        n_total = len(dataset)
        logger.info(
            f"Pre-scan complete: {n_total} patches "
            f"({n_pos} positive [{n_pos/n_total:.1%}], "
            f"{n_neg} negative [{n_neg/n_total:.1%}])"
        )

        # Initialize with epoch 0
        self._current_indices = list(range(len(dataset)))
        self.set_epoch(0)

    def _get_fraction(self, epoch):
        """Linearly ramp fraction from initial to max over ramp_epochs."""
        if epoch >= self.ramp_epochs:
            return self.max_fraction
        t = epoch / max(self.ramp_epochs, 1)
        return self.initial_fraction + t * (self.max_fraction - self.initial_fraction)

    def set_epoch(self, epoch):
        """Resample a new subset for this epoch. Call before each epoch."""
        fraction = self._get_fraction(epoch)
        rng = random.Random(self.seed + epoch)

        if self.pos_neg_ratio is not None:
            # ── Stratified sampling with target ratio ──
            n_total_subset = int(len(self.dataset) * fraction)
            n_pos_want = int(n_total_subset * self.pos_neg_ratio)
            n_neg_want = n_total_subset - n_pos_want

            # Clamp to available
            n_pos_want = min(n_pos_want, len(self.pos_indices))
            n_neg_want = min(n_neg_want, len(self.neg_indices))

            pos_sample = rng.sample(self.pos_indices, n_pos_want)
            neg_sample = rng.sample(self.neg_indices, n_neg_want)
            self._current_indices = pos_sample + neg_sample
        else:
            # ── Random subset maintaining natural ratio ──
            n_pos_want = int(len(self.pos_indices) * fraction)
            n_neg_want = int(len(self.neg_indices) * fraction)

            n_pos_want = min(n_pos_want, len(self.pos_indices))
            n_neg_want = min(n_neg_want, len(self.neg_indices))

            pos_sample = rng.sample(self.pos_indices, n_pos_want)
            neg_sample = rng.sample(self.neg_indices, n_neg_want)
            self._current_indices = pos_sample + neg_sample

        rng.shuffle(self._current_indices)
        n_pos_sampled = len(pos_sample)
        n_neg_sampled = len(neg_sample)
        logger.info(
            f"Epoch {epoch}: subsampled {len(self._current_indices)}/{len(self.dataset)} "
            f"patches ({fraction:.0%}) — {n_pos_sampled} pos, {n_neg_sampled} neg"
        )

    def __len__(self):
        return len(self._current_indices)

    def __getitem__(self, idx):
        real_idx = self._current_indices[idx]
        return self.dataset[real_idx]

    @property
    def image_paths(self):
        """Proxy for compatibility."""
        return [self.dataset.image_paths[i] for i in self._current_indices]


class CutMixSegmentation:
    """CutMix augmentation adapted for segmentation.

    Mixes two (image, mask) pairs by pasting a random rectangle from one
    onto the other. Both the image AND the mask are mixed consistently.
    This provides:
      - Stronger regularization (reduces overfitting on 111k patches)
      - Synthetic diversity without extra data
      - Forces the model to handle partial objects

    Usage: Apply at the batch level in the training loop.
    """

    def __init__(self, p=0.5, alpha=1.0):
        self.p = p
        self.alpha = alpha

    def __call__(self, images, masks, has_object):
        """Apply CutMix to a batch.

        Args:
            images: (B, C, H, W) tensor
            masks: (B, 1, H, W) tensor
            has_object: (B,) tensor

        Returns:
            Mixed images, masks, has_object
        """
        if random.random() > self.p:
            return images, masks, has_object

        B, C, H, W = images.shape
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation for pairing
        indices = torch.randperm(B, device=images.device)

        # Random bounding box
        cut_ratio = math.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cy = random.randint(0, H)
        cx = random.randint(0, W)

        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        # Mix both image and mask consistently
        images_mixed = images.clone()
        masks_mixed = masks.clone()

        images_mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        masks_mixed[:, :, y1:y2, x1:x2] = masks[indices, :, y1:y2, x1:x2]

        # Update presence label: object present if either original or pasted has object
        has_object_mixed = torch.clamp(has_object + has_object[indices], max=1.0)
        # Re-derive from actual mixed mask (more accurate)
        has_object_mixed = (masks_mixed.sum(dim=(1, 2, 3)) > 0).float()

        return images_mixed, masks_mixed, has_object_mixed


def build_augmentations(cfg: dict, train: bool = True):
    """Build albumentations pipeline from config."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    h = cfg["dataset"]["image_height"]
    w = cfg["dataset"]["image_width"]
    aug_cfg = cfg.get("augmentation", {})

    if train:
        transform = A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=aug_cfg.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=aug_cfg.get("vertical_flip_p", 0.5)),
            A.Rotate(
                limit=aug_cfg.get("rotate_limit", 15),
                p=aug_cfg.get("rotate_p", 0.5),
                border_mode=0,   # cv2.BORDER_CONSTANT
            ),
            A.Affine(
                translate_percent={
                    "x": (-aug_cfg.get("shift_limit", 0.1), aug_cfg.get("shift_limit", 0.1)),
                    "y": (-aug_cfg.get("shift_limit", 0.1), aug_cfg.get("shift_limit", 0.1)),
                },
                scale=(
                    1 - aug_cfg.get("scale_limit", 0.1),
                    1 + aug_cfg.get("scale_limit", 0.1),
                ),
                rotate=0,           # rotation handled above
                border_mode=0,      # cv2.BORDER_CONSTANT
                p=aug_cfg.get("shift_scale_rotate_p", 0.7),
            ),
            A.GaussNoise(
                std_range=aug_cfg.get("gaussian_noise_std_range", (0.1, 0.224)),
                p=aug_cfg.get("gaussian_noise_p", 0.3),
            ),
            A.GaussianBlur(
                blur_limit=aug_cfg.get("gaussian_blur_limit", (3, 5)),
                p=aug_cfg.get("gaussian_blur_p", 0.2),
            ),
            A.RandomBrightnessContrast(
                p=aug_cfg.get("brightness_contrast_p", 0.3),
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                p=aug_cfg.get("elastic_transform_p", 0.1),
            ),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    return transform


# ═════════════════════════════════════════════════════════════════════════════
# 3. LOSSES — Focal + Dice + Presence BCE
# ═════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) — downweights easy negatives."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Soft Dice Loss — optimizes spatial overlap directly.

    smooth term prevents 0/0 on all-background patches.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        pred = torch.sigmoid(logits)
        intersection = (pred * targets).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLossWithPresence(nn.Module):
    """Combined loss: FocalLoss + DiceLoss + λ * PresenceBCE.

    Presence label is auto-derived from mask (has_object = mask.sum() > 0).
    """

    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, dice_smooth=1.0,
                 presence_weight=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=dice_smooth)
        self.presence_weight = presence_weight

    def forward(self, seg_logits, presence_logit, masks, has_object):
        """
        Args:
            seg_logits: (B, 1, H, W) raw segmentation logits
            presence_logit: (B, 1) raw presence logit
            masks: (B, 1, H, W) binary ground truth
            has_object: (B,) float 0/1 presence label

        Returns:
            dict with total_loss, seg_loss, presence_loss
        """
        focal_loss = self.focal(seg_logits, masks)
        dice_loss = self.dice(seg_logits, masks)
        seg_loss = focal_loss + dice_loss

        presence_target = has_object.unsqueeze(1)  # (B,) → (B, 1)
        presence_loss = F.binary_cross_entropy_with_logits(
            presence_logit, presence_target
        )

        total_loss = seg_loss + self.presence_weight * presence_loss

        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "focal_loss": focal_loss,
            "dice_loss": dice_loss,
            "presence_loss": presence_loss,
        }


# ═════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(seg_logits, presence_logit, masks, has_object, threshold=0.5):
    """Compute segmentation + presence metrics.

    Returns:
        dict with dice, iou, presence_acc, presence_f1, neg_fp_rate
    """
    pred = (torch.sigmoid(seg_logits) > threshold).float()

    # ── Segmentation metrics (only on positive patches) ──
    pos_mask = has_object.bool()
    if pos_mask.any():
        p = pred[pos_mask]
        t = masks[pos_mask]
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    else:
        dice = torch.tensor(0.0)
        iou = torch.tensor(0.0)

    # ── Presence metrics ──
    presence_pred = (presence_logit.squeeze(1) > 0).float()
    presence_acc = (presence_pred == has_object).float().mean()

    # Presence precision / recall / F1
    tp = ((presence_pred == 1) & (has_object == 1)).sum().float()
    fp = ((presence_pred == 1) & (has_object == 0)).sum().float()
    fn = ((presence_pred == 0) & (has_object == 1)).sum().float()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    presence_f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # ── False positive rate on negative patches ──
    # (do predictions fire on empty patches?)
    neg_mask = ~pos_mask
    if neg_mask.any():
        neg_fp_rate = (pred[neg_mask].sum() / pred[neg_mask].numel()).item()
    else:
        neg_fp_rate = 0.0

    return {
        "dice": dice.item(),
        "iou": iou.item(),
        "presence_acc": presence_acc.item(),
        "presence_f1": presence_f1.item(),
        "neg_fp_rate": neg_fp_rate,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. SCHEDULER WITH WARMUP
# ═════════════════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing to min_lr."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6,
                 last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            factor = step / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs
        ]


# ═════════════════════════════════════════════════════════════════════════════
# 6. EARLY STOPPING
# ═════════════════════════════════════════════════════════════════════════════

def _monitor_to_metric_key(monitor_name: str) -> str:
    """Map config monitor (e.g. val_dice, val_iou, val_loss) to val_metrics key."""
    key = monitor_name.replace("val_", "", 1) if monitor_name.startswith("val_") else monitor_name
    return {"loss": "total_loss"}.get(key, key)


def _monitor_mode(monitor_name: str) -> str:
    """Return 'min' for metrics where lower is better, 'max' otherwise."""
    return "min" if ("loss" in monitor_name or "neg_fp" in monitor_name) else "max"


class EarlyStopping:
    """Early stopping on a monitored metric."""

    def __init__(self, patience=25, min_delta=0.001, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metric) -> bool:
        """Returns True if training should stop."""
        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ═════════════════════════════════════════════════════════════════════════════
# 7. CHECKPOINT MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """Save top-K checkpoints by monitored metric."""

    def __init__(self, save_dir, save_top_k=3, monitor="val_dice", mode="max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.best_scores = []  # list of (score, path)

    def save_if_best(self, model, optimizer, scheduler, epoch, metric_value,
                     extra_info=None):
        """Save checkpoint if it's among the top-K best."""
        # Check if this is a top-K score
        is_top_k = len(self.best_scores) < self.save_top_k
        if not is_top_k:
            worst = min(self.best_scores, key=lambda x: x[1]) if self.mode == "max" \
                else max(self.best_scores, key=lambda x: x[0])
            if self.mode == "max":
                is_top_k = metric_value > worst[0]
            else:
                is_top_k = metric_value < worst[0]

        if not is_top_k:
            return False

        # Save checkpoint
        timestamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
        fname = f"epoch_{epoch:03d}_{self.monitor}_{metric_value:.4f}_{timestamp}.pt"
        path = self.save_dir / fname

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict() if not isinstance(model, DDP) \
                else model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            self.monitor: metric_value,
        }
        if extra_info:
            state.update(extra_info)

        torch.save(state, path)
        logger.info(f"  Saved checkpoint: {fname}")

        self.best_scores.append((metric_value, path))

        # Remove worst if over limit
        if len(self.best_scores) > self.save_top_k:
            if self.mode == "max":
                worst = min(self.best_scores, key=lambda x: x[0])
            else:
                worst = max(self.best_scores, key=lambda x: x[0])
            self.best_scores.remove(worst)
            if worst[1].exists():
                worst[1].unlink()
                logger.info(f"  Removed old checkpoint: {worst[1].name}")

        return True


# ═════════════════════════════════════════════════════════════════════════════
# 8. TRAINING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _set_bn_training(model, mode: bool):
    """Set only BatchNorm layers to train/eval mode (freeze/unfreeze running stats).

    When mode=False: BN uses running_mean/running_var (frozen stats).
    When mode=True:  BN updates running stats from batch statistics.
    This does NOT affect dropout or other layers.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.training = mode


def train_one_epoch(
    model, dataloader, criterion, optimizer, scheduler, scaler,
    device, grad_accum_steps, max_grad_norm, use_amp, epoch,
    writer=None, global_step=0, log_every=10, cutmix=None,
):
    """Train for one epoch with AMP, gradient accumulation, and logging.

    BatchNorm + gradient accumulation handling:
      When grad_accum_steps > 1, BN running stats are only updated on the
      LAST micro-step of each accumulation window. Intermediate micro-steps
      freeze BN to eval mode (running stats frozen, but gradients still flow).
      This prevents noisy running stats from tiny micro-batches.
    """
    model.train()
    running = {"total_loss": 0, "seg_loss": 0, "presence_loss": 0,
               "presence_acc": 0, "n_batches": 0}
    freeze_bn = grad_accum_steps > 1  # only freeze if actually accumulating

    for batch_idx, (images, masks, has_object) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_object = has_object.to(device, non_blocking=True)

        # ── CutMix augmentation (fast training mode) ──
        if cutmix is not None:
            images, masks, has_object = cutmix(images, masks, has_object)

        # ── BatchNorm handling for gradient accumulation ──
        # On non-final micro-steps: freeze BN running stats (eval mode)
        # On the final micro-step: unfreeze so stats get one clean update
        is_accum_boundary = (
            (batch_idx + 1) % grad_accum_steps == 0
            or (batch_idx + 1) == len(dataloader)
        )
        if freeze_bn:
            _set_bn_training(model, mode=is_accum_boundary)

        # Forward with AMP
        # In DDP with gradient accumulation, skip gradient sync on non-boundary
        # micro-steps to avoid redundant NCCL all-reduce calls.
        maybe_no_sync = model.no_sync if (
            freeze_bn and not is_accum_boundary and hasattr(model, "no_sync")
        ) else nullcontext

        with maybe_no_sync():
            with autocast(device_type="cuda", enabled=use_amp):
                seg_logits, presence_logit = model(images)
                losses = criterion(seg_logits, presence_logit, masks, has_object)
                loss = losses["total_loss"] / grad_accum_steps

            # Backward
            scaler.scale(loss).backward()

        # Step every grad_accum_steps
        if is_accum_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # GradScaler skips optimizer.step() when gradients overflow (inf/nan).
            # Track the scale to detect whether the step actually happened.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Only advance the LR schedule when weights were actually updated.
            # If scale_before > scale_after, overflow was detected and the step
            # was skipped — stepping the scheduler then triggers a spurious
            # "scheduler.step() before optimizer.step()" warning.
            if scheduler is not None and scaler.get_scale() >= scale_before:
                scheduler.step()

        # Track metrics
        with torch.no_grad():
            pres_pred = (presence_logit.squeeze(1) > 0).float()
            pres_acc = (pres_pred == has_object).float().mean()

        running["total_loss"] += losses["total_loss"].item()
        running["seg_loss"] += losses["seg_loss"].item()
        running["presence_loss"] += losses["presence_loss"].item()
        running["presence_acc"] += pres_acc.item()
        running["n_batches"] += 1

        # TensorBoard logging
        if writer and (global_step % log_every == 0):
            writer.add_scalar("train/total_loss", losses["total_loss"].item(), global_step)
            writer.add_scalar("train/seg_loss", losses["seg_loss"].item(), global_step)
            writer.add_scalar("train/focal_loss", losses["focal_loss"].item(), global_step)
            writer.add_scalar("train/dice_loss", losses["dice_loss"].item(), global_step)
            writer.add_scalar("train/presence_loss", losses["presence_loss"].item(), global_step)
            writer.add_scalar("train/presence_acc", pres_acc.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        global_step += 1

    n = running["n_batches"]
    avg = {k: running[k] / max(n, 1) for k in ["total_loss", "seg_loss",
                                                  "presence_loss", "presence_acc"]}
    return avg, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_amp):
    """Validation pass — returns averaged metrics."""
    model.eval()
    running = {"total_loss": 0, "seg_loss": 0, "presence_loss": 0,
               "dice": 0, "iou": 0, "presence_acc": 0, "presence_f1": 0,
               "neg_fp_rate": 0, "n_batches": 0}

    for images, masks, has_object in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_object = has_object.to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            seg_logits, presence_logit = model(images)
            losses = criterion(seg_logits, presence_logit, masks, has_object)

        metrics = compute_metrics(seg_logits, presence_logit, masks, has_object)

        running["total_loss"] += losses["total_loss"].item()
        running["seg_loss"] += losses["seg_loss"].item()
        running["presence_loss"] += losses["presence_loss"].item()
        running["dice"] += metrics["dice"]
        running["iou"] += metrics["iou"]
        running["presence_acc"] += metrics["presence_acc"]
        running["presence_f1"] += metrics["presence_f1"]
        running["neg_fp_rate"] += metrics["neg_fp_rate"]
        running["n_batches"] += 1

    n = running["n_batches"]
    return {k: running[k] / max(n, 1) for k in running if k != "n_batches"}


# ═════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def setup_ddp():
    """Initialize DDP if launched with torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def main():
    parser = argparse.ArgumentParser(description="Train EdgeAttentionUNet + Presence Head")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--amp", type=bool, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── DDP setup ──
    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0
    ddp = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ── Config ──
    cfg = load_config(args.config, vars(args))
    tcfg = cfg["training"]
    lcfg = cfg["loss"]
    ccfg = cfg["checkpointing"]

    if is_main:
        logger.info(f"Device: {device} | World size: {world_size}")
        logger.info(f"Config: {args.config}")

    # ── Model ──
    # Import here so the script works standalone even if nbdev export hasn't run
    try:
        from private_easy_end_test.patching.edge_segmentation_model import EdgeAttentionUNet
    except ImportError:
        logger.warning("Could not import from installed package, trying relative import...")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from private_easy_end_test.patching.edge_segmentation_model import EdgeAttentionUNet

    mcfg = cfg["model"]
    model = EdgeAttentionUNet(
        in_channels=mcfg["in_channels"],
        out_channels=mcfg["out_channels"],
        init_features=mcfg["init_features"],
        expand_ratio=mcfg["expand_ratio"],
        dropout_p=mcfg["dropout_p"],
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if is_main:
            logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    model = model.to(device)

    # ── BatchNorm safety ──
    grad_accum = tcfg.get("gradient_accumulation_steps", 1)
    micro_batch = tcfg["batch_size"]  # per-GPU micro-batch size

    if ddp:
        # SyncBatchNorm: compute BN stats across all GPUs so each BN layer
        # sees (batch_size × world_size) samples instead of just batch_size.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main:
            logger.info(f"Converted BatchNorm → SyncBatchNorm (BN sees {micro_batch}×{world_size}={micro_batch*world_size} samples)")
        model = DDP(model, device_ids=[local_rank])

    if grad_accum > 1 and micro_batch < 8:
        if is_main:
            logger.warning(
                f"⚠ gradient_accumulation_steps={grad_accum} with micro-batch={micro_batch}. "
                f"BatchNorm running stats update on every forward pass (micro-batch), not the "
                f"effective batch ({micro_batch * grad_accum}). BN stats will be noisy. "
                f"Consider: (1) increase batch_size to ≥16 and reduce accum, or "
                f"(2) the script will freeze BN stats during non-final accumulation steps."
            )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: EdgeAttentionUNet ({n_params:,} params)")

    # ── Data ──
    dcfg = cfg["dataset"]
    train_transform = build_augmentations(cfg, train=True)
    val_transform = build_augmentations(cfg, train=False)

    train_ds = PatchSegmentationDataset(dcfg["trn_im_path"], dcfg["trn_msk_path"],
                                        transform=train_transform)
    val_ds = PatchSegmentationDataset(dcfg["val_im_path"], dcfg["val_msk_path"],
                                      transform=val_transform)

    # ── Fast Training Mode (expert strategy for 100k+ patches) ──────────
    fast_cfg = cfg.get("fast_training", {})
    use_fast_training = fast_cfg.get("enabled", False)
    subsampling_ds = None
    cutmix = None

    if use_fast_training:
        if is_main:
            logger.info("=" * 75)
            logger.info("FAST TRAINING MODE — Epoch subsampling + CutMix")
            logger.info(f"  Each epoch sees {fast_cfg.get('initial_fraction', 0.10):.0%} → "
                         f"{fast_cfg.get('max_fraction', 0.30):.0%} of patches "
                         f"(ramps over {fast_cfg.get('ramp_epochs', 60)} epochs)")
            logger.info("=" * 75)

        subsampling_ds = EpochSubsamplingDataset(
            train_ds,
            initial_fraction=fast_cfg.get("initial_fraction", 0.10),
            max_fraction=fast_cfg.get("max_fraction", 0.30),
            ramp_epochs=fast_cfg.get("ramp_epochs", 60),
            pos_neg_ratio=fast_cfg.get("pos_neg_ratio"),
            seed=fast_cfg.get("seed", 42),
        )

        cutmix_cfg = fast_cfg.get("cutmix", {})
        if cutmix_cfg.get("enabled", False):
            cutmix = CutMixSegmentation(
                p=cutmix_cfg.get("p", 0.3),
                alpha=cutmix_cfg.get("alpha", 1.0),
            )
            if is_main:
                logger.info(f"CutMix enabled: p={cutmix.p}, alpha={cutmix.alpha}")

        # In fast mode: DataLoader is rebuilt each epoch (dataset length changes)
        effective_train_ds = subsampling_ds
    else:
        effective_train_ds = train_ds
        if is_main:
            logger.info("NORMAL TRAINING MODE — all patches every epoch")

    # ── DataLoaders ──
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    _pin = torch.cuda.is_available()
    val_dl = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=tcfg["num_workers"],
        pin_memory=_pin,
        persistent_workers=tcfg["num_workers"] > 0,
    )

    if not use_fast_training:
        # Normal mode: create train DataLoader once
        train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
        train_dl = DataLoader(
            train_ds,
            batch_size=tcfg["batch_size"],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=tcfg["num_workers"],
            pin_memory=_pin,
            persistent_workers=tcfg["num_workers"] > 0,
            drop_last=True,
        )
    else:
        # Fast mode: train_dl is created per-epoch (see training loop)
        train_sampler = None
        train_dl = None  # placeholder; rebuilt each epoch

    if is_main:
        if use_fast_training:
            n_pos = len(subsampling_ds.pos_indices)
            n_neg = len(subsampling_ds.neg_indices)
            logger.info(f"Train: {len(train_ds)} total patches "
                         f"({n_pos} positive [{n_pos/len(train_ds):.1%}], "
                         f"{n_neg} negative [{n_neg/len(train_ds):.1%}])")
            logger.info(f"  → Fast mode: ~{int(len(train_ds)*fast_cfg.get('initial_fraction',0.10))} "
                         f"patches/epoch initially, growing to "
                         f"~{int(len(train_ds)*fast_cfg.get('max_fraction',0.30))}")
        else:
            # Avoid iterating all patches just for logging — too slow on NFS
            logger.info(f"Train: {len(train_ds)} patches (pos/neg ratio not counted in normal mode)")
        logger.info(f"Val:   {len(val_ds)} patches")

    # ── Loss ──
    criterion = CombinedLossWithPresence(
        focal_alpha=lcfg["focal_alpha"],
        focal_gamma=lcfg["focal_gamma"],
        dice_smooth=lcfg["dice_smooth"],
        presence_weight=lcfg["presence_loss_weight"],
    )

    # ── Optimizer ──
    opt_name = tcfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                       weight_decay=tcfg["weight_decay"])
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=tcfg["lr"],
                                     momentum=0.9, weight_decay=tcfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # ── Scheduler ──
    if use_fast_training:
        # Precompute total steps accounting for ramping fraction.
        # In DDP each rank sees ceil(n_patches / world_size) samples via
        # DistributedSampler, so per-rank batches = that // batch_size.
        _total_steps = 0
        for _e in range(tcfg["epochs"]):
            _frac = subsampling_ds._get_fraction(_e)
            _n_patches = int(len(train_ds) * _frac)
            _n_per_rank = math.ceil(_n_patches / world_size) if ddp else _n_patches
            _total_steps += max(_n_per_rank // tcfg["batch_size"], 1)
        total_steps = _total_steps
        _warmup_batches = 0
        for _e in range(tcfg.get("warmup_epochs", 5)):
            _frac = subsampling_ds._get_fraction(_e)
            _n_patches = int(len(train_ds) * _frac)
            _n_per_rank = math.ceil(_n_patches / world_size) if ddp else _n_patches
            _warmup_batches += max(_n_per_rank // tcfg["batch_size"], 1)
        warmup_steps = _warmup_batches
        if is_main:
            logger.info(f"Fast mode scheduler: {total_steps} total steps, "
                         f"{warmup_steps} warmup steps")
    else:
        total_steps = tcfg["epochs"] * len(train_dl)
        warmup_steps = tcfg.get("warmup_epochs", 5) * len(train_dl)

    sched_name = tcfg.get("scheduler", "cosine_warmup")
    if sched_name == "cosine_warmup":
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
            min_lr=tcfg.get("min_lr", 1e-6),
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"], eta_min=tcfg.get("min_lr", 1e-6),
        )
    elif sched_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tcfg["lr"], total_steps=total_steps,
        )
    elif sched_name == "plateau":
        _plateau_monitor = ccfg.get("monitor", "val_dice")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=_monitor_mode(_plateau_monitor), patience=10, factor=0.5,
        )
    else:
        scheduler = None

    if args.resume and scheduler and "scheduler_state_dict" in ckpt:
        if ckpt["scheduler_state_dict"]:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    # ── AMP ──
    use_amp = tcfg.get("amp", True) and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Logging + Checkpointing ──
    writer = None
    if is_main:
        tb_dir = cfg.get("logging", {}).get("tensorboard_dir", "./runs")
        run_name = datetime.now().strftime("edge_unet_%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=f"{tb_dir}/{run_name}")
        logger.info(f"TensorBoard: {tb_dir}/{run_name}")

    monitor_name = ccfg.get("monitor", "val_dice")
    ckpt_mgr = CheckpointManager(
        save_dir=ccfg["save_dir"],
        save_top_k=ccfg.get("save_top_k", 3),
        monitor=monitor_name,
        mode=_monitor_mode(monitor_name),
    ) if is_main else None

    es_cfg = cfg.get("early_stopping", {})
    es_monitor = es_cfg.get("monitor", monitor_name)
    early_stopper = EarlyStopping(
        patience=es_cfg.get("patience", 25),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode=_monitor_mode(es_monitor),
    )

    # ═══════════════════════════════════════════════════════════
    # TRAINING LOOP
    # ═══════════════════════════════════════════════════════════
    global_step = 0
    log_every = cfg.get("logging", {}).get("log_every_n_steps", 10)
    grad_accum = tcfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)

    if is_main:
        mode_str = "FAST (subsampling)" if use_fast_training else "NORMAL (all patches)"
        logger.info("=" * 75)
        logger.info(f"Starting training [{mode_str}]: {tcfg['epochs']} epochs, "
                     f"bs={tcfg['batch_size']}, lr={tcfg['lr']}, "
                     f"AMP={use_amp}, DDP={ddp}")
        logger.info(f"Loss: Focal(α={lcfg['focal_alpha']}, γ={lcfg['focal_gamma']}) "
                     f"+ Dice(smooth={lcfg['dice_smooth']}) "
                     f"+ {lcfg['presence_loss_weight']}·PresenceBCE")
        if use_fast_training:
            logger.info(f"Fast: fraction {fast_cfg.get('initial_fraction',0.10):.0%}"
                         f"→{fast_cfg.get('max_fraction',0.30):.0%} over "
                         f"{fast_cfg.get('ramp_epochs',60)} epochs | "
                         f"CutMix={'ON' if cutmix else 'OFF'}")
        logger.info("=" * 75)

    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()

        # ── Fast mode: resample subset + rebuild DataLoader each epoch ──
        if use_fast_training:
            subsampling_ds.set_epoch(epoch)
            if ddp:
                train_sampler = DistributedSampler(
                    subsampling_ds, shuffle=True)
                train_sampler.set_epoch(epoch)
            else:
                train_sampler = None
            train_dl = DataLoader(
                subsampling_ds,
                batch_size=tcfg["batch_size"],
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=tcfg["num_workers"],
                pin_memory=_pin,
                persistent_workers=False,  # must be False — dataset changes each epoch
                drop_last=True,
            )
        else:
            if ddp:
                train_sampler.set_epoch(epoch)

        # ── Train ──
        train_metrics, global_step = train_one_epoch(
            model, train_dl, criterion, optimizer,
            scheduler if sched_name in ("cosine_warmup", "onecycle") else None,
            scaler, device, grad_accum, max_grad_norm, use_amp, epoch,
            writer, global_step, log_every, cutmix=cutmix,
        )

        # Epoch-level scheduler step (cosine, plateau)
        if sched_name == "cosine" and scheduler:
            scheduler.step()

        # ── Validate ──
        val_metrics = validate(model, val_dl, criterion, device, use_amp)

        # ── Aggregate val metrics across all DDP ranks ──
        # Each rank only sees 1/world_size of the val set via DistributedSampler,
        # so we must all_reduce before logging, checkpointing, or early stopping.
        if ddp:
            for key in ["total_loss", "seg_loss", "presence_loss",
                        "dice", "iou", "presence_acc", "presence_f1", "neg_fp_rate"]:
                t = torch.tensor(val_metrics[key], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_metrics[key] = t.item()

        # Plateau scheduler needs val metric
        if sched_name == "plateau" and scheduler:
            scheduler.step(val_metrics.get(_monitor_to_metric_key(monitor_name), val_metrics["dice"]))

        epoch_time = time.time() - t0

        # ── Log ──
        if is_main:
            logger.info(
                f"Epoch {epoch:3d}/{tcfg['epochs']} ({epoch_time:.1f}s) | "
                f"Train: loss={train_metrics['total_loss']:.4f} "
                f"seg={train_metrics['seg_loss']:.4f} "
                f"pres_acc={train_metrics['presence_acc']:.1%} | "
                f"Val: dice={val_metrics['dice']:.4f} "
                f"iou={val_metrics['iou']:.4f} "
                f"pres_acc={val_metrics['presence_acc']:.1%} "
                f"pres_f1={val_metrics['presence_f1']:.3f} "
                f"neg_fp={val_metrics['neg_fp_rate']:.4f}"
            )

            if writer:
                writer.add_scalar("val/total_loss", val_metrics["total_loss"], epoch)
                writer.add_scalar("val/seg_loss", val_metrics["seg_loss"], epoch)
                writer.add_scalar("val/presence_loss", val_metrics["presence_loss"], epoch)
                writer.add_scalar("val/dice", val_metrics["dice"], epoch)
                writer.add_scalar("val/iou", val_metrics["iou"], epoch)
                writer.add_scalar("val/presence_acc", val_metrics["presence_acc"], epoch)
                writer.add_scalar("val/presence_f1", val_metrics["presence_f1"], epoch)
                writer.add_scalar("val/neg_fp_rate", val_metrics["neg_fp_rate"], epoch)
                writer.add_scalar("epoch_time_s", epoch_time, epoch)

            # ── Checkpoint ──
            monitor_val = val_metrics.get(
                _monitor_to_metric_key(ccfg.get("monitor", "val_dice")), 0
            )
            ckpt_mgr.save_if_best(
                model, optimizer, scheduler, epoch, monitor_val,
                extra_info={"val_metrics": val_metrics, "config": cfg},
            )

        # ── Early stopping — decision must be identical on every rank ──
        # Only rank 0 runs the patience counter (it has the aggregated metric).
        # The boolean is then broadcast so all ranks break together, preventing
        # any rank from entering the next NCCL collective while others have exited.
        es_metric = val_metrics.get(
            _monitor_to_metric_key(es_cfg.get("monitor", "val_dice")),
            0,
        )
        should_stop = early_stopper.step(es_metric) if is_main else False
        if ddp:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            dist.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())
        if should_stop:
            if is_main:
                logger.info(f"Early stopping at epoch {epoch} "
                             f"(patience={es_cfg.get('patience', 25)})")
            break

    # ── Cleanup ──
    if is_main and writer:
        writer.close()
        logger.info("Training complete.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
