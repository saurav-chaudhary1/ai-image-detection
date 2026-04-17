import time
import copy
import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


log = logging.getLogger("openfake")


def build_loss(class_counts: dict, device: torch.device) -> nn.BCEWithLogitsLoss:
    
    n_real = class_counts.get(0, 1)
    n_fake = class_counts.get(1, 1)
    pos_weight = torch.tensor([n_real / n_fake], dtype=torch.float32).to(device)
    print(f"[Loss] pos_weight (real/fake ratio) = {pos_weight.item():.3f}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)



def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one full pass over the training DataLoader.
    Returns (avg_loss, accuracy_percent).
    """
    model.train()

    total_loss   = 0.0
    correct      = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="  Train", leave=False, unit="batch")):
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits.squeeze(1), labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    
    model.eval()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False, unit="batch"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits.squeeze(1), labels)

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits.squeeze(1)) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy


class EarlyStopping:
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_weights = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        
        if val_loss < self.best_loss - self.min_delta:

            self.best_loss    = val_loss
            self.counter      = 0
            self.best_weights = copy.deepcopy(model.state_dict())
        else:

            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Load the best-seen weights back into the model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print("[EarlyStopping] Best weights restored.")


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    class_counts: dict,
    num_epochs: int = 20,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    patience: int = 5,
    checkpoint_dir: str = "checkpoints",
    device: torch.device = None,
):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info(f"Training: {model_name.upper()}  |  Device: {device}")
    log.info("=" * 60)

    model = model.to(device)

    criterion = build_loss(class_counts, device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    early_stop = EarlyStopping(patience=patience)

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")

    history = {
        "train_loss":   [],
        "val_loss":     [],
        "train_acc":    [],
        "val_acc":      [],
        "epoch_times":  [],
    }

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        log.info(
            f"[{model_name}] Epoch {epoch:03d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
            log.info(f"  ✓ New best checkpoint saved → {ckpt_path}  (val_acc={val_acc:.2f}%)")

        if early_stop.step(val_loss, model):
            log.warning(f"[EarlyStopping] No improvement for {patience} epochs. Stopping early.")
            break

    early_stop.restore_best(model)
    log.info(f"[{model_name}] Training complete. Best val acc: {best_val_acc:.2f}%")

    return history