import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)



@torch.no_grad()
def get_predictions(model: nn.Module, loader, device: torch.device):
    
    model.eval()

    all_probs  = []
    all_preds  = []
    all_labels = []

    start_time = time.time()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)                               
        probs  = torch.sigmoid(logits.squeeze(1))            
        preds  = (probs >= 0.5).float()                      

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    elapsed = time.time() - start_time


    all_probs  = np.concatenate(all_probs)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    n_images   = len(all_labels)
    throughput = n_images / elapsed

    return all_probs, all_preds, all_labels, throughput



def plot_loss_accuracy(histories: dict, save_dir: str):

    n_models = len(histories)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    fig.suptitle("Training History: Loss & Accuracy", fontsize=15, fontweight="bold")


    if n_models == 1:
        axes = [axes]

    for row_idx, (name, hist) in enumerate(histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)

        # ── Loss subplot ─────────────────────────────────────────
        ax_loss = axes[row_idx][0]
        ax_loss.plot(epochs, hist["train_loss"], label="Train", marker="o", ms=3)
        ax_loss.plot(epochs, hist["val_loss"],   label="Val",   marker="s", ms=3)
        ax_loss.set_title(f"{name} – Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("BCE Loss")
        ax_loss.legend()
        ax_loss.grid(alpha=0.3)


        ax_acc = axes[row_idx][1]
        ax_acc.plot(epochs, hist["train_acc"], label="Train", marker="o", ms=3)
        ax_acc.plot(epochs, hist["val_acc"],   label="Val",   marker="s", ms=3)
        ax_acc.set_title(f"{name} – Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0, 105)
        ax_acc.legend()
        ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "loss_accuracy_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Saved: {path}")



def plot_confusion_matrices(predictions: dict, save_dir: str):

    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Confusion Matrices (Test Set)", fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, (name, (_, preds, labels, _)) in zip(axes, predictions.items()):
        cm = confusion_matrix(labels, preds)


        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Real (0)", "Fake (1)"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}")

        
        tn, fp, fn, tp = cm.ravel()
        print(f"[{name}] TN={tn} | FP={fp} | FN={fn} | TP={tp}")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Saved: {path}")



def plot_roc_curves(predictions: dict, save_dir: str):
    
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["royalblue", "darkorange", "green", "red"]

    for (name, (probs, _, labels, _)), color in zip(predictions.items(), colors):

        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random (AUC = 0.5)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Saved: {path}")



def plot_inference_speed(predictions: dict, save_dir: str):
    
    names       = list(predictions.keys())
    throughputs = [v[3] for v in predictions.values()]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, throughputs, color=["royalblue", "darkorange"][:len(names)],
                  edgecolor="black", width=0.4)


    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Images / second", fontsize=12)
    ax.set_title("Inference Speed on Test Set", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(throughputs) * 1.25)    
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(save_dir, "inference_speed.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Saved: {path}")



def evaluate_all(
    models: dict,
    test_loaders: dict,
    histories: dict,
    device: torch.device,
    save_dir: str = "results",
):
    
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EVALUATION (test set)")
    print(f"{'='*60}")


    predictions = {}
    for name, model in models.items():
        print(f"\n[Evaluate] Running inference → {name}")
        probs, preds, labels, throughput = get_predictions(
            model, test_loaders[name], device
        )
        predictions[name] = (probs, preds, labels, throughput)
        acc = 100.0 * (preds == labels).mean()
        print(f"  Test Accuracy : {acc:.2f}%")
        print(f"  Throughput    : {throughput:.1f} images/sec")


    plot_loss_accuracy(histories, save_dir)
    plot_confusion_matrices(predictions, save_dir)
    plot_roc_curves(predictions, save_dir)
    plot_inference_speed(predictions, save_dir)


    print(f"\n{'─'*60}")
    print(f"  {'Model':<18} {'Test Acc':>10} {'AUC':>10} {'Imgs/sec':>12}")
    print(f"{'─'*60}")

    summary = {}
    for name, (probs, preds, labels, throughput) in predictions.items():
        acc     = 100.0 * (preds == labels).mean()
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        print(f"  {name:<18} {acc:>9.2f}% {roc_auc:>10.4f} {throughput:>11.1f}")
        summary[name] = {"test_acc": acc, "auc": roc_auc, "throughput": throughput}

    print(f"{'─'*60}")
    print(f"\n[Evaluate] All plots saved to: {save_dir}/")

    return summary