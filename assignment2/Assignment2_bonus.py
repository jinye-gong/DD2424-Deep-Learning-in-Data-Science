from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from Assignment2 import load_batch, normalize_data


def load_all_train_split_with_val(
    cifar_root: str,
    val_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    parts_X: List[np.ndarray] = []
    parts_Y: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for i in range(1, 6):
        Xb, Yb, yb = load_batch(os.path.join(cifar_root, f"data_batch_{i}"))
        parts_X.append(Xb)
        parts_Y.append(Yb)
        parts_y.append(yb)

    X_all = np.concatenate(parts_X, axis=1)  # d x n_all
    Y_all = np.concatenate(parts_Y, axis=1)  # K x n_all
    y_all = np.concatenate(parts_y, axis=0)  # n_all

    n_all = X_all.shape[1]
    if val_size >= n_all:
        raise ValueError("val_size must be smaller than number of candidates.")

    perm = rng.permutation(n_all)
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]

    train_X = X_all[:, tr_idx]
    train_Y = Y_all[:, tr_idx]
    train_y = y_all[tr_idx]
    val_X = X_all[:, val_idx]
    val_Y = Y_all[:, val_idx]
    val_y = y_all[val_idx]

    test_X, test_Y, test_y = load_batch(os.path.join(cifar_root, "test_batch"))
    train_X, val_X, test_X, mean_X, std_X = normalize_data(train_X, val_X, test_X)

    return {
        "train_X": train_X,
        "train_Y": train_Y,
        "train_y": train_y,
        "val_X": val_X,
        "val_Y": val_Y,
        "val_y": val_y,
        "test_X": test_X,
        "test_Y": test_Y,
        "test_y": test_y,
        "mean_X": mean_X,
        "std_X": std_X,
    }


def build_horizontal_flip_indices() -> np.ndarray:
    """CIFAR column layout: 3072 = 3 * 1024 (R,G,B planes)."""
    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
    inds = np.vstack((ind_flip, 1024 + ind_flip))
    inds = np.vstack((inds, 2048 + ind_flip))
    return inds.ravel()


class TwoLayerNetTorch(nn.Module):
    def __init__(self, d: int, m: int, K: int, dropout_p: float):
        super().__init__()
        self.fc1 = nn.Linear(d, m, bias=True)
        self.fc2 = nn.Linear(m, K, bias=True)
        self.dropout_p = dropout_p

        # Match Assignment 2 initialization style
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0 / np.sqrt(d))
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0 / np.sqrt(m))
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d)
        h = F.relu(self.fc1(x))
        if self.dropout_p > 0:
            h = F.dropout(h, p=self.dropout_p, training=self.training)
        logits = self.fc2(h)
        return logits


@torch.no_grad()
def evaluate(net: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> Tuple[float, float]:
    net.eval()
    n = X.shape[0]
    losses: List[float] = []
    accs: List[float] = []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xb = X[start:end]
        yb = y[start:end]
        logits = net(xb)
        ce = F.cross_entropy(logits, yb, reduction="mean")
        preds = logits.argmax(dim=1)
        acc = (preds == yb).float().mean().item()
        losses.append(ce.item())
        accs.append(acc)
    return float(np.mean(losses)), float(np.mean(accs))


def train_bonus(
    data: Dict[str, np.ndarray],
    m: int,
    lam: float,
    dropout_p: float,
    lr: float,
    batch_size: int,
    epochs: int,
    flip_p: float,
    seed: int,
    outdir: str,
) -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)

    # Determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bonus] device={device}")

    d = data["train_X"].shape[0]
    K = 10

    # Store datasets as torch tensors on CPU
    X_train = torch.from_numpy(data["train_X"]).float().T.contiguous()  # (n, d)
    y_train = torch.from_numpy(data["train_y"]).long().contiguous()  # (n,)
    X_val = torch.from_numpy(data["val_X"]).float().T.contiguous()
    y_val = torch.from_numpy(data["val_y"]).long().contiguous()
    X_test = torch.from_numpy(data["test_X"]).float().T.contiguous()
    y_test = torch.from_numpy(data["test_y"]).long().contiguous()

    net = TwoLayerNetTorch(d=d, m=m, K=K, dropout_p=dropout_p).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)

    flip_inds = build_horizontal_flip_indices()
    flip_inds_t = torch.tensor(flip_inds, dtype=torch.long, device=device)

    n = X_train.shape[0]
    steps_per_epoch = n // batch_size

    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(epochs):
        net.train()
        perm = rng.permutation(n)
        Xp = X_train[perm]
        yp = y_train[perm]

        for step in range(steps_per_epoch):
            start = step * batch_size
            end = start + batch_size
            xb = Xp[start:end].to(device)
            yb = yp[start:end].to(device)

            # Data augmentation: random horizontal flip (column-layout aware indices)
            if flip_p > 0:
                flip_mask = (torch.rand(batch_size, device=device) < flip_p)
                xb_flipped = xb[:, flip_inds_t]
                xb = torch.where(flip_mask.unsqueeze(1), xb_flipped, xb)

            logits = net(xb)
            ce = F.cross_entropy(logits, yb, reduction="mean")

            # L2 penalty matching Assignment 2: lam * (||W1||^2 + ||W2||^2)
            reg = lam * (net.fc1.weight.pow(2).sum() + net.fc2.weight.pow(2).sum())
            loss = ce + reg

            opt.zero_grad()
            loss.backward()
            opt.step()

        train_loss, train_acc = evaluate(net, X_train.to(device), y_train.to(device), batch_size=batch_size)
        val_loss, val_acc = evaluate(net, X_val.to(device), y_val.to(device), batch_size=batch_size)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[bonus] epoch {epoch+1}/{epochs} | train_acc={train_acc*100:.2f}% val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    # Restore best
    if best_state is not None:
        net.load_state_dict(best_state, strict=True)

    # Final test acc
    net.eval()
    with torch.no_grad():
        logits = []
        for start in range(0, X_test.shape[0], batch_size):
            end = min(X_test.shape[0], start + batch_size)
            xb = X_test[start:end].to(device)
            logits.append(net(xb).cpu())
        logits = torch.cat(logits, dim=0)
        preds = logits.argmax(dim=1).numpy()
    test_acc = float(np.mean(preds == data["test_y"]))

    # Plot curves
    epochs_axis = np.arange(1, epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, history["val_acc"], label="val_acc")
    plt.plot(epochs_axis, history["train_acc"], label="train_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Bonus Training Accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(outdir, f"bonus_acc_m{m}_lam{lam}.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, history["val_loss"], label="val_loss")
    plt.plot(epochs_axis, history["train_loss"], label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Bonus Training Loss")
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(outdir, f"bonus_loss_m{m}_lam{lam}.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(f"[bonus] best_val_acc={best_val_acc*100:.2f}% test_acc={test_acc*100:.2f}%")
    return {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "acc_plot": acc_path,
        "loss_plot": loss_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar_root", type=str, default="Datasets/cifar-10-batches-py")
    parser.add_argument("--m", type=int, default=200)
    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--flip_p", type=float, default=0.5)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=".")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = load_all_train_split_with_val(
        cifar_root=args.cifar_root,
        val_size=args.val_size,
        rng=rng,
    )

    train_bonus(
        data=data,
        m=args.m,
        lam=args.lam,
        dropout_p=args.dropout_p,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        flip_p=args.flip_p,
        seed=args.seed,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()

