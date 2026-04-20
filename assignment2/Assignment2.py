from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_batch(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one CIFAR-10 batch.

    Returns
    - X: (d, n) float64 in [0, 1]
    - Y: (K, n) one-hot float64
    - y: (n,) int64 labels in [0, 9]
    """
    with open(filename, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")

    X = data_dict[b"data"].astype(np.float64) / 255.0  # (n, d)
    X = X.T  # (d, n)

    y = np.array(data_dict[b"labels"], dtype=np.int64)  # (n,)
    K = 10
    n = y.shape[0]
    Y = np.zeros((K, n), dtype=np.float64)
    Y[y, np.arange(n)] = 1.0
    return X, Y, y


def normalize_data(
    train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize using train mean/std per feature dimension."""
    mean_X = np.mean(train_X, axis=1, keepdims=True)
    std_X = np.std(train_X, axis=1, keepdims=True)
    std_X = np.where(std_X < 1e-12, 1.0, std_X)
    train_X_n = (train_X - mean_X) / std_X
    val_X_n = (val_X - mean_X) / std_X
    test_X_n = (test_X - mean_X) / std_X
    return train_X_n, val_X_n, test_X_n, mean_X, std_X


def load_exercise3_data(
    cifar_root: str,
) -> Dict[str, np.ndarray]:
    """
    Exercise 3 / 4 split:
    - data_batch_1: train
    - data_batch_2: validation
    - test_batch: test
    """
    train_X, train_Y, train_y = load_batch(os.path.join(cifar_root, "data_batch_1"))
    val_X, val_Y, val_y = load_batch(os.path.join(cifar_root, "data_batch_2"))
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


def load_full_train_with_val_split(
    cifar_root: str, val_size: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """
    Used by lambda search / final evaluation:
    - concatenate data_batch_1..5
    - split into train/val using val_size
    - normalize using train split stats only
    """
    parts_X: List[np.ndarray] = []
    parts_Y: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for i in range(1, 6):
        Xb, Yb, yb = load_batch(os.path.join(cifar_root, f"data_batch_{i}"))
        parts_X.append(Xb)
        parts_Y.append(Yb)
        parts_y.append(yb)

    X_all = np.concatenate(parts_X, axis=1)
    Y_all = np.concatenate(parts_Y, axis=1)
    y_all = np.concatenate(parts_y, axis=0)

    n_all = X_all.shape[1]
    if val_size >= n_all:
        raise ValueError("val_size must be smaller than number of training examples.")

    perm = rng.permutation(n_all)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_X = X_all[:, train_idx]
    train_Y = Y_all[:, train_idx]
    train_y = y_all[train_idx]
    val_X = X_all[:, val_idx]
    val_Y = Y_all[:, val_idx]
    val_y = y_all[val_idx]

    # Test split is fixed.
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


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(scores: np.ndarray) -> np.ndarray:
    """
    Softmax over classes for each column.

    scores: (K, n)
    returns: (K, n)
    """
    scores_shifted = scores - np.max(scores, axis=0, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)


def init_two_layer_network(
    K: int, d: int, m: int, rng: np.random.Generator
) -> Dict[str, List[np.ndarray]]:
    """
    Network:
      s1 = W1 x + b1   (m x n)
      h = ReLU(s1)     (m x n)
      s2 = W2 h + b2   (K x n)
      p = softmax(s2)  (K x n)
    """
    W1 = (1.0 / np.sqrt(d)) * rng.standard_normal((m, d))
    b1 = np.zeros((m, 1), dtype=np.float64)
    W2 = (1.0 / np.sqrt(m)) * rng.standard_normal((K, m))
    b2 = np.zeros((K, 1), dtype=np.float64)
    return {"W": [W1, W2], "b": [b1, b2]}


def apply_network(
    X: np.ndarray, net_params: Dict[str, List[np.ndarray]]
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Forward pass. Returns (P, fp_data)."""
    W1, W2 = net_params["W"]
    b1, b2 = net_params["b"]
    s1 = W1 @ X + b1
    h = relu(s1)
    scores = W2 @ h + b2
    P = softmax(scores)
    fp_data = {"X": X, "s1": s1, "h": h, "scores": scores, "P": P}
    return P, fp_data


def compute_loss_from_P(P: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss: mean over columns."""
    n = P.shape[1]
    probs = P[y, np.arange(n)]
    probs = np.clip(probs, 1e-15, None)
    return float(np.mean(-np.log(probs)))


def compute_cost(
    P: np.ndarray,
    y: np.ndarray,
    net_params: Dict[str, List[np.ndarray]],
    lam: float,
) -> float:
    """Loss + L2 regularization on W1 and W2."""
    W1, W2 = net_params["W"]
    loss = compute_loss_from_P(P, y)
    reg = lam * (np.sum(W1 * W1) + np.sum(W2 * W2))
    return float(loss + reg)


def compute_accuracy(P: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(P, axis=0)
    return float(np.mean(preds == y))


def backward_pass(
    X: np.ndarray,
    Y: np.ndarray,
    fp_data: Dict[str, np.ndarray],
    net_params: Dict[str, List[np.ndarray]],
    lam: float,
) -> Dict[str, List[np.ndarray]]:
    """
    Backprop for two-layer ReLU + softmax.

    Y is one-hot: (K, n).
    fp_data comes from apply_network and includes s1/h/P.
    """
    W1, W2 = net_params["W"]
    s1 = fp_data["s1"]
    h = fp_data["h"]
    P = fp_data["P"]

    n = X.shape[1]
    # For mean cross-entropy over columns: d(scores2) = (P - Y) / n
    delta2 = (P - Y) / n  # (K, n)

    grad_W2 = delta2 @ h.T + 2.0 * lam * W2  # (K, m)
    grad_b2 = np.sum(delta2, axis=1, keepdims=True)  # (K, 1)

    # Backprop into hidden: d(h) = W2^T @ d(scores2)
    delta_h = W2.T @ delta2  # (m, n)
    relu_mask = (s1 > 0.0).astype(np.float64)
    delta1 = delta_h * relu_mask  # (m, n)

    grad_W1 = delta1 @ X.T + 2.0 * lam * W1  # (m, d)
    grad_b1 = np.sum(delta1, axis=1, keepdims=True)  # (m, 1)

    return {"W": [grad_W1, grad_W2], "b": [grad_b1, grad_b2]}


def cyclic_eta(t: int, eta_min: float, eta_max: float, ns: int) -> float:
    """
    Triangular cyclical learning rate with period 2*ns.

    We use update step index t starting from 1 such that:
    - eta(1) = eta_min
    - eta(ns) = eta_max
    - eta(2*ns) = eta_min
    """
    if ns <= 1:
        return eta_max
    r = (t - 1) % (2 * ns)  # 0..2ns-1
    # Increasing: r in [0, ns-1]
    if r <= ns - 1:
        return eta_min + (r / (ns - 1)) * (eta_max - eta_min)
    # Decreasing: r in [ns, 2ns-1]
    # At r=ns -> t=ns+1: eta slightly below eta_max
    # At r=2ns-1 -> eta=eta_min
    r_dec = r - (ns - 1)  # 1..ns
    return eta_max - (r_dec / ns) * (eta_max - eta_min)


def make_eval_update_indices(
    cycles: int, ns: int, eval_per_cycle: int
) -> List[int]:
    """Evaluation points used for plotting/recording curves."""
    if eval_per_cycle <= 0:
        return []
    total_updates = cycles * 2 * ns
    out: List[int] = []
    for c in range(cycles):
        cycle_start = c * 2 * ns
        for j in range(1, eval_per_cycle + 1):
            # t is 1-indexed
            idx = cycle_start + int(round(j * (2 * ns) / eval_per_cycle))
            idx = min(idx, total_updates)
            if idx not in out:
                out.append(idx)
    out.sort()
    return out


def train_cyclical_sgd(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    y_val: np.ndarray,
    init_net: Dict[str, List[np.ndarray]],
    lam: float,
    rng: np.random.Generator,
    batch_size: int,
    eta_min: float,
    eta_max: float,
    ns: int,
    cycles: int,
    eval_per_cycle: int = 10,
    verbose: bool = True,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[float]]]:
    net = copy.deepcopy(init_net)

    n = X_train.shape[1]
    if batch_size <= 0 or batch_size > n:
        raise ValueError("Invalid batch_size.")
    batches_per_epoch = n // batch_size
    if batches_per_epoch <= 0:
        raise ValueError("batch_size is too large for current training set size.")

    total_updates = cycles * 2 * ns
    eval_indices = set(make_eval_update_indices(cycles, ns, eval_per_cycle))

    history: Dict[str, List[float]] = {
        "update_step": [],
        "train_loss": [],
        "val_loss": [],
        "train_cost": [],
        "val_cost": [],
        "train_acc": [],
        "val_acc": [],
    }

    t = 0
    epoch = 0
    perm = rng.permutation(n)
    X_shuf = X_train[:, perm]
    Y_shuf = Y_train[:, perm]
    y_shuf = y_train[perm]
    batch_ptr = 0

    best_val_acc = -1.0
    best_state = copy.deepcopy(net)

    while t < total_updates:
        if batch_ptr >= batches_per_epoch:
            epoch += 1
            perm = rng.permutation(n)
            X_shuf = X_train[:, perm]
            Y_shuf = Y_train[:, perm]
            y_shuf = y_train[perm]
            batch_ptr = 0

        j_start = batch_ptr * batch_size
        j_end = j_start + batch_size
        X_batch = X_shuf[:, j_start:j_end]
        Y_batch = Y_shuf[:, j_start:j_end]
        y_batch = y_shuf[j_start:j_end]

        t += 1
        eta_t = cyclic_eta(t, eta_min=eta_min, eta_max=eta_max, ns=ns)

        P_batch, fp_data = apply_network(X_batch, net)
        grads = backward_pass(X_batch, Y_batch, fp_data, net, lam=lam)

        # Gradient descent update
        for i in range(2):
            net["W"][i] -= eta_t * grads["W"][i]
            net["b"][i] -= eta_t * grads["b"][i]

        if t in eval_indices:
            P_train, _ = apply_network(X_train, net)
            P_val, _ = apply_network(X_val, net)

            train_loss = compute_loss_from_P(P_train, y_train)
            val_loss = compute_loss_from_P(P_val, y_val)
            train_cost = compute_cost(P_train, y_train, net, lam)
            val_cost = compute_cost(P_val, y_val, net, lam)
            train_acc = compute_accuracy(P_train, y_train)
            val_acc = compute_accuracy(P_val, y_val)

            history["update_step"].append(t)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_cost"].append(train_cost)
            history["val_cost"].append(val_cost)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if verbose:
                print(
                    f"t={t:5d}/{total_updates} | eta={eta_t:.3e} | "
                    f"val_acc={val_acc * 100:.2f}% | val_cost={val_cost:.4f}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(net)

        batch_ptr += 1

    return best_state, history


def plot_curves(
    x: List[int],
    train: List[float],
    val: List[float],
    title: str,
    ylabel: str,
    outpath: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, train, label="train")
    plt.plot(x, val, label="validation")
    plt.xlabel("update step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_three_metrics(history: Dict[str, List[float]], prefix: str) -> None:
    x = history["update_step"]
    plot_curves(
        x,
        history["train_cost"],
        history["val_cost"],
        title=f"Cost ({prefix})",
        ylabel="cost",
        outpath=f"{prefix}_cost.png",
    )
    plot_curves(
        x,
        history["train_loss"],
        history["val_loss"],
        title=f"Loss ({prefix})",
        ylabel="loss",
        outpath=f"{prefix}_loss.png",
    )
    plot_curves(
        x,
        history["train_acc"],
        history["val_acc"],
        title=f"Accuracy ({prefix})",
        ylabel="accuracy",
        outpath=f"{prefix}_accuracy.png",
    )


def gradient_check_two_layer(
    data: Dict[str, np.ndarray],
    rng: np.random.Generator,
    d_small: int = 5,
    n_small: int = 3,
    m: int = 6,
    lam: float = 0.0,
) -> None:
    """
    Compare your analytic gradients with PyTorch autograd.

    NOTE: requires `torch` to be installed.
    """
    X = data["train_X"]
    Y = data["train_Y"]
    y = data["train_y"]
    K = Y.shape[0]

    X_small = X[:d_small, :n_small]
    Y_small = Y[:, :n_small]
    y_small = y[:n_small]

    net = init_two_layer_network(K=K, d=d_small, m=m, rng=rng)
    P, fp_data = apply_network(X_small, net)
    grads_analytic = backward_pass(X_small, Y_small, fp_data, net, lam=lam)

    try:
        from torch_gradient_computations import ComputeGradsWithTorch
    except Exception as exc:
        print(f"[gradient_check] skipped (torch not available?): {exc}")
        return

    grads_torch = ComputeGradsWithTorch(X_small, y_small, net, lam=lam)

    eps = 1e-12
    rels_W = []
    rels_b = []
    for i in range(2):
        rel_W = np.max(
            np.abs(grads_analytic["W"][i] - grads_torch["W"][i])
            / np.maximum(eps, np.abs(grads_analytic["W"][i]) + np.abs(grads_torch["W"][i]))
        )
        rel_b = np.max(
            np.abs(grads_analytic["b"][i] - grads_torch["b"][i])
            / np.maximum(eps, np.abs(grads_analytic["b"][i]) + np.abs(grads_torch["b"][i]))
        )
        rels_W.append(rel_W)
        rels_b.append(rel_b)

    print(
        f"[gradient_check] relative error: "
        f"W1={rels_W[0]:.3e}, W2={rels_W[1]:.3e}, b1={rels_b[0]:.3e}, b2={rels_b[1]:.3e}"
    )


def train_vanilla_overfit_check(
    X: np.ndarray,
    Y: np.ndarray,
    y: np.ndarray,
    init_net: Dict[str, List[np.ndarray]],
    lam: float,
    rng: np.random.Generator,
    batch_size: int = 50,
    eta: float = 0.01,
    n_epochs: int = 200,
) -> Dict[str, List[float]]:
    """
    A sanity check: with small data and lam=0, the network should be able to
    overfit and get very low loss on the training subset.
    """
    net = copy.deepcopy(init_net)
    n = X.shape[1]
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    history: Dict[str, List[float]] = {"loss": [], "acc": [], "cost": []}
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuf = X[:, perm]
        Y_shuf = Y[:, perm]
        y_shuf = y[perm]

        # Mini-batch updates
        for j_start in range(0, n - batch_size + 1, batch_size):
            X_batch = X_shuf[:, j_start : j_start + batch_size]
            Y_batch = Y_shuf[:, j_start : j_start + batch_size]

            P_batch, fp_data = apply_network(X_batch, net)
            grads = backward_pass(X_batch, Y_batch, fp_data, net, lam=lam)
            for i in range(2):
                net["W"][i] -= eta * grads["W"][i]
                net["b"][i] -= eta * grads["b"][i]

        P_all, _ = apply_network(X, net)
        loss = compute_loss_from_P(P_all, y)
        cost = compute_cost(P_all, y, net, lam)
        acc = compute_accuracy(P_all, y)
        history["loss"].append(loss)
        history["cost"].append(cost)
        history["acc"].append(acc)

        if (epoch + 1) % max(1, (n_epochs // 10)) == 0:
            print(
                f"[overfit] epoch {epoch + 1}/{n_epochs} | loss={loss:.4f} | cost={cost:.4f} | acc={acc * 100:.2f}%"
            )

    return history


def log_uniform_samples(l_min: float, l_max: float, num_samples: int, rng: np.random.Generator) -> List[float]:
    """Sample lam = 10^l where l ~ Uniform(l_min, l_max)."""
    ls = l_min + (l_max - l_min) * rng.random(num_samples)
    return [float(10**l) for l in ls]


def lambda_search_coarse_to_fine(
    cifar_root: str,
    rng: np.random.Generator,
    hidden_m: int,
    batch_size: int,
    eta_min: float,
    eta_max: float,
    lam_coarse_range: Tuple[float, float] = (-5, -1),
    lam_fine_radius: float = 1.0,
    coarse_grid_points: int = 8,
    coarse_cycles: int = 2,
    fine_cycles: int = 3,
    fine_num_samples: int = 8,
    val_size_coarse: int = 5000,
    val_size_final: int = 1000,
    outdir: str = ".",
    verbose: bool = True,
) -> float:
    os.makedirs(outdir, exist_ok=True)

    # Coarse search split
    data = load_full_train_with_val_split(cifar_root, val_size=val_size_coarse, rng=rng)
    K = 10
    d = data["train_X"].shape[0]
    n_train = data["train_X"].shape[1]
    ns = 2 * (n_train // batch_size)
    if ns <= 0:
        raise ValueError("ns computed as 0; increase training size or reduce batch_size.")

    l_min, l_max = lam_coarse_range
    # Coarse: uniform grid in log-space (as suggested by the PDF experiments)
    l_vals = np.linspace(l_min, l_max, coarse_grid_points)
    coarse_lams = [float(10**l) for l in l_vals]

    coarse_results: List[Dict] = []
    for idx, lam in enumerate(coarse_lams):
        if verbose:
            print(f"\n[lambda coarse] {idx+1}/{len(coarse_lams)} lam={lam:g} (log10={np.log10(lam):.2f})")
        init_net = init_two_layer_network(K=K, d=d, m=hidden_m, rng=rng)
        trained_net, hist = train_cyclical_sgd(
            data["train_X"],
            data["train_Y"],
            data["train_y"],
            data["val_X"],
            data["val_Y"],
            data["val_y"],
            init_net=init_net,
            lam=lam,
            rng=rng,
            batch_size=batch_size,
            eta_min=eta_min,
            eta_max=eta_max,
            ns=ns,
            cycles=coarse_cycles,
            eval_per_cycle=5,
            verbose=verbose,
        )
        best_val_acc = float(max(hist["val_acc"]) if hist["val_acc"] else 0.0)
        coarse_results.append(
            {
                "lam": lam,
                "ns": ns,
                "batch_size": batch_size,
                "cycles": coarse_cycles,
                "best_val_acc": best_val_acc,
            }
        )

    coarse_path = os.path.join(outdir, "lambda_search_coarse.json")
    with open(coarse_path, "w", encoding="utf-8") as f:
        json.dump(coarse_results, f, indent=2)
    if verbose:
        print(f"[lambda coarse] saved to {coarse_path}")

    # Pick best from coarse
    coarse_results_sorted = sorted(coarse_results, key=lambda x: x["best_val_acc"], reverse=True)
    best_lam = float(coarse_results_sorted[0]["lam"])
    l_best = float(np.log10(best_lam))

    # Fine search split (use a new RNG split so that val set stays consistent with spec)
    data_fine = load_full_train_with_val_split(cifar_root, val_size=val_size_coarse, rng=rng)
    n_train_fine = data_fine["train_X"].shape[1]
    ns_fine = 2 * (n_train_fine // batch_size)

    l_min_f = l_best - lam_fine_radius
    l_max_f = l_best + lam_fine_radius
    fine_lams = log_uniform_samples(l_min_f, l_max_f, fine_num_samples, rng=rng)

    fine_results: List[Dict] = []
    for idx, lam in enumerate(fine_lams):
        if verbose:
            print(f"\n[lambda fine] {idx+1}/{len(fine_lams)} lam={lam:g} (log10={np.log10(lam):.2f})")
        init_net = init_two_layer_network(K=K, d=d, m=hidden_m, rng=rng)
        trained_net, hist = train_cyclical_sgd(
            data_fine["train_X"],
            data_fine["train_Y"],
            data_fine["train_y"],
            data_fine["val_X"],
            data_fine["val_Y"],
            data_fine["val_y"],
            init_net=init_net,
            lam=lam,
            rng=rng,
            batch_size=batch_size,
            eta_min=eta_min,
            eta_max=eta_max,
            ns=ns_fine,
            cycles=fine_cycles,
            eval_per_cycle=5,
            verbose=verbose,
        )
        best_val_acc = float(max(hist["val_acc"]) if hist["val_acc"] else 0.0)
        fine_results.append(
            {
                "lam": lam,
                "ns": ns_fine,
                "batch_size": batch_size,
                "cycles": fine_cycles,
                "best_val_acc": best_val_acc,
            }
        )

    fine_path = os.path.join(outdir, "lambda_search_fine.json")
    with open(fine_path, "w", encoding="utf-8") as f:
        json.dump(fine_results, f, indent=2)
    if verbose:
        print(f"[lambda fine] saved to {fine_path}")

    fine_sorted = sorted(fine_results, key=lambda x: x["best_val_acc"], reverse=True)
    best_fine_lam = float(fine_sorted[0]["lam"])
    if verbose:
        print(f"\n[lambda] best coarse lam={best_lam:g}, best fine lam={best_fine_lam:g}")

    # Final training for reporting (val_size_final)
    data_final = load_full_train_with_val_split(cifar_root, val_size=val_size_final, rng=rng)
    d_final = data_final["train_X"].shape[0]
    init_net = init_two_layer_network(K=K, d=d_final, m=hidden_m, rng=rng)
    # For the final run, follow the Exercise 4 defaults.
    ns_final = 800
    cycles_final = 3
    trained_net, hist = train_cyclical_sgd(
        data_final["train_X"],
        data_final["train_Y"],
        data_final["train_y"],
        data_final["val_X"],
        data_final["val_Y"],
        data_final["val_y"],
        init_net=init_net,
        lam=best_fine_lam,
        rng=rng,
        batch_size=batch_size,
        eta_min=eta_min,
        eta_max=eta_max,
        ns=ns_final,
        cycles=cycles_final,
        eval_per_cycle=5,
        verbose=verbose,
    )

    plot_three_metrics(hist, prefix=os.path.join(outdir, "final_best_lam"))

    P_test, _ = apply_network(data_final["test_X"], trained_net)
    test_acc = compute_accuracy(P_test, data_final["test_y"])
    print(f"[final] best_lam={best_fine_lam:g}, test_acc={test_acc * 100:.2f}%")
    return best_fine_lam


def run_exercise3_4(
    cifar_root: str,
    rng: np.random.Generator,
    hidden_m: int,
    lam: float = 0.01,
    outdir: str = ".",
    verbose: bool = True,
    run_3: bool = True,
    run_4: bool = True,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    data = load_exercise3_data(cifar_root)
    K = 10
    d = data["train_X"].shape[0]

    eta_min, eta_max = 1e-5, 1e-1
    batch_size = 100

    if run_3:
        # Exercise 3: 1 cycle, ns=500
        init_net = init_two_layer_network(K=K, d=d, m=hidden_m, rng=rng)
        trained_net, hist3 = train_cyclical_sgd(
            data["train_X"],
            data["train_Y"],
            data["train_y"],
            data["val_X"],
            data["val_Y"],
            data["val_y"],
            init_net=init_net,
            lam=lam,
            rng=rng,
            batch_size=batch_size,
            eta_min=eta_min,
            eta_max=eta_max,
            ns=500,
            cycles=1,
            eval_per_cycle=10,
            verbose=verbose,
        )
        plot_three_metrics(hist3, prefix=os.path.join(outdir, "exercise3"))
        P_test, _ = apply_network(data["test_X"], trained_net)
        test_acc3 = compute_accuracy(P_test, data["test_y"])
        print(f"[exercise3] test_acc={test_acc3 * 100:.2f}%")

    if run_4:
        # Exercise 4: 3 cycles, ns=800
        init_net = init_two_layer_network(K=K, d=d, m=hidden_m, rng=rng)
        trained_net, hist4 = train_cyclical_sgd(
            data["train_X"],
            data["train_Y"],
            data["train_y"],
            data["val_X"],
            data["val_Y"],
            data["val_y"],
            init_net=init_net,
            lam=lam,
            rng=rng,
            batch_size=batch_size,
            eta_min=eta_min,
            eta_max=eta_max,
            ns=800,
            cycles=3,
            eval_per_cycle=10,
            verbose=verbose,
        )
        plot_three_metrics(hist4, prefix=os.path.join(outdir, "exercise4"))
        P_test, _ = apply_network(data["test_X"], trained_net)
        test_acc4 = compute_accuracy(P_test, data["test_y"])
        print(f"[exercise4] test_acc={test_acc4 * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="gradient_check",
                        choices=[
                            "gradient_check",
                            "overfit_check",
                            "exercise3",
                            "exercise4",
                            "exercise3_4",
                            "lambda_search",
                            "full",
                        ])
    parser.add_argument("--data_dir", type=str, default="Datasets/cifar-10-batches-py")
    parser.add_argument("--hidden_m", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--lam_default", type=float, default=0.01)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cifar_root = os.path.join(args.data_dir)
    if not os.path.isdir(cifar_root):
        raise FileNotFoundError(f"Cannot find dataset dir: {cifar_root}")

    rng = np.random.default_rng(args.seed)
    data_ex3 = load_exercise3_data(cifar_root)

    if args.task == "gradient_check":
        gradient_check_two_layer(data_ex3, rng, d_small=5, n_small=3, m=6, lam=0.0)
        return

    if args.task == "overfit_check":
        X = data_ex3["train_X"][:, :100]
        Y = data_ex3["train_Y"][:, :100]
        y = data_ex3["train_y"][:100]
        d = X.shape[0]
        K = 10
        net = init_two_layer_network(K=K, d=d, m=args.hidden_m, rng=rng)
        hist = train_vanilla_overfit_check(
            X, Y, y, init_net=net, lam=0.0, rng=rng, batch_size=50, eta=0.01, n_epochs=200
        )
        print(f"[overfit] final loss={hist['loss'][-1]:.4f}, acc={hist['acc'][-1] * 100:.2f}%")
        return

    if args.task == "exercise3":
        run_exercise3_4(
            cifar_root,
            rng,
            hidden_m=args.hidden_m,
            lam=args.lam_default,
            outdir=args.outdir,
            verbose=args.verbose,
            run_3=True,
            run_4=False,
        )
        return

    if args.task == "exercise4":
        run_exercise3_4(
            cifar_root,
            rng,
            hidden_m=args.hidden_m,
            lam=args.lam_default,
            outdir=args.outdir,
            verbose=args.verbose,
            run_3=False,
            run_4=True,
        )
        return

    if args.task == "exercise3_4":
        run_exercise3_4(cifar_root, rng, hidden_m=args.hidden_m, lam=args.lam_default, outdir=args.outdir, verbose=args.verbose)
        return

    if args.task == "lambda_search":
        lambda_search_coarse_to_fine(
            cifar_root=cifar_root,
            rng=rng,
            hidden_m=args.hidden_m,
            batch_size=100,
            eta_min=1e-5,
            eta_max=1e-1,
            lam_coarse_range=(-5, -1),
            lam_fine_radius=1.0,
            coarse_grid_points=8,
            coarse_cycles=2,
            fine_cycles=3,
            fine_num_samples=8,
            val_size_coarse=5000,
            val_size_final=1000,
            outdir=args.outdir,
            verbose=args.verbose,
        )
        return

    if args.task == "full":
        # Default "full": gradient check + overfit + exercise3/4 + lambda search.
        gradient_check_two_layer(data_ex3, rng, d_small=5, n_small=3, m=6, lam=0.0)
        X = data_ex3["train_X"][:, :100]
        Y = data_ex3["train_Y"][:, :100]
        y = data_ex3["train_y"][:100]
        net = init_two_layer_network(K=10, d=X.shape[0], m=args.hidden_m, rng=rng)
        train_vanilla_overfit_check(X, Y, y, init_net=net, lam=0.0, rng=rng, batch_size=50, eta=0.01, n_epochs=200)
        run_exercise3_4(cifar_root, rng, hidden_m=args.hidden_m, lam=args.lam_default, outdir=args.outdir, verbose=args.verbose)
        lambda_search_coarse_to_fine(
            cifar_root=cifar_root,
            rng=rng,
            hidden_m=args.hidden_m,
            batch_size=100,
            eta_min=1e-5,
            eta_max=1e-1,
            lam_coarse_range=(-5, -1),
            lam_fine_radius=1.0,
            coarse_grid_points=8,
            coarse_cycles=2,
            fine_cycles=3,
            fine_num_samples=8,
            val_size_coarse=5000,
            val_size_final=1000,
            outdir=args.outdir,
            verbose=args.verbose,
        )
        return

    raise RuntimeError("Unknown task.")


if __name__ == "__main__":
    main()

