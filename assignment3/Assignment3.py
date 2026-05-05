from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def softmax(scores: np.ndarray) -> np.ndarray:
    scores_shifted = scores - np.max(scores, axis=0, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)


def load_batch(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(filename, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
    X = data_dict[b"data"].astype(np.float32).T / 255.0
    y = np.array(data_dict[b"labels"], dtype=np.int64)
    Y = one_hot(y, k=10).astype(np.float32)
    return X, Y, y


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    n = y.shape[0]
    Y = np.zeros((k, n), dtype=np.float32)
    Y[y, np.arange(n)] = 1.0
    return Y


def normalize_data(
    train_X: np.ndarray, val_X: np.ndarray, test_X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_X = np.mean(train_X, axis=1, keepdims=True)
    std_X = np.std(train_X, axis=1, keepdims=True)
    std_X = np.where(std_X < 1e-8, 1.0, std_X)
    return (train_X - mean_X) / std_X, (val_X - mean_X) / std_X, (test_X - mean_X) / std_X, mean_X, std_X


def load_ex3_data(cifar_root: str) -> Dict[str, np.ndarray]:
    train_X, train_Y, train_y = load_batch(os.path.join(cifar_root, "data_batch_1"))
    val_X, val_Y, val_y = load_batch(os.path.join(cifar_root, "data_batch_2"))
    test_X, test_Y, test_y = load_batch(os.path.join(cifar_root, "test_batch"))
    train_X, val_X, test_X, _, _ = normalize_data(train_X, val_X, test_X)
    return {
        "train_X": train_X.astype(np.float32),
        "train_Y": train_Y.astype(np.float32),
        "train_y": train_y,
        "val_X": val_X.astype(np.float32),
        "val_Y": val_Y.astype(np.float32),
        "val_y": val_y,
        "test_X": test_X.astype(np.float32),
        "test_Y": test_Y.astype(np.float32),
        "test_y": test_y,
    }


def cifar_cols_to_images(X: np.ndarray) -> np.ndarray:
    """
    Convert CIFAR flattened columns (3072, n) to (32, 32, 3, n).
    """
    n = X.shape[1]
    return np.transpose(X.reshape((32, 32, 3, n), order="F"), (1, 0, 2, 3))


def build_mx(X_ims: np.ndarray, f: int) -> np.ndarray:
    """
    Build matrix representation MX for patchify convolution.

    X_ims: (32, 32, 3, n)
    returns MX: (n_p, 3*f*f, n)
    """
    _, _, c, n = X_ims.shape
    if c != 3:
        raise ValueError("Expected 3 channels.")

    out_hw = 32 // f
    n_p = out_hw * out_hw
    MX = np.zeros((n_p, 3 * f * f, n), dtype=X_ims.dtype)

    for i in range(n):
        l = 0
        # Match assignment order: second dim changes first, then first dim.
        for row in range(0, 32, f):
            for col in range(0, 32, f):
                patch = X_ims[row : row + f, col : col + f, :, i]
                MX[l, :, i] = patch.reshape((1, 3 * f * f), order="C")
                l += 1

    return MX


def conv_slow(X_ims: np.ndarray, Fs: np.ndarray) -> np.ndarray:
    """
    Slow reference patchify convolution using explicit loops.

    X_ims: (32, 32, 3, n), Fs: (f, f, 3, nf)
    returns: (32/f, 32/f, nf, n)
    """
    f, _, c, nf = Fs.shape
    _, _, c_x, n = X_ims.shape
    if c != c_x:
        raise ValueError("Channel mismatch.")

    out_hw = 32 // f
    conv_outputs = np.zeros((out_hw, out_hw, nf, n), dtype=X_ims.dtype)

    for i in range(n):
        for k in range(nf):
            out_r = 0
            for row in range(0, 32, f):
                out_c = 0
                for col in range(0, 32, f):
                    patch = X_ims[row : row + f, col : col + f, :, i]
                    conv_outputs[out_r, out_c, k, i] = np.sum(
                        np.multiply(patch, Fs[:, :, :, k])
                    )
                    out_c += 1
                out_r += 1

    return conv_outputs


def conv_fast(MX: np.ndarray, Fs_flat: np.ndarray) -> np.ndarray:
    """
    Fast patchify convolution with einsum.

    MX: (n_p, 3*f*f, n), Fs_flat: (3*f*f, nf)
    returns: (n_p, nf, n)
    """
    return np.einsum("ijn,jl->iln", MX, Fs_flat, optimize=True)


@dataclass
class NetParams:
    Fs_flat: np.ndarray  # (3*f*f, nf)
    W1: np.ndarray  # (nh, n_p*nf)
    W2: np.ndarray  # (K, nh)
    b_conv: np.ndarray  # (nf, 1)
    b1: np.ndarray  # (nh, 1)
    b2: np.ndarray  # (K, 1)


@dataclass
class TrainConfig:
    f: int
    nf: int
    nh: int
    lam: float
    n_batch: int
    eta_min: float
    eta_max: float
    n_cycles: int
    step_size: int
    increasing_cycle: bool
    label_smoothing_eps: float
    eval_every: int
    seed: int
    eta_max_decay: float = 1.0


def forward_pass(
    MX: np.ndarray,
    params: NetParams,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Forward pass.
    """
    conv_raw = conv_fast(MX, params.Fs_flat) + params.b_conv.reshape((1, -1, 1))
    conv_flat_pre_relu = conv_raw.reshape(
        (conv_raw.shape[0] * conv_raw.shape[1], conv_raw.shape[2]), order="C"
    )
    conv_flat = relu(conv_flat_pre_relu)
    s1 = params.W1 @ conv_flat + params.b1
    x1 = relu(s1)
    s2 = params.W2 @ x1 + params.b2
    P = softmax(s2)
    cache = {
        "conv_raw": conv_raw,
        "conv_flat_pre_relu": conv_flat_pre_relu,
        "conv_flat": conv_flat,
        "s1": s1,
        "x1": x1,
        "s2": s2,
        "P": P,
    }
    return P, cache


def backward_pass(
    MX: np.ndarray,
    Y: np.ndarray,
    cache: Dict[str, np.ndarray],
    params: NetParams,
    lam: float = 0.0,
    label_smoothing_eps: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Backward pass with optional label smoothing and L2 on weights.
    """
    n = Y.shape[1]
    K = Y.shape[0]

    if label_smoothing_eps > 0.0:
        Y_smooth = (1.0 - label_smoothing_eps) * Y + label_smoothing_eps / (K - 1) * (
            1.0 - Y
        )
    else:
        Y_smooth = Y

    P = cache["P"]
    x1 = cache["x1"]
    s1 = cache["s1"]
    conv_flat = cache["conv_flat"]
    conv_flat_pre_relu = cache["conv_flat_pre_relu"]

    G = -(Y_smooth - P)  # (K, n)

    grad_W2 = (G @ x1.T) / n + 2.0 * lam * params.W2
    grad_b2 = np.sum(G, axis=1, keepdims=True) / n

    G = params.W2.T @ G
    G = G * (s1 > 0)

    grad_W1 = (G @ conv_flat.T) / n + 2.0 * lam * params.W1
    grad_b1 = np.sum(G, axis=1, keepdims=True) / n

    G = params.W1.T @ G
    G = G * (conv_flat_pre_relu > 0)
    GG = G.reshape((MX.shape[0], params.Fs_flat.shape[1], n), order="C")

    MXt = np.transpose(MX, (1, 0, 2))
    grad_Fs_flat = np.einsum("ijn,jln->il", MXt, GG, optimize=True) / n
    grad_Fs_flat += 2.0 * lam * params.Fs_flat
    grad_b_conv = np.sum(GG, axis=(0, 2), keepdims=False).reshape((-1, 1)) / n

    return {
        "grad_Fs_flat": grad_Fs_flat,
        "grad_b_conv": grad_b_conv,
        "grad_W1": grad_W1,
        "grad_b1": grad_b1,
        "grad_W2": grad_W2,
        "grad_b2": grad_b2,
    }


def init_params(k: int, f: int, nf: int, nh: int, n_p: int, rng: np.random.Generator) -> NetParams:
    d0 = n_p * nf
    Fs_flat = (np.sqrt(2.0 / (3 * f * f)) * rng.standard_normal((3 * f * f, nf))).astype(np.float32)
    W1 = (np.sqrt(2.0 / d0) * rng.standard_normal((nh, d0))).astype(np.float32)
    W2 = (np.sqrt(2.0 / nh) * rng.standard_normal((k, nh))).astype(np.float32)
    b_conv = np.zeros((nf, 1), dtype=np.float32)
    b1 = np.zeros((nh, 1), dtype=np.float32)
    b2 = np.zeros((k, 1), dtype=np.float32)
    return NetParams(Fs_flat=Fs_flat, W1=W1, W2=W2, b_conv=b_conv, b1=b1, b2=b2)


def cyclic_lr(step_t: int, step_size: int, eta_min: float, eta_max: float) -> float:
    cycle = np.floor(1.0 + step_t / (2.0 * step_size))
    x = np.abs(step_t / step_size - 2.0 * cycle + 1.0)
    return float(eta_min + (eta_max - eta_min) * np.maximum(0.0, 1.0 - x))


def cyclic_lr_increasing(step_t: int, step1: int, eta_min: float, eta_max: float) -> float:
    local_t = step_t
    cycle_step = step1
    while local_t >= 2 * cycle_step:
        local_t -= 2 * cycle_step
        cycle_step *= 2
    return cyclic_lr(local_t, cycle_step, eta_min, eta_max)


def total_updates(cfg: TrainConfig) -> int:
    if not cfg.increasing_cycle:
        return 2 * cfg.step_size * cfg.n_cycles
    total = 0
    cycle_step = cfg.step_size
    for _ in range(cfg.n_cycles):
        total += 2 * cycle_step
        cycle_step *= 2
    return total


def predict_probs_batched(MX: np.ndarray, params: NetParams, batch_size: int = 1000) -> np.ndarray:
    n = MX.shape[2]
    k = params.b2.shape[0]
    P = np.zeros((k, n), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        p_sub, _ = forward_pass(MX[:, :, start:end], params)
        P[:, start:end] = p_sub
    return P


def cross_entropy_from_probs(P: np.ndarray, y: np.ndarray) -> float:
    probs = np.clip(P[y, np.arange(y.shape[0])], 1e-12, None)
    return float(np.mean(-np.log(probs)))


def accuracy_from_probs(P: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(P, axis=0) == y))


def regularization_cost(params: NetParams, lam: float) -> float:
    return float(lam * (np.sum(params.Fs_flat * params.Fs_flat) + np.sum(params.W1 * params.W1) + np.sum(params.W2 * params.W2)))


def evaluate_split(MX: np.ndarray, y: np.ndarray, params: NetParams, lam: float) -> Dict[str, float]:
    P = predict_probs_batched(MX, params, batch_size=1000)
    loss = cross_entropy_from_probs(P, y)
    acc = accuracy_from_probs(P, y)
    cost = loss + regularization_cost(params, lam)
    return {"loss": loss, "acc": acc, "cost": cost}


def build_mx_from_cifar_cols(X_cols: np.ndarray, f: int) -> np.ndarray:
    return build_mx(cifar_cols_to_images(X_cols), f).astype(np.float32)


def train_network(
    MX_train: np.ndarray,
    Y_train: np.ndarray,
    y_train: np.ndarray,
    MX_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
) -> Tuple[NetParams, Dict[str, List[float]]]:
    rng = np.random.default_rng(cfg.seed)
    n_p = MX_train.shape[0]
    params = init_params(k=10, f=cfg.f, nf=cfg.nf, nh=cfg.nh, n_p=n_p, rng=rng)

    n = MX_train.shape[2]
    updates = total_updates(cfg)
    epoch_indices = rng.permutation(n)
    ptr = 0

    hist: Dict[str, List[float]] = {
        "step": [],
        "eta": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    def log_metrics(step_i: int, eta_now: float) -> None:
        train_metrics = evaluate_split(MX_train, y_train, params, cfg.lam)
        test_metrics = evaluate_split(MX_test, y_test, params, cfg.lam)
        hist["step"].append(step_i)
        hist["eta"].append(float(eta_now))
        hist["train_loss"].append(train_metrics["loss"])
        hist["train_acc"].append(train_metrics["acc"])
        hist["test_loss"].append(test_metrics["loss"])
        hist["test_acc"].append(test_metrics["acc"])

    eta0 = cyclic_lr_increasing(0, cfg.step_size, cfg.eta_min, cfg.eta_max) if cfg.increasing_cycle else cyclic_lr(0, cfg.step_size, cfg.eta_min, cfg.eta_max)
    log_metrics(0, eta0)

    for t in range(updates):
        if ptr + cfg.n_batch > n:
            epoch_indices = rng.permutation(n)
            ptr = 0
        idx = epoch_indices[ptr : ptr + cfg.n_batch]
        ptr += cfg.n_batch

        MX_b = MX_train[:, :, idx]
        Y_b = Y_train[:, idx]
        P_b, cache_b = forward_pass(MX_b, params)
        del P_b
        grads = backward_pass(
            MX=MX_b,
            Y=Y_b,
            cache=cache_b,
            params=params,
            lam=cfg.lam,
            label_smoothing_eps=cfg.label_smoothing_eps,
        )

        if cfg.increasing_cycle:
            local_t = t
            cycle_step = cfg.step_size
            cycle_idx = 0
            while local_t >= 2 * cycle_step:
                local_t -= 2 * cycle_step
                cycle_step *= 2
                cycle_idx += 1
            eta_max_now = cfg.eta_max * (cfg.eta_max_decay ** cycle_idx)
            eta = cyclic_lr(local_t, cycle_step, cfg.eta_min, eta_max_now)
        else:
            cycle_idx = t // (2 * cfg.step_size)
            eta_max_now = cfg.eta_max * (cfg.eta_max_decay ** cycle_idx)
            eta = cyclic_lr(t, cfg.step_size, cfg.eta_min, eta_max_now)

        params.Fs_flat -= eta * grads["grad_Fs_flat"].astype(np.float32)
        params.b_conv -= eta * grads["grad_b_conv"].astype(np.float32)
        params.W1 -= eta * grads["grad_W1"].astype(np.float32)
        params.b1 -= eta * grads["grad_b1"].astype(np.float32)
        params.W2 -= eta * grads["grad_W2"].astype(np.float32)
        params.b2 -= eta * grads["grad_b2"].astype(np.float32)

        if (t + 1) % cfg.eval_every == 0 or (t + 1) == updates:
            log_metrics(t + 1, eta)

    return params, hist


def save_history_plot(
    hist: Dict[str, List[float]],
    out_png: str,
    title: str,
) -> None:
    steps = np.array(hist["step"])
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(steps, hist["train_loss"], label="train loss")
    plt.plot(steps, hist["test_loss"], label="test loss")
    plt.xlabel("update step")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(steps, hist["train_acc"], label="train acc")
    plt.plot(steps, hist["test_acc"], label="test acc")
    plt.xlabel("update step")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_bar_chart(labels: List[str], values: List[float], ylabel: str, title: str, out_png: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def maybe_quick_override(cfg: TrainConfig, quick: bool) -> TrainConfig:
    if not quick:
        return cfg
    return TrainConfig(
        f=cfg.f,
        nf=cfg.nf,
        nh=max(20, min(cfg.nh, 80)),
        lam=cfg.lam,
        n_batch=cfg.n_batch,
        eta_min=cfg.eta_min,
        eta_max=min(cfg.eta_max, 5e-2),
        n_cycles=1,
        step_size=80,
        increasing_cycle=cfg.increasing_cycle,
        label_smoothing_eps=cfg.label_smoothing_eps,
        eval_every=40,
        seed=cfg.seed,
    )


def run_ex3(cifar_root: str, out_dir: str, quick: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = load_ex3_data(cifar_root)
    train_X = data["train_X"]
    train_Y = data["train_Y"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    short_arches = [
        ("a1_f2_nf3_nh50", 2, 3, 50),
        ("a2_f4_nf10_nh50", 4, 10, 50),
        ("a3_f8_nf40_nh50", 8, 40, 50),
        ("a4_f16_nf160_nh50", 16, 160, 50),
    ]

    short_results: List[Dict[str, float]] = []
    for name, f, nf, nh in short_arches:
        cfg = TrainConfig(
            f=f,
            nf=nf,
            nh=nh,
            lam=0.003,
            n_batch=100,
            eta_min=1e-5,
            eta_max=1e-1,
            n_cycles=3,
            step_size=800,
            increasing_cycle=False,
            label_smoothing_eps=0.0,
            eval_every=200,
            seed=42,
        )
        cfg = maybe_quick_override(cfg, quick)
        MX_train = build_mx_from_cifar_cols(train_X, cfg.f)
        MX_test = build_mx_from_cifar_cols(test_X, cfg.f)

        t0 = time.time()
        _, hist = train_network(MX_train, train_Y, train_y, MX_test, test_y, cfg)
        t1 = time.time()
        save_history_plot(hist, os.path.join(out_dir, f"{name}_short_curves.png"), f"{name} short training")
        short_results.append(
            {
                "name": name,
                "f": f,
                "nf": nf,
                "nh": nh,
                "final_test_acc": float(hist["test_acc"][-1]),
                "train_time_sec": float(t1 - t0),
            }
        )

    labels = [r["name"] for r in short_results]
    accs = [r["final_test_acc"] for r in short_results]
    times = [r["train_time_sec"] for r in short_results]
    save_bar_chart(labels, accs, "Final test accuracy", "Exercise 3 short runs accuracy", os.path.join(out_dir, "ex3_short_test_accuracy.png"))
    save_bar_chart(labels, times, "Training time (sec)", "Exercise 3 short runs training time", os.path.join(out_dir, "ex3_short_train_time.png"))

    long_runs = [
        ("a2_long_f4_nf10_nh50", 4, 10, 50),
        ("a3_long_f8_nf40_nh50", 8, 40, 50),
        ("a2_wide_long_f4_nf40_nh50", 4, 40, 50),
    ]
    long_results: List[Dict[str, float]] = []
    for name, f, nf, nh in long_runs:
        cfg = TrainConfig(
            f=f,
            nf=nf,
            nh=nh,
            lam=0.003,
            n_batch=100,
            eta_min=1e-5,
            eta_max=1e-1,
            n_cycles=3,
            step_size=800,
            increasing_cycle=True,
            label_smoothing_eps=0.0,
            eval_every=400,
            seed=43,
        )
        cfg = maybe_quick_override(cfg, quick)
        MX_train = build_mx_from_cifar_cols(train_X, cfg.f)
        MX_test = build_mx_from_cifar_cols(test_X, cfg.f)

        t0 = time.time()
        _, hist = train_network(MX_train, train_Y, train_y, MX_test, test_y, cfg)
        t1 = time.time()
        save_history_plot(hist, os.path.join(out_dir, f"{name}_long_curves.png"), f"{name} long training")
        long_results.append(
            {
                "name": name,
                "f": f,
                "nf": nf,
                "nh": nh,
                "final_test_acc": float(hist["test_acc"][-1]),
                "train_time_sec": float(t1 - t0),
            }
        )

    with open(os.path.join(out_dir, "ex3_results.json"), "w", encoding="utf-8") as f:
        json.dump({"short_runs": short_results, "long_runs": long_results, "quick": quick}, f, indent=2)


def run_ex4(cifar_root: str, out_dir: str, quick: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = load_ex3_data(cifar_root)
    train_X = data["train_X"]
    train_Y = data["train_Y"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    runs = [
        ("arch5_no_label_smoothing", 0.0025, 0.0),
        ("arch5_with_label_smoothing", 0.0015, 0.1),
    ]
    results = []
    for name, lam, eps in runs:
        cfg = TrainConfig(
            f=4,
            nf=40,
            nh=300,
            lam=lam,
            n_batch=100,
            eta_min=1e-5,
            eta_max=1e-1,
            n_cycles=4,
            step_size=800,
            increasing_cycle=True,
            label_smoothing_eps=eps,
            eval_every=500,
            seed=44,
        )
        cfg = maybe_quick_override(cfg, quick)
        MX_train = build_mx_from_cifar_cols(train_X, cfg.f)
        MX_test = build_mx_from_cifar_cols(test_X, cfg.f)
        t0 = time.time()
        _, hist = train_network(MX_train, train_Y, train_y, MX_test, test_y, cfg)
        t1 = time.time()
        save_history_plot(hist, os.path.join(out_dir, f"{name}_curves.png"), name)
        results.append(
            {
                "name": name,
                "lam": lam,
                "label_smoothing_eps": eps,
                "final_test_acc": float(hist["test_acc"][-1]),
                "final_test_loss": float(hist["test_loss"][-1]),
                "train_time_sec": float(t1 - t0),
            }
        )

    with open(os.path.join(out_dir, "ex4_results.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": results, "quick": quick}, f, indent=2)


def run_bonus_5_1(cifar_root: str, out_dir: str, quick: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = load_ex3_data(cifar_root)
    train_X = data["train_X"]
    train_Y = data["train_Y"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    runs = [
        # Baseline
        ("baseline_arch5", TrainConfig(4, 40, 300, 0.0025, 100, 1e-5, 1e-1, 4, 800, True, 0.0, 500, 52)),
        # Improvement 1: wider network
        ("impr1_wider", TrainConfig(4, 64, 500, 0.0020, 100, 1e-5, 1e-1, 4, 800, True, 0.0, 500, 53)),
        # Improvement 2: label smoothing + tuned regularization
        ("impr2_wider_ls", TrainConfig(4, 64, 500, 0.0015, 100, 1e-5, 1e-1, 4, 800, True, 0.1, 500, 54)),
        # Improvement 3: decay eta_max per cycle
        ("impr3_wider_ls_decay", TrainConfig(4, 64, 500, 0.0015, 100, 1e-5, 1e-1, 4, 800, True, 0.1, 500, 55, eta_max_decay=0.9)),
    ]

    results: List[Dict[str, float]] = []
    for name, cfg in runs:
        cfg = maybe_quick_override(cfg, quick)
        MX_train = build_mx_from_cifar_cols(train_X, cfg.f)
        MX_test = build_mx_from_cifar_cols(test_X, cfg.f)
        t0 = time.time()
        _, hist = train_network(MX_train, train_Y, train_y, MX_test, test_y, cfg)
        t1 = time.time()
        save_history_plot(hist, os.path.join(out_dir, f"{name}_curves.png"), name)
        results.append(
            {
                "name": name,
                "f": cfg.f,
                "nf": cfg.nf,
                "nh": cfg.nh,
                "lam": cfg.lam,
                "eps": cfg.label_smoothing_eps,
                "eta_max_decay": cfg.eta_max_decay,
                "final_test_acc": float(hist["test_acc"][-1]),
                "final_test_loss": float(hist["test_loss"][-1]),
                "train_time_sec": float(t1 - t0),
            }
        )

    best = max(results, key=lambda r: r["final_test_acc"])
    labels = [r["name"] for r in results]
    accs = [r["final_test_acc"] for r in results]
    save_bar_chart(labels, accs, "Final test accuracy", "Bonus 5.1 improvements", os.path.join(out_dir, "bonus_5_1_accuracy_compare.png"))

    with open(os.path.join(out_dir, "bonus_5_1_results.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": results, "best": best, "quick": quick}, f, indent=2)


def _run_torch_timing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TrainConfig,
) -> Dict[str, float]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    class TorchPatchNet(nn.Module):
        def __init__(self, f: int, nf: int, nh: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, nf, kernel_size=f, stride=f, bias=True)
            n_p = (32 // f) * (32 // f)
            self.fc1 = nn.Linear(n_p * nf, nh)
            self.fc2 = nn.Linear(nh, 10)
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
            nn.init.zeros_(self.conv.bias)
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.conv(x))
            h = h.flatten(1)
            h = F.relu(self.fc1(h))
            return self.fc2(h)

    def to_img(x_cols: np.ndarray) -> np.ndarray:
        return np.transpose(x_cols.reshape((32, 32, 3, x_cols.shape[1]), order="F"), (3, 2, 0, 1)).astype(np.float32)

    Xtr = torch.from_numpy(to_img(X_train)).to(device)
    ytr = torch.from_numpy(y_train).to(device)
    Xte = torch.from_numpy(to_img(X_test)).to(device)
    yte = torch.from_numpy(y_test).to(device)

    model = TorchPatchNet(cfg.f, cfg.nf, cfg.nh).to(device)
    n = Xtr.shape[0]
    updates = total_updates(cfg)
    perm = torch.randperm(n, device=device)
    ptr = 0
    t0 = time.time()
    for t in range(updates):
        if ptr + cfg.n_batch > n:
            perm = torch.randperm(n, device=device)
            ptr = 0
        idx = perm[ptr : ptr + cfg.n_batch]
        ptr += cfg.n_batch

        if cfg.increasing_cycle:
            local_t = t
            cycle_step = cfg.step_size
            cycle_idx = 0
            while local_t >= 2 * cycle_step:
                local_t -= 2 * cycle_step
                cycle_step *= 2
                cycle_idx += 1
            eta_max_now = cfg.eta_max * (cfg.eta_max_decay ** cycle_idx)
            eta = cyclic_lr(local_t, cycle_step, cfg.eta_min, eta_max_now)
        else:
            cycle_idx = t // (2 * cfg.step_size)
            eta_max_now = cfg.eta_max * (cfg.eta_max_decay ** cycle_idx)
            eta = cyclic_lr(t, cfg.step_size, cfg.eta_min, eta_max_now)

        xb = Xtr[idx]
        yb = ytr[idx]
        logits = model(xb)
        ce = F.cross_entropy(logits, yb, reduction="mean")
        reg = cfg.lam * (
            torch.sum(model.conv.weight * model.conv.weight)
            + torch.sum(model.fc1.weight * model.fc1.weight)
            + torch.sum(model.fc2.weight * model.fc2.weight)
        )
        loss = ce + reg
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
    t1 = time.time()

    with torch.no_grad():
        pred = torch.argmax(model(Xte), dim=1)
        acc = torch.mean((pred == yte).float()).item()
    return {"train_time_sec": float(t1 - t0), "final_test_acc": float(acc)}


def run_bonus_5_2(cifar_root: str, out_dir: str, quick: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data = load_ex3_data(cifar_root)
    train_X = data["train_X"]
    train_Y = data["train_Y"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    arches = [
        ("cmp_f4_nf10_nh50", TrainConfig(4, 10, 50, 0.003, 100, 1e-5, 1e-1, 1, 300, False, 0.0, 150, 61)),
        ("cmp_f8_nf40_nh50", TrainConfig(8, 40, 50, 0.003, 100, 1e-5, 1e-1, 1, 300, False, 0.0, 150, 62)),
        ("cmp_f4_nf40_nh300", TrainConfig(4, 40, 300, 0.0025, 100, 1e-5, 1e-1, 1, 300, False, 0.0, 150, 63)),
    ]
    if quick:
        arches = [(name, maybe_quick_override(cfg, True)) for name, cfg in arches]

    results = []
    for name, cfg in arches:
        MX_train = build_mx_from_cifar_cols(train_X, cfg.f)
        MX_test = build_mx_from_cifar_cols(test_X, cfg.f)
        t0 = time.time()
        _, hist_np = train_network(MX_train, train_Y, train_y, MX_test, test_y, cfg)
        t1 = time.time()
        np_time = float(t1 - t0)
        np_acc = float(hist_np["test_acc"][-1])
        torch_res = _run_torch_timing(train_X, train_y, test_X, test_y, cfg)
        speed_ratio = np_time / torch_res["train_time_sec"] if torch_res["train_time_sec"] > 0 else 0.0
        results.append(
            {
                "name": name,
                "f": cfg.f,
                "nf": cfg.nf,
                "nh": cfg.nh,
                "numpy_train_time_sec": np_time,
                "numpy_test_acc": np_acc,
                "torch_train_time_sec": torch_res["train_time_sec"],
                "torch_test_acc": torch_res["final_test_acc"],
                "numpy_vs_torch_time_ratio": speed_ratio,
            }
        )

    labels = [r["name"] for r in results]
    np_times = [r["numpy_train_time_sec"] for r in results]
    torch_times = [r["torch_train_time_sec"] for r in results]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, np_times, width=width, label="numpy")
    plt.bar(x + width / 2, torch_times, width=width, label="torch")
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("Training time (sec)")
    plt.title("Bonus 5.2 NumPy vs Torch CPU timing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bonus_5_2_timing_compare.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "bonus_5_2_results.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": results, "quick": quick}, f, indent=2)


def run_bonus(cifar_root: str, out_dir: str, quick: bool = False) -> None:
    run_bonus_5_1(cifar_root, out_dir, quick=quick)
    run_bonus_5_2(cifar_root, out_dir, quick=quick)


def check_debug(debug_npz_path: str) -> None:
    data = np.load(debug_npz_path)
    X = data["X"]
    Y = data["Y"]
    Fs = data["Fs"]
    W1 = data["W1"]
    W2 = data["W2"]
    b1 = data["b1"]
    b2 = data["b2"]

    f = Fs.shape[0]
    nf = Fs.shape[3]
    n = X.shape[1]
    b_conv = np.zeros((nf, 1), dtype=X.dtype)  # Debug setup excludes conv bias.

    X_ims = cifar_cols_to_images(X)
    MX = build_mx(X_ims, f)
    Fs_flat = Fs.reshape((3 * f * f, nf), order="C")

    conv_ref = conv_slow(X_ims, Fs)
    conv_ref_flat = conv_ref.reshape((MX.shape[0], nf, n), order="C")
    conv_fast_out = conv_fast(MX, Fs_flat)

    print("max|MX - debug MX| =", np.max(np.abs(MX - data["MX"])))
    print(
        "max|conv_slow_flat - debug conv_outputs_mat| =",
        np.max(np.abs(conv_ref_flat - data["conv_outputs_mat"])),
    )
    print(
        "max|conv_fast - debug conv_outputs_mat| =",
        np.max(np.abs(conv_fast_out - data["conv_outputs_mat"])),
    )

    params = NetParams(
        Fs_flat=Fs_flat,
        W1=W1,
        W2=W2,
        b_conv=b_conv,
        b1=b1,
        b2=b2,
    )

    P, cache = forward_pass(MX, params)
    print("max|conv_flat - debug conv_flat| =", np.max(np.abs(cache["conv_flat"] - data["conv_flat"])))

    # Some debug files store X1 as (1, nh, n), so squeeze singleton axes.
    x1_debug = np.squeeze(data["X1"])
    print("max|x1 - debug X1| =", np.max(np.abs(cache["x1"] - x1_debug)))
    print("max|P - debug P| =", np.max(np.abs(P - data["P"])))

    grads = backward_pass(
        MX=MX,
        Y=Y,
        cache=cache,
        params=params,
        lam=0.0,
        label_smoothing_eps=0.0,
    )
    print(
        "max|grad_Fs_flat - debug| =",
        np.max(np.abs(grads["grad_Fs_flat"] - data["grad_Fs_flat"])),
    )
    print("max|grad_W1 - debug| =", np.max(np.abs(grads["grad_W1"] - data["grad_W1"])))
    print("max|grad_b1 - debug| =", np.max(np.abs(grads["grad_b1"] - data["grad_b1"])))
    print("max|grad_W2 - debug| =", np.max(np.abs(grads["grad_W2"] - data["grad_W2"])))
    print("max|grad_b2 - debug| =", np.max(np.abs(grads["grad_b2"] - data["grad_b2"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DD2424 Assignment 3 full pipeline.")
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "ex3", "ex4", "all", "bonus"],
        help="Which part to run.",
    )
    parser.add_argument(
        "--debug-npz",
        type=str,
        default="debug_info.npz",
        help="Path to debug_info.npz.",
    )
    parser.add_argument(
        "--cifar-root",
        type=str,
        default="Datasets/cifar-10-batches-py",
        help="Path to CIFAR-10 python batches directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_assignment3",
        help="Directory to save figures and json outputs.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run reduced updates for smoke testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "debug":
        check_debug(args.debug_npz)
        return
    if args.mode in {"ex3", "all"}:
        run_ex3(args.cifar_root, args.out_dir, quick=args.quick)
    if args.mode in {"ex4", "all"}:
        run_ex4(args.cifar_root, args.out_dir, quick=args.quick)
    if args.mode == "bonus":
        run_bonus(args.cifar_root, args.out_dir, quick=args.quick)


if __name__ == "__main__":
    main()
