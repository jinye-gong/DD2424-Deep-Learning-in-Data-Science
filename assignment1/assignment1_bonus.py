from __future__ import annotations

import copy
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import Assignment1 as A


def build_horizontal_flip_indices() -> np.ndarray:
    """CIFAR-10 column layout: 3072 = 3 * 1024 (R,G,B planes), per assignment PDF."""
    aa = np.int32(np.arange(32)).reshape((32, 1))
    bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((32 * 32, 1)) + np.tile(bb, (32, 1))
    inds = np.vstack((ind_flip, 1024 + ind_flip))
    inds = np.vstack((inds, 2048 + ind_flip))
    return inds.ravel()


def apply_horizontal_flip_batch(
    X: np.ndarray, rng: np.random.Generator, p: float, row_perm: np.ndarray
) -> np.ndarray:
    """Each column flipped i.i.d. with probability p."""
    n = X.shape[1]
    flip = rng.random(n) < p
    if not np.any(flip):
        return X
    out = X.copy()
    cols = np.nonzero(flip)[0]
    out[:, cols] = X[np.ix_(row_perm, cols)]
    return out


def load_full_train_split(
    cifar_root: str, val_size: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Concatenate data_batch_1..5, random split: N-val train, val_size validation."""
    parts_X: List[np.ndarray] = []
    parts_Y: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for i in range(1, 6):
        path = os.path.join(cifar_root, f"data_batch_{i}")
        Xb, Yb, yb = A.load_batch(path)
        parts_X.append(Xb)
        parts_Y.append(Yb)
        parts_y.append(yb)
    X_all = np.concatenate(parts_X, axis=1)
    Y_all = np.concatenate(parts_Y, axis=1)
    y_all = np.concatenate(parts_y, axis=0)
    n = X_all.shape[1]
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]
    train_X = X_all[:, tr_idx]
    train_Y = Y_all[:, tr_idx]
    train_y = y_all[tr_idx]
    val_X = X_all[:, val_idx]
    val_Y = Y_all[:, val_idx]
    val_y = y_all[val_idx]

    test_X, test_Y, test_y = A.load_batch(os.path.join(cifar_root, "test_batch"))
    train_X, val_X, test_X, mean_X, std_X = A.normalize_data(train_X, val_X, test_X)
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


def apply_network_sigmoid(X: np.ndarray, network: Dict[str, np.ndarray]) -> np.ndarray:
    s = network["W"] @ X + network["b"]
    return 1.0 / (1.0 + np.exp(-s))


def compute_loss_multibce(P: np.ndarray, Y: np.ndarray) -> float:
    K = P.shape[0]
    eps = 1e-15
    Pc = np.clip(P, eps, 1.0 - eps)
    term = (1.0 - Y) * np.log(1.0 - Pc) + Y * np.log(Pc)
    return float(np.mean(-np.sum(term, axis=0) / K))


def compute_cost_sigmoid(
    P: np.ndarray, Y: np.ndarray, network: Dict[str, np.ndarray], lam: float
) -> float:
    return float(compute_loss_multibce(P, Y) + lam * np.sum(network["W"] ** 2))


def backward_pass_sigmoid(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    network: Dict[str, np.ndarray],
    lam: float,
) -> Dict[str, np.ndarray]:
    K = Y.shape[0]
    n = X.shape[1]
    G = (P - Y) / K
    grad_W = (G @ X.T) / n + 2.0 * lam * network["W"]
    grad_b = np.mean(G, axis=1, keepdims=True)
    return {"W": grad_W, "b": grad_b}


def mini_batch_gd_improved(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    y_val: np.ndarray,
    GDparams: Dict,
    init_net: Dict[str, np.ndarray],
    lam: float,
    rng: np.random.Generator,
    flip_indices: np.ndarray,
    augment_prob: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    n_batch = int(GDparams["n_batch"])
    eta0 = float(GDparams["eta"])
    n_epochs = int(GDparams["n_epochs"])
    decay_every = int(GDparams.get("decay_every", 15))
    decay_factor = float(GDparams.get("decay_factor", 0.1))

    net = copy.deepcopy(init_net)
    n = X_train.shape[1]
    eta = eta0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_cost": [],
        "val_cost": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(n_epochs):
        if epoch > 0 and decay_every > 0 and epoch % decay_every == 0:
            eta *= decay_factor
            print(f"  (step decay) eta -> {eta:g}")

        perm = rng.permutation(n)
        X_shuf = X_train[:, perm]
        Y_shuf = Y_train[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            X_batch = apply_horizontal_flip_batch(
                X_shuf[:, j_start:j_end], rng, augment_prob, flip_indices
            )
            Y_batch = Y_shuf[:, j_start:j_end]
            P_batch = A.apply_network(X_batch, net)
            grads = A.backward_pass(X_batch, Y_batch, P_batch, net, lam)
            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        P_train = A.apply_network(X_train, net)
        P_val = A.apply_network(X_val, net)
        history["train_loss"].append(A.compute_loss(P_train, y_train))
        history["val_loss"].append(A.compute_loss(P_val, y_val))
        history["train_cost"].append(A.compute_cost(P_train, y_train, net, lam))
        history["val_cost"].append(A.compute_cost(P_val, y_val, net, lam))
        history["train_acc"].append(A.compute_accuracy(P_train, y_train))
        history["val_acc"].append(A.compute_accuracy(P_val, y_val))
        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | eta={eta:g} | "
            f"train_cost={history['train_cost'][-1]:.4f} | "
            f"val_cost={history['val_cost'][-1]:.4f} | "
            f"val_acc={history['val_acc'][-1]*100:.2f}%"
        )

    return net, history


def mini_batch_gd_sigmoid(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    y_val: np.ndarray,
    GDparams: Dict[str, float],
    init_net: Dict[str, np.ndarray],
    lam: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    n_batch = int(GDparams["n_batch"])
    eta = float(GDparams["eta"])
    n_epochs = int(GDparams["n_epochs"])

    net = copy.deepcopy(init_net)
    n = X_train.shape[1]
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_cost": [],
        "val_cost": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        X_shuf = X_train[:, perm]
        Y_shuf = Y_train[:, perm]

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            X_batch = X_shuf[:, j_start:j_end]
            Y_batch = Y_shuf[:, j_start:j_end]
            P_batch = apply_network_sigmoid(X_batch, net)
            grads = backward_pass_sigmoid(X_batch, Y_batch, P_batch, net, lam)
            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        P_train = apply_network_sigmoid(X_train, net)
        P_val = apply_network_sigmoid(X_val, net)
        history["train_loss"].append(compute_loss_multibce(P_train, Y_train))
        history["val_loss"].append(compute_loss_multibce(P_val, Y_val))
        history["train_cost"].append(compute_cost_sigmoid(P_train, Y_train, net, lam))
        history["val_cost"].append(compute_cost_sigmoid(P_val, Y_val, net, lam))
        history["train_acc"].append(A.compute_accuracy(P_train, y_train))
        history["val_acc"].append(A.compute_accuracy(P_val, y_val))
        print(
            f"[sigmoid BCE] Epoch {epoch + 1:02d}/{n_epochs} | "
            f"train_cost={history['train_cost'][-1]:.4f} | "
            f"val_cost={history['val_cost'][-1]:.4f}"
        )

    return net, history


def plot_compare_curves(
    hist_a: Dict[str, List[float]],
    hist_b: Dict[str, List[float]],
    label_a: str,
    label_b: str,
    outfile: str,
    title: str,
) -> None:
    e = np.arange(1, len(hist_a["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(e, hist_a["train_loss"], label=f"{label_a} train")
    plt.plot(e, hist_a["val_loss"], "--", label=f"{label_a} val")
    plt.plot(e, hist_b["train_loss"], label=f"{label_b} train")
    plt.plot(e, hist_b["val_loss"], "--", label=f"{label_b} val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def histogram_true_class_probs(
    P: np.ndarray,
    y: np.ndarray,
    title: str,
    outfile: str,
) -> None:
    n = y.shape[0]
    pred = np.argmax(P, axis=0)
    correct = pred == y
    idx = np.arange(n)
    p_true = P[y, idx]
    p_corr = p_true[correct]
    p_wrong = p_true[~correct]

    plt.figure(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    plt.hist(p_corr, bins=bins, alpha=0.6, label=f"correct (n={len(p_corr)})")
    plt.hist(p_wrong, bins=bins, alpha=0.6, label=f"incorrect (n={len(p_wrong)})")
    plt.xlabel("predicted probability of true class")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def run_bonus_2_1(cifar_root: str, rng: np.random.Generator) -> None:
    print("\n=== Bonus 2.1: full data + flip aug + step decay ===\n")
    flip_idx = build_horizontal_flip_indices()
    data = load_full_train_split(cifar_root, val_size=1000, rng=rng)
    print(
        f"train {data['train_X'].shape[1]}, val {data['val_X'].shape[1]}, "
        f"lam=0.01 (lower with augmentation per assignment)"
    )
    K, d = 10, data["train_X"].shape[0]
    init_net = A.init_network(K, d, rng)
    GDparams = {
        "n_batch": 100,
        "eta": 0.001,
        "n_epochs": 40,
        "decay_every": 15,
        "decay_factor": 0.1,
    }
    lam = 0.01
    trained, hist = mini_batch_gd_improved(
        data["train_X"],
        data["train_Y"],
        data["train_y"],
        data["val_X"],
        data["val_Y"],
        data["val_y"],
        GDparams,
        init_net,
        lam,
        rng,
        flip_idx,
        augment_prob=0.5,
    )
    P_test = A.apply_network(data["test_X"], trained)
    acc = A.compute_accuracy(P_test, data["test_y"])
    print(f"\nBonus 2.1 final test accuracy: {acc * 100:.2f}%")

    ep = np.arange(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(ep, hist["train_loss"], label="train loss")
    plt.plot(ep, hist["val_loss"], label="val loss")
    plt.plot(ep, hist["train_cost"], "--", label="train cost")
    plt.plot(ep, hist["val_cost"], "--", label="val cost")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Bonus 2.1: softmax + improvements")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bonus_curves_λ0p01_η0p001.png", dpi=150)
    plt.close()
    A.visualize_weights(trained, "bonus_λ0p01")


def run_bonus_2_2(cifar_root: str, rng: np.random.Generator) -> None:
    print("\n=== Bonus 2.2: sigmoid + multi BCE vs softmax (same split as Ex1) ===\n")
    base = A.load_all_data(cifar_root)

    K, d = 10, base["train_X"].shape[0]
    n_batch, n_epochs = 100, 40

    net_soft_trained = A.init_network(K, d, rng)
    net_soft_trained, hist_soft = A.mini_batch_gd(
        base["train_X"],
        base["train_Y"],
        base["train_y"],
        base["val_X"],
        base["val_Y"],
        base["val_y"],
        {"n_batch": n_batch, "eta": 0.001, "n_epochs": n_epochs},
        net_soft_trained,
        0.0,
        rng,
    )
    P_t_soft = A.apply_network(base["test_X"], net_soft_trained)
    acc_soft = A.compute_accuracy(P_t_soft, base["test_y"])

    net_sig = A.init_network(K, d, rng)
    net_sig_trained, hist_sig = mini_batch_gd_sigmoid(
        base["train_X"],
        base["train_Y"],
        base["train_y"],
        base["val_X"],
        base["val_Y"],
        base["val_y"],
        {"n_batch": n_batch, "eta": 0.01, "n_epochs": n_epochs},
        net_sig,
        0.0,
        rng,
    )
    P_t_sig = apply_network_sigmoid(base["test_X"], net_sig_trained)
    acc_sig = A.compute_accuracy(P_t_sig, base["test_y"])

    print(f"Softmax+CE test accuracy: {acc_soft * 100:.2f}%")
    print(f"Sigmoid+multi-BCE test accuracy: {acc_sig * 100:.2f}%")

    plot_compare_curves(
        hist_soft,
        hist_sig,
        "softmax+CE",
        "sigmoid+BCE",
        "bonus_compare_softmax_η0p001_vs_sigmoid_η0p01.png",
        "Train/val loss: softmax cross-entropy vs sigmoid multi-BCE",
    )

    histogram_true_class_probs(
        P_t_soft,
        base["test_y"],
        "Softmax: prob of true class (test)",
        "bonus_hist_test_softmax_η0p001.png",
    )
    histogram_true_class_probs(
        P_t_sig,
        base["test_y"],
        "Sigmoid: prob of true class (test)",
        "bonus_hist_test_sigmoid_η0p01.png",
    )


def check_sigmoid_gradients(
    train_X: np.ndarray, train_Y: np.ndarray, train_y: np.ndarray, rng: np.random.Generator
) -> None:
    from torch_gradient_computations import ComputeGradsSigmoidMultiBCE

    d_small, n_small, lam = 10, 3, 0.1
    small_net = A.init_network(10, d_small, rng)
    X_small = train_X[:d_small, :n_small]
    Y_small = train_Y[:, :n_small]
    y_small = train_y[:n_small]
    P = apply_network_sigmoid(X_small, small_net)
    my_g = backward_pass_sigmoid(X_small, Y_small, P, small_net, lam)
    torch_g = ComputeGradsSigmoidMultiBCE(X_small, y_small, small_net, lam)
    eps = 1e-12
    rel_W = np.max(
        np.abs(my_g["W"] - torch_g["W"])
        / np.maximum(eps, np.abs(my_g["W"]) + np.abs(torch_g["W"]))
    )
    rel_b = np.max(
        np.abs(my_g["b"] - torch_g["b"])
        / np.maximum(eps, np.abs(my_g["b"]) + np.abs(torch_g["b"]))
    )
    print(f"Sigmoid BCE gradient check: W={rel_W:.3e}, b={rel_b:.3e}")


if __name__ == "__main__":
    cifar_root = "Datasets/cifar-10-batches-py"
    if not os.path.isdir(cifar_root):
        raise FileNotFoundError(f"Missing {cifar_root}")

    base = A.load_all_data(cifar_root)
    rng_chk = np.random.default_rng(42)
    check_sigmoid_gradients(
        base["train_X"], base["train_Y"], base["train_y"], rng_chk
    )

    rng1 = np.random.default_rng(42)
    run_bonus_2_1(cifar_root, rng1)

    rng2 = np.random.default_rng(42)
    run_bonus_2_2(cifar_root, rng2)
    print(
        "\nDone. Figures: bonus_curves_λ0p01_η0p001.png, weights_bonus_λ0p01.png, "
        "bonus_compare_softmax_η0p001_vs_sigmoid_η0p01.png, bonus_hist_*.png"
    )
