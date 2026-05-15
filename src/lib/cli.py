"""Tiny 2-layer MLP trained on two-moons with NumPy."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def make_moons(rng, n, noise):
    n1 = n // 2
    t = np.linspace(0, np.pi, n1)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    X = np.vstack([a, b]) + rng.normal(0, noise, (n1 * 2, 2))
    y = np.hstack([np.zeros(n1), np.ones(n1)]).astype(int)
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_forward(W1, b1, W2, b2):
    def forward(X):
        h = np.tanh(X @ W1 + b1)
        p = sigmoid(h @ W2 + b2).ravel()
        return h, p
    return forward


def boundary_plot(path, title, forward, X, y):
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1.0, 1.5, 200))
    _, p = forward(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, p.reshape(xx.shape), levels=20, cmap="RdBu", alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolor="k", s=25)
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def parse_args():
    default_out = Path(__file__).resolve().parents[1] / "docs" / "public"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=400, help="number of training points")
    p.add_argument("--noise", type=float, default=0.2, help="gaussian noise on moons")
    p.add_argument("--hidden", type=int, default=16, help="hidden layer width")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05, help="learning rate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=default_out, help="output directory for plots")
    p.add_argument("--prefix", default="", help="filename prefix for outputs")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    X, y = make_moons(rng, args.samples, args.noise)
    H = args.hidden
    W1 = rng.normal(0, 0.5, (2, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 0.5, (H, 1)); b2 = np.zeros(1)
    forward = make_forward(W1, b1, W2, b2)

    pre = args.prefix
    boundary_plot(args.out / f"{pre}boundary_before.png", "Decision boundary — epoch 0", forward, X, y)

    losses = []
    for _ in range(args.epochs):
        h, p = forward(X)
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        losses.append(loss)
        dz2 = (p - y).reshape(-1, 1) / len(y)
        dW2 = h.T @ dz2; db2 = dz2.sum(0)
        dh = dz2 @ W2.T * (1 - h ** 2)
        dW1 = X.T @ dh; db1 = dh.sum(0)
        W1 -= args.lr * dW1; b1 -= args.lr * db1
        W2 -= args.lr * dW2; b2 -= args.lr * db2

    acc = ((forward(X)[1] > 0.5) == y).mean()
    print(f"final loss={losses[-1]:.4f}  acc={acc:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("epoch"); plt.ylabel("binary cross-entropy"); plt.title("Training loss")
    plt.tight_layout(); plt.savefig(args.out / f"{pre}loss_curve.png", dpi=120); plt.close()

    boundary_plot(args.out / f"{pre}boundary_after.png", f"Decision boundary — epoch {args.epochs} (acc={acc:.2f})", forward, X, y)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
