"""
Minimal NumPy-only CNN for binary image classification.

Class 0: images in ../data/test_data_256x256 (excluding target/)
Class 1: images in ../data/test_data_256x256/target
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_images(
    base_dir: Path,
    target_subdir: str = "target",
    img_size: int = 256,
    max_images: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    class0_dir = base_dir
    class1_dir = base_dir / target_subdir
    if not class0_dir.exists():
        raise FileNotFoundError(f"Missing class-0 dir: {class0_dir}")
    if not class1_dir.exists():
        raise FileNotFoundError(f"Missing class-1 dir: {class1_dir}")

    class0_files = sorted(
        p for p in class0_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    class1_files = sorted(
        p for p in class1_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )

    if max_images is not None:
        class0_files = class0_files[:max_images]
        class1_files = class1_files[:max_images]

    def load_one(path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    x0 = np.stack([load_one(p) for p in class0_files], axis=0)
    x1 = np.stack([load_one(p) for p in class1_files], axis=0)
    y0 = np.zeros(len(x0), dtype=np.int64)
    y1 = np.ones(len(x1), dtype=np.int64)

    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([y0, y1], axis=0)
    x = x[:, None, :, :]  # (N, 1, H, W)
    return x, y


class Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
        self.rng = rng or np.random.default_rng()

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        self.w = self.rng.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels,), dtype=np.float32)
        self.xpad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1
        out = np.zeros((n, self.out_channels, h_out, w_out), dtype=np.float32)

        xpad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge")
        self.xpad = xpad

        for i in range(n):
            for oc in range(self.out_channels):
                for r in range(h_out):
                    rs = r * s
                    for c_out in range(w_out):
                        cs = c_out * s
                        patch = xpad[i, :, rs:rs + k, cs:cs + k]
                        out[i, oc, r, c_out] = np.sum(patch * self.w[oc]) + self.b[oc]
        return out

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        n, _, h_out, w_out = d_out.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        d_w = np.zeros_like(self.w)
        d_b = np.zeros_like(self.b)
        d_xpad = np.zeros_like(self.xpad)

        for i in range(n):
            for oc in range(self.out_channels):
                d_b[oc] += np.sum(d_out[i, oc])
                for r in range(h_out):
                    rs = r * s
                    for c_out in range(w_out):
                        cs = c_out * s
                        patch = self.xpad[i, :, rs:rs + k, cs:cs + k]
                        d_w[oc] += d_out[i, oc, r, c_out] * patch
                        d_xpad[i, :, rs:rs + k, cs:cs + k] += d_out[i, oc, r, c_out] * self.w[oc]

        self.w -= lr * d_w
        self.b -= lr * d_b

        if p == 0:
            return d_xpad
        return d_xpad[:, :, p:-p, p:-p]


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * self.mask


class MaxPool2D:
    def __init__(self, pool: int = 2, stride: int | None = None):
        self.pool = pool
        self.stride = pool if stride is None else stride
        self.argmax = None
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        p = self.pool
        s = self.stride
        h_out = (h - p) // s + 1
        w_out = (w - p) // s + 1
        out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)
        self.argmax = np.zeros((n, c, h_out, w_out), dtype=np.int64)
        self.x_shape = x.shape
        self.out_shape = out.shape

        for i in range(n):
            for ch in range(c):
                for r in range(h_out):
                    rs = r * s
                    for c_out in range(w_out):
                        cs = c_out * s
                        window = x[i, ch, rs:rs + p, cs:cs + p]
                        idx = np.argmax(window)
                        out[i, ch, r, c_out] = window.reshape(-1)[idx]
                        self.argmax[i, ch, r, c_out] = idx
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        n, c, h_out, w_out = d_out.shape
        p = self.pool
        s = self.stride
        h, w = self.x_shape[2], self.x_shape[3]
        d_x = np.zeros((n, c, h, w), dtype=d_out.dtype)

        for i in range(n):
            for ch in range(c):
                for r in range(h_out):
                    rs = r * s
                    for c_out in range(w_out):
                        cs = c_out * s
                        idx = self.argmax[i, ch, r, c_out]
                        rr = idx // p
                        cc = idx % p
                        d_x[i, ch, rs + rr, cs + cc] += d_out[i, ch, r, c_out]
        return d_x


class Dense:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.w = self.rng.uniform(-limit, limit, (in_dim, out_dim))
        self.b = np.zeros((out_dim,), dtype=np.float32)
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.w + self.b

    def backward(self, d_out: np.ndarray, lr: float) -> np.ndarray:
        d_w = self.x.T @ d_out
        d_b = d_out.sum(axis=0)
        d_x = d_out @ self.w.T
        self.w -= lr * d_w
        self.b -= lr * d_b
        return d_x


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    d_logits = probs
    d_logits[np.arange(n), y] -= 1.0
    d_logits /= n
    return loss, d_logits


class SimpleCNN:
    def __init__(self, img_size: int, rng: np.random.Generator):
        self.conv1 = Conv2D(1, 4, kernel_size=3, stride=1, rng=rng)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool=2)
        self.conv2 = Conv2D(4, 8, kernel_size=3, stride=1, rng=rng)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool=2)

        # After two pool layers, spatial dims are quartered twice.
        reduced = img_size // 4
        self.fc1 = Dense(8 * reduced * reduced, 32, rng=rng)
        self.relu3 = ReLU()
        self.fc2 = Dense(32, 2, rng=rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, d_logits: np.ndarray, lr: float) -> None:
        d = self.fc2.backward(d_logits, lr)
        d = self.relu3.backward(d)
        d = self.fc1.backward(d, lr)
        d = d.reshape(self.pool2.out_shape)
        d = self.pool2.backward(d)
        d = self.relu2.backward(d)
        d = self.conv2.backward(d, lr)
        d = self.pool1.backward(d)
        d = self.relu1.backward(d)
        _ = self.conv1.backward(d, lr)


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return (preds == y).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy CNN on two image classes.")
    parser.add_argument("--data-dir", type=str, default="../data/test_data_256x256")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    x, y = load_images(Path(args.data_dir), img_size=args.img_size, max_images=args.max_images)

    indices = np.arange(len(x))
    rng.shuffle(indices)
    split = max(1, int(0.8 * len(x)))
    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    model = SimpleCNN(args.img_size, rng)

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm]

        total_loss = 0.0
        total_acc = 0.0
        steps = 0

        for i in range(0, len(x_train), args.batch):
            xb = x_train[i:i + args.batch]
            yb = y_train[i:i + args.batch]
            logits = model.forward(xb)
            loss, d_logits = softmax_cross_entropy(logits, yb)
            model.backward(d_logits, args.lr)
            total_loss += loss
            total_acc += accuracy(logits, yb)
            steps += 1

        val_logits = model.forward(x_val) if len(x_val) else logits
        val_acc = accuracy(val_logits, y_val) if len(x_val) else 0.0
        print(
            f"epoch {epoch:02d} | "
            f"loss {total_loss / max(1, steps):.4f} | "
            f"train acc {total_acc / max(1, steps):.3f} | "
            f"val acc {val_acc:.3f}"
        )


if __name__ == "__main__":
    main()
