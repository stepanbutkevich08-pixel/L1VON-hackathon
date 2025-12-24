"""
gen.py — воспроизведение вычисления координат центров с нуля.
Алгоритм: MiniBatchKMeans → прогрев → TTT-оптимизация → swap-поиск → мягкое уточнение (PyTorch).
Запуск:
    python gen.py          # клиенты из clinis-data.csv или clinic-data.csv, выводит TTT, сохраняет clinics.csv и centers.npy
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

# Глобальные параметры
N_CLINICS = 250
ALPHA = 0.05
BETA = 0.02
KMEANS_BATCH = 4096
KMEANS_INIT = 20
KMEANS_ITER = 200
CAPACITY_WARM_ITERS = 6
TTT_ITERS = 80
SWAP_TRIES = 20
SWAP_REFINE_ITERS = 12
SEED = 42

ANNEAL_SCHEDULE = [
    (0.08, 25),
    (0.05, 25),
    (0.03, 25),
    (0.02, 25),
    (0.015, 25),
    (0.012, 25),
    (0.01, 25),
    (0.008, 25),
]


def load_points() -> Tuple[pd.DataFrame, np.ndarray]:
    path = "clinis-data.csv" if os.path.exists("clinis-data.csv") else "clinic-data.csv"
    df = pd.read_csv(path)
    pts = df[["x", "y"]].to_numpy(float)
    return df, pts


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def metric_ttt(points: np.ndarray, centers: np.ndarray) -> float:
    d = pairwise_dist(points, centers)
    best = np.min(d, axis=1)
    return float(np.sum(ALPHA * best + BETA * (best ** 2)))


def capacity_warm(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    cur = centers.copy()
    for _ in range(CAPACITY_WARM_ITERS):
        d = pairwise_dist(points, cur)
        labels = d.argmin(axis=1)
        new = cur.copy()
        for j in range(len(cur)):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                continue
            new[j] = points[idx].mean(axis=0)
        cur = new
    return cur


def update_centers_ttt(points: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    new = centers.copy()
    for j in range(len(centers)):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        cluster = points[idx]
        c = centers[j]
        for _ in range(20):
            diff = cluster - c
            r = np.linalg.norm(diff, axis=1)
            r[r == 0] = 1e-9
            w = ALPHA / r + 2 * BETA
            c_new = (w[:, None] * cluster).sum(axis=0) / w.sum()
            if np.allclose(c_new, c, rtol=1e-7, atol=1e-8):
                c = c_new
                break
            c = c_new
        new[j] = c
    return new


def ttt_optimize(points: np.ndarray, centers: np.ndarray, iters: int) -> np.ndarray:
    best = centers.copy()
    best_val = metric_ttt(points, best)
    cur = centers.copy()
    for _ in range(iters):
        d = pairwise_dist(points, cur)
        labels = d.argmin(axis=1)
        cur = update_centers_ttt(points, labels, cur)
        val = metric_ttt(points, cur)
        if val < best_val:
            best_val = val
            best = cur.copy()
    return best


def swap_search(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    best = centers.copy()
    best_val = metric_ttt(points, best)
    for _ in range(SWAP_TRIES):
        cand = best.copy()
        j = int(rng.integers(0, len(cand)))
        idx = int(rng.integers(0, len(points)))
        cand[j] = points[idx]
        cand = ttt_optimize(points, cand, SWAP_REFINE_ITERS)
        val = metric_ttt(points, cand)
        if val < best_val:
            best = cand
            best_val = val
    return best


def torch_refine(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    pts = torch.tensor(points, dtype=torch.float32)
    cent = torch.tensor(centers, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([cent], lr=0.008)
    best_cent = cent.detach().clone()
    best_val = float("inf")
    for tau, steps in ANNEAL_SCHEDULE:
        for _ in range(steps):
            diff = pts.unsqueeze(1) - cent.unsqueeze(0)
            dist = torch.norm(diff, dim=2)
            w = torch.softmax(-dist / tau, dim=1)
            exp_r = (w * dist).sum(dim=1)
            exp_r2 = (w * (dist ** 2)).sum(dim=1)
            loss = (ALPHA * exp_r + BETA * exp_r2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            dist = torch.norm(pts.unsqueeze(1) - cent.unsqueeze(0), dim=2)
            best = dist.min(dim=1).values
            val = (ALPHA * best + BETA * best * best).sum().item()
            if val < best_val:
                best_val = val
                best_cent = cent.detach().clone()
        for g in optimizer.param_groups:
            g["lr"] *= 0.7
    return best_cent.cpu().numpy()


def compute_centers(points: np.ndarray) -> np.ndarray:
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLINICS,
        random_state=SEED,
        batch_size=KMEANS_BATCH,
        n_init=KMEANS_INIT,
        max_iter=KMEANS_ITER,
    )
    kmeans.fit(points)
    centers = kmeans.cluster_centers_.astype(float)
    centers = capacity_warm(points, centers)
    centers = ttt_optimize(points, centers, TTT_ITERS)
    centers = swap_search(points, centers)
    centers = torch_refine(points, centers)
    return centers


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    _, points = load_points()
    centers = compute_centers(points)
    pd.DataFrame(centers, columns=["x", "y"]).to_csv("clinics.csv", index=False)
    np.save("centers.npy", centers)
    print(f"Saved clinics.csv with {len(centers)} clinics")
    print(f"TTT={metric_ttt(points, centers):.6f}")


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    main()
