import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans

import clinic_placement_solver as solver

# Constants
N_CLINICS = 250
ALPHA = 0.05
BETA = 0.02


def metric_ttt(points: np.ndarray, centers: np.ndarray) -> float:
    d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    best = np.sqrt(d2.min(axis=1))
    return float(np.sum(ALPHA * best + BETA * best * best))


def load_points() -> Tuple[pd.DataFrame, np.ndarray]:
    candidates: List[str] = ["clinis-data.csv", "clinic-data.csv"]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("Neither clinis-data.csv nor clinic-data.csv found.")
    df = pd.read_csv(path)
    missing = [c for c in solver.REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    pts = df[["x", "y"]].to_numpy(float)
    return df, pts


def build_base_centers(df: pd.DataFrame, points: np.ndarray) -> np.ndarray:
    """Fast approximation of the earlier 2337 TTT solution."""
    weights = solver.build_client_weights(df)

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLINICS,
        random_state=42,
        batch_size=4096,
        n_init=5,
    )
    kmeans.fit(points)
    centers = kmeans.cluster_centers_.astype(float)

    # Warm capacity-aware positioning (short)
    centers, _ = solver.capacity_warm_opt(points, centers, weights, n_iters=6)
    # TTT-only refinement
    centers, _ = solver.optimize_centers_for_ttt(points, centers, n_iters=80, log_every=0)
    # Swap search
    centers, _ = solver.swap_local_search(points, centers, n_swaps=25, refine_iters=12, rng_seed=123)
    return centers


def torch_refine(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Soft-assignment annealing to push TTT toward ~2334."""
    device = "cpu"
    pts = torch.tensor(points, dtype=torch.float32, device=device)
    cent = torch.tensor(centers, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([cent], lr=0.008)
    best_cent = cent.detach().clone()
    best_val = float("inf")

    schedule = [
        (0.08, 25),
        (0.05, 25),
        (0.03, 25),
        (0.02, 25),
        (0.015, 25),
        (0.012, 25),
        (0.01, 25),
        (0.008, 25),
    ]

    for tau, steps in schedule:
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
            ttt = (ALPHA * best + BETA * best * best).sum().item()
            if ttt < best_val:
                best_val = ttt
                best_cent = cent.detach().clone()

        for g in optimizer.param_groups:
            g["lr"] *= 0.7

    return best_cent.cpu().numpy()


def main() -> None:
    # deterministic seeds
    np.random.seed(0)
    torch.manual_seed(0)

    df, points = load_points()
    centers = build_base_centers(df, points)
    centers = torch_refine(points, centers)

    pd.DataFrame(centers, columns=["x", "y"]).to_csv("clinics.csv", index=False)
    ttt = metric_ttt(points, centers)
    print(f"Saved clinics.csv with {len(centers)} clinics")
    print(f"TTT={ttt:.6f}")


if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    main()
