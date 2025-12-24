
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

DATA_CSV_PATH = "clinic-data.csv"
OUT_CLINICS_CSV = "clinics.csv"
PLOTS_DIR = "plots"

# Task constraints
N_CLINICS = 250
CAPACITY_L = 1000

# Task TTT parameters
ALPHA = 0.05
BETA = 0.02

# Optimization
WEISZFELD_STEPS = 15
RANDOM_STATE = 42
KMEANS_BATCH_SIZE = 8192
TTT_ITERS = 120          # iterations focused on TTT minimization (no capacity)
TTT_LOG_EVERY = 0
N_RESTARTS = 1           # k-means restarts (we rely on swap-search for extra exploration)

# Social-feature weights for client importance (ALL features except clinick_distance)
FEATURE_COEFS = {
    "client_age": 0.06917052,
    "density_area": 0.1621698,
    "park_distance": -0.00008931274,
    "vulnerable_group_density": 0.00299977,
    "social_infrastructure_rating": -0.00035131,
}

# 1=metro(best), 2=tram, 3=bus, 4=taxi(worst)
TRANSPORT_MULTIPLIER = {
    1: 1.78705976,
    2: 1.80623717,
    3: 1.83286559,
    4: 2.04023626,
                       }

# Weight clipping to keep optimization stable
W_CLIP = (0.20234528, 2.9232447)

# Capacity-aware warmup before TTT optimization (uses weights)
CAPACITY_WARM_ITERS = 12

# Swap-based local search (helps escape local minima for TTT)
SWAP_TRIES = 60
SWAP_REFINE_ITERS = 20
SWAP_RANDOM_SEED = 123

# LCS weights (task metric #3).
LCS_WEIGHTS = {
    "density_area": 0.5,                 # DA_j
    "park_distance": 0.5,                 # PD_j 
    "vulnerable_group_density": 0.5,     # VD_j
    "social_infrastructure_rating": 0.5,  # SI_j
}

# Visualization
MAX_POINTS_PLOT = 60000
PLOT_DPI = 160

REQUIRED_COLS = [
    "x", "y",
    "client_age",
    "clinick_distance",
    "density_area",
    "park_distance",
    "vulnerable_group_density",
    "social_infrastructure_rating",
]

def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between a(M,2) and b(N,2) -> (M,N)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def robust_zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust z-score: (x - median) / IQR."""
    x = x.astype(float)
    med = np.nanmedian(x)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = (q3 - q1) + eps
    return (x - med) / iqr

def build_client_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Client importance weight:
      base_w = exp(sum_k coef_k * robust_z(f_k))
      w = clip(base_w * transport_multiplier(clinick_distance), W_CLIP)
    """
    score = np.zeros(len(df), dtype=float)
    for col, coef in FEATURE_COEFS.items():
        z = robust_zscore(df[col].to_numpy())
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        score += float(coef) * z

    base_w = np.exp(score)

    cd = df["clinick_distance"].to_numpy()
    tm = np.array([TRANSPORT_MULTIPLIER.get(int(v), 1.0) for v in cd], dtype=float)

    w = base_w * tm
    w = np.clip(w, W_CLIP[0], W_CLIP[1])
    return w

def assign_with_capacity(points: np.ndarray, centers: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Greedy assignment with capacity:
      - higher w clients first
      - tie-break: farther from nearest clinic first
      - assign to nearest clinic with remaining capacity
    """
    d = pairwise_dist(points, centers)
    order = np.argsort(d, axis=1)

    d1 = d[np.arange(len(points)), order[:, 0]]
    # primary: w desc, secondary: d1 desc
    clients = np.lexsort((-d1, -w))

    remaining = np.full(len(centers), CAPACITY_L, dtype=np.int32)
    assign = np.full(len(points), -1, dtype=np.int32)

    for i in clients:
        for c in order[i]:
            if remaining[c] > 0:
                assign[i] = c
                remaining[c] -= 1
                break

        if assign[i] == -1:
            raise RuntimeError("Total capacity is insufficient for all clients.")

    return assign

def weighted_geometric_median(pts: np.ndarray, weights: np.ndarray, x0: np.ndarray,
                              steps: int = WEISZFELD_STEPS, eps: float = 1e-9) -> np.ndarray:
    x = x0.astype(float)
    for _ in range(steps):
        diff = pts - x
        r = np.sqrt(np.sum(diff * diff, axis=1)) + eps
        wr = weights / r
        x_new = np.sum(pts * wr[:, None], axis=0) / (np.sum(wr) + eps)
        if np.linalg.norm(x_new - x) < 1e-6:
            x = x_new
            break
        x = x_new
    return x

def update_centers(points: np.ndarray, assign: np.ndarray, centers: np.ndarray, w: np.ndarray) -> np.ndarray:
    new_centers = centers.copy()
    for j in range(len(centers)):
        idx = np.where(assign == j)[0]
        if len(idx) == 0:
            continue
        new_centers[j] = weighted_geometric_median(points[idx], w[idx], centers[j])
    return new_centers

def update_centers_ttt(points: np.ndarray, assign: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Move centers to minimize TTT directly:
      weight for point i: alpha / dist + 2*beta.
    """
    new_centers = centers.copy()
    for j in range(len(centers)):
        idx = np.where(assign == j)[0]
        if len(idx) == 0:
            continue
        pts_j = points[idx]
        diff = pts_j - centers[j]
        r = np.sqrt(np.sum(diff * diff, axis=1))
        r = np.where(r < 1e-8, 1e-8, r)
        weights = ALPHA / r + 2 * BETA
        new_centers[j] = np.sum(pts_j * weights[:, None], axis=0) / np.sum(weights)
    return new_centers

def capacity_warm_opt(points: np.ndarray, centers: np.ndarray, w: np.ndarray,
                      n_iters: int) -> tuple[np.ndarray, float]:
    """
    Capacity-aware warmup:
      - assign with capacity and weights
      - shift centers by weighted geometric median
    Returns centers after warmup and best TTT observed during the warm stage.
    """
    centers_warm = centers.copy()
    best_ttt = metric_TTT(points, centers_warm)

    for _ in range(n_iters):
        assign = assign_with_capacity(points, centers_warm, w)
        centers_warm = update_centers(points, assign, centers_warm, w)
        ttt = metric_TTT(points, centers_warm)
        if ttt < best_ttt:
            best_ttt = ttt

    return centers_warm, best_ttt

def metric_TTT(points: np.ndarray, centers: np.ndarray) -> float:
    """TTT = sum_i min_j (alpha*d + beta*d^2)."""
    d = pairwise_dist(points, centers)          # (M,N)
    best = np.min(d, axis=1)                   # (M,)
    return float(np.sum(ALPHA * best + BETA * (best ** 2)))

def metric_CO(assign: np.ndarray, n_centers: int) -> int:
    """CO = sum_j max(0, K_j - L)."""
    counts = np.bincount(assign, minlength=n_centers)
    overload = np.maximum(0, counts - CAPACITY_L)
    return int(np.sum(overload))

def metric_LCS(df: pd.DataFrame, assign: np.ndarray, n_centers: int) -> float:
    """
    LCS = (1/N) * sum_j (w1*DA_j + w2*PD_j + w3*VD_j + w4*SI_j),
    where we aggregate per clinic as mean over assigned clients.
    PD_j uses NEGATED mean park_distance so that "closer park" increases LCS.
    """
    da = df["density_area"].to_numpy(float)
    pdist = df["park_distance"].to_numpy(float)
    vd = df["vulnerable_group_density"].to_numpy(float)
    si = df["social_infrastructure_rating"].to_numpy(float)

    lcs_sum = 0.0
    for j in range(n_centers):
        idx = np.where(assign == j)[0]
        if len(idx) == 0:
            continue
        DA_j = float(np.mean(da[idx]))
        PD_j = float(np.mean(pdist[idx]))
        
        VD_j = float(np.mean(vd[idx]))
        SI_j = float(np.mean(si[idx]))

        lcs_sum += (
            LCS_WEIGHTS["density_area"] * DA_j +
            LCS_WEIGHTS["park_distance"] * PD_j +
            LCS_WEIGHTS["vulnerable_group_density"] * VD_j +
            LCS_WEIGHTS["social_infrastructure_rating"] * SI_j
        )

    return float(lcs_sum / n_centers)

def optimize_centers_for_ttt(points: np.ndarray, init_centers: np.ndarray,
                             n_iters: int = TTT_ITERS, log_every: int = 20) -> tuple[np.ndarray, float]:
    """
    Lloyd-like loop tailored to TTT:
      1) assign by nearest center (no capacity)
      2) shift centers with TTT weights (alpha / r + 2*beta)
    Keeps the best TTT seen.
    """
    centers = init_centers.copy()
    best_centers = centers.copy()
    best_ttt = metric_TTT(points, centers)

    for it in range(n_iters):
        d = pairwise_dist(points, centers)
        assign = np.argmin(d, axis=1)
        centers = update_centers_ttt(points, assign, centers)

        ttt = metric_TTT(points, centers)
        if ttt < best_ttt:
            best_ttt = ttt
            best_centers = centers.copy()

        if log_every and (it + 1) % log_every == 0:
            print(f"TTT iter {it+1:03d}/{n_iters} | TTT={ttt:,.6f} | best={best_ttt:,.6f}")

    return best_centers, best_ttt

def swap_local_search(points: np.ndarray, centers: np.ndarray, n_swaps: int,
                      refine_iters: int, rng_seed: int = SWAP_RANDOM_SEED) -> tuple[np.ndarray, float]:
    """
    Random-swap local search:
      - replace one center with random client point
      - run short TTT-only optimization
      - keep the candidate if it improves TTT
    Deterministic via rng_seed.
    """
    rng = np.random.default_rng(rng_seed)
    best_centers = centers.copy()
    best_ttt = metric_TTT(points, best_centers)

    for s in range(n_swaps):
        cand = best_centers.copy()
        j = int(rng.integers(0, len(cand)))
        idx = int(rng.integers(0, len(points)))
        cand[j] = points[idx]

        cand, cand_ttt = optimize_centers_for_ttt(points, cand, n_iters=refine_iters, log_every=0)
        if cand_ttt < best_ttt:
            best_centers = cand
            best_ttt = cand_ttt

    return best_centers, best_ttt

def visualize_and_save(df: pd.DataFrame, centers: np.ndarray, assign: np.ndarray) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    pts = df[["x", "y"]].to_numpy(float)

    if len(pts) > MAX_POINTS_PLOT:
        idx = np.random.choice(len(pts), size=MAX_POINTS_PLOT, replace=False)
        pts_v = pts[idx]
        assign_v = assign[idx]
    else:
        pts_v = pts
        assign_v = assign

    # 01: base map
    plt.figure(figsize=(9, 7))
    plt.scatter(pts_v[:, 0], pts_v[:, 1], s=3, alpha=0.35)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=90, linewidths=2)
    plt.title("Clients and clinic locations")
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, "01_map_clients_clinics.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # 02: assignment map
    plt.figure(figsize=(9, 7))
    plt.scatter(pts_v[:, 0], pts_v[:, 1], s=3, c=assign_v, alpha=0.40)
    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=90, linewidths=2)
    plt.title("Clients colored by assigned clinic")
    plt.xlabel("x"); plt.ylabel("y")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, "02_map_by_assignment.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # 03: clinic loads histogram
    counts = np.bincount(assign, minlength=len(centers))
    plt.figure(figsize=(9, 4))
    plt.hist(counts, bins=30)
    plt.title("Clinic load distribution (clients per clinic)")
    plt.xlabel("clients per clinic"); plt.ylabel("number of clinics")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, "03_clinic_load_hist.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

def main():
    df = pd.read_csv(DATA_CSV_PATH)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{DATA_CSV_PATH} is missing required columns: {missing}")

    points = df[["x", "y"]].to_numpy(float)

    if len(points) > N_CLINICS * CAPACITY_L:
        raise ValueError(f"Clients={len(points)} > total capacity={N_CLINICS*CAPACITY_L}. Increase N_CLINICS or CAPACITY_L.")

    # weights for capacity-aware assignment/plots
    w = build_client_weights(df)

    # build initial centers: warm start + several k-means seeds
    init_candidates = []
    for i in range(N_RESTARTS):
        seed = RANDOM_STATE + i
        kmeans = MiniBatchKMeans(
            n_clusters=N_CLINICS,
            random_state=seed,
            batch_size=KMEANS_BATCH_SIZE,
            n_init="auto",
        )
        kmeans.fit(points)
        init_candidates.append((f"kmeans_seed_{seed}", kmeans.cluster_centers_.astype(float)))
        print(f"Prepared k-means init seed={seed}")

    # run TTT optimization for each init, keep the best
    best = None
    for label, init_centers in init_candidates:
        print(f"\nStarting TTT optimization from {label}...")
        centers_start = init_centers
        warm_ttt = None
        if CAPACITY_WARM_ITERS > 0:
            centers_start, warm_ttt = capacity_warm_opt(points, centers_start, w, n_iters=CAPACITY_WARM_ITERS)
            print(f"  Warm stage ({CAPACITY_WARM_ITERS} iters) best TTT={warm_ttt:,.6f}")

        centers_base, ttt_base = optimize_centers_for_ttt(points, centers_start, n_iters=TTT_ITERS, log_every=TTT_LOG_EVERY)
        centers_swap, ttt_swap = swap_local_search(points, centers_base, n_swaps=SWAP_TRIES, refine_iters=SWAP_REFINE_ITERS, rng_seed=SWAP_RANDOM_SEED)

        print(f"Finished {label}: base TTT={ttt_base:,.6f} | after swaps TTT={ttt_swap:,.6f}")
        if best is None or ttt_swap < best[0]:
            best = (ttt_swap, centers_swap, label, warm_ttt, ttt_base)

    best_ttt, centers, best_label, best_warm_ttt, best_base_ttt = best
    print(f"\nBest centers from {best_label} with TTT={best_ttt:,.6f}")
    if best_warm_ttt is not None:
        print(f"Best warm-stage TTT={best_warm_ttt:,.6f}")
    print(f"Best pre-swap TTT={best_base_ttt:,.6f}")

    # capacity-aware assignment for reporting/plots
    assign_cap = assign_with_capacity(points, centers, w)
    co = metric_CO(assign_cap, len(centers))
    lcs = metric_LCS(df, assign_cap, len(centers))

    pd.DataFrame(centers, columns=["x", "y"]).to_csv(OUT_CLINICS_CSV, index=False, float_format="%.10f")
    visualize_and_save(df, centers, assign_cap)

    print("\nFinished.")
    print(f"TTT={best_ttt:,.6f} | CO={co} | LCS={lcs:,.6f}")
    print("Saved:", OUT_CLINICS_CSV)
    print("Saved plots to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
