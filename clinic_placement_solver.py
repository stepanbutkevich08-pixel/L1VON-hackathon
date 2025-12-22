import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

DATA_CSV_PATH = "Data.csv"
OUT_CLINICS_CSV = "clinics.csv"
PLOTS_DIR = "plots"

# Task constraints
N_CLINICS = 250
CAPACITY_L = 1000

# Task TTT parameters
ALPHA = 0.05
BETA = 0.02

# Optimization
N_ITERS = 10
WEISZFELD_STEPS = 15
RANDOM_STATE = 42
KMEANS_BATCH_SIZE = 8192

# Social-feature weights for client importance (ALL features except clinick_distance)
FEATURE_COEFS = {
    "client_age": 0.20032162404146892,
    "density_area": 0.20991077818689963,
    "park_distance": 0.27823183679795566,
    "vulnerable_group_density": 0.03099478848113446,
    "social_infrastructure_rating": -0.3823305394463611,
}

# 1=metro(best), 2=tram, 3=bus, 4=taxi(worst)
TRANSPORT_MULTIPLIER = {
    1: 0.8391553042008542,
    2: 1.1832826541689683,
    3: 1.450205661235429,
    4: 1.9030399519634689
}

# Weight clipping to keep optimization stable
W_CLIP = (0.25, 6.0)

# LCS weights (task metric #3).
LCS_WEIGHTS = {
    "density_area": 1.0,                  # DA_j
    "park_distance": 1.0,                 # PD_j will be NEGATED inside LCS to make "closer park" better
    "vulnerable_group_density": 1.0,      # VD_j
    "social_infrastructure_rating": 1.0,  # SI_j
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
        PD_j = float(-np.mean(pdist[idx]))
        VD_j = float(np.mean(vd[idx]))
        SI_j = float(np.mean(si[idx]))

        lcs_sum += (
            LCS_WEIGHTS["density_area"] * DA_j +
            LCS_WEIGHTS["park_distance"] * PD_j +
            LCS_WEIGHTS["vulnerable_group_density"] * VD_j +
            LCS_WEIGHTS["social_infrastructure_rating"] * SI_j
        )

    return float(lcs_sum / n_centers)

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
        raise ValueError(f"Data.csv is missing required columns: {missing}")

    points = df[["x", "y"]].to_numpy(float)

    if len(points) > N_CLINICS * CAPACITY_L:
        raise ValueError(f"Clients={len(points)} > total capacity={N_CLINICS*CAPACITY_L}. Increase N_CLINICS or CAPACITY_L.")

    w = build_client_weights(df)

    # init centers
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLINICS,
        random_state=RANDOM_STATE,
        batch_size=KMEANS_BATCH_SIZE,
        n_init="auto",
    )
    kmeans.fit(points)
    centers = kmeans.cluster_centers_.astype(float)

    best_centers = centers.copy()
    best_ttt = float("inf")
    best_assign = None

    for it in range(N_ITERS):
        assign = assign_with_capacity(points, centers, w)
        centers = update_centers(points, assign, centers, w)

        TTT = metric_TTT(points, centers)
        CO = metric_CO(assign, len(centers))
        LCS = metric_LCS(df, assign, len(centers))

        print(f"ITER {it+1:02d}/{N_ITERS} | TTT={TTT:,.3f} | CO={CO} | LCS={LCS:,.6f}")

        if TTT < best_ttt:
            best_ttt = TTT
            best_centers = centers.copy()
            best_assign = assign.copy()

    pd.DataFrame(best_centers, columns=["x", "y"]).to_csv(OUT_CLINICS_CSV, index=False)

    if best_assign is None:
        best_assign = assign_with_capacity(points, best_centers, w)
    visualize_and_save(df, best_centers, best_assign)

    print("\nSaved:", OUT_CLINICS_CSV)
    print("Saved plots to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
