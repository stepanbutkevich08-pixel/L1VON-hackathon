import numpy as np
import pandas as pd

WEISZFELD_STEPS = 15
N_CLINICS = 250
CAPACITY_L = 1000
ALPHA = 0.05
BETA = 0.02

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


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distances between a(M,2) and b(N,2) -> (M,N)."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

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

def build_client_weights_parametric(df, feature_coefs, transport_multiplier, w_clip=(0.25, 6.0)):
    # robust z-score
    def robust_zscore(x, eps=1e-9):
        x = x.astype(float)
        med = np.nanmedian(x)
        q1 = np.nanpercentile(x, 25)
        q3 = np.nanpercentile(x, 75)
        iqr = (q3 - q1) + eps
        return (x - med) / iqr

    score = np.zeros(len(df), dtype=float)
    for col, coef in feature_coefs.items():
        z = robust_zscore(df[col].to_numpy())
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        score += float(coef) * z

    base_w = np.exp(score)
    cd = df["clinick_distance"].to_numpy()
    tm = np.array([transport_multiplier.get(int(v), 1.0) for v in cd], dtype=float)

    w = base_w * tm
    w = np.clip(w, w_clip[0], w_clip[1])
    return w

def run_solver_once(df, points, init_centers, feature_coefs, transport_multiplier,
                    n_iters=10, weiszfeld_steps=15, capacity=1000,
                    alpha=1.0, beta=0.0, w_clip=(0.25, 6.0)):
    # 1) веса (влияют на assignment + сдвиг центров)
    w = build_client_weights_parametric(df, feature_coefs, transport_multiplier, w_clip=w_clip)

    centers = init_centers.copy()
    best_ttt = float("inf")
    best_centers = centers.copy()

    # 2) итерации оптимизации
    for _ in range(n_iters):
        assign = assign_with_capacity(points, centers, w)  # использует capacity
        centers = update_centers(points, assign, centers, w)  # использует weiszfeld_steps

        # 3) считаем TTT по формуле задания (без весов!)
        ttt = metric_TTT(points, centers)  # использует alpha,beta
        if ttt < best_ttt:
            best_ttt = ttt
            best_centers = centers.copy()

    return best_ttt, best_centers

def metric_TTT(points: np.ndarray, centers: np.ndarray) -> float:
    """TTT = sum_i min_j (alpha*d + beta*d^2)."""
    d = pairwise_dist(points, centers)
    best = np.min(d, axis=1)
    return float(np.sum(ALPHA * best + BETA * (best ** 2)))

def sample_feature_coefs(rng):
    # Диапазоны подобраны так, чтобы веса не становились сумасшедшими
    # Можно расширять после первых прогонов.
    return {
        "client_age": rng.uniform(0.0, 1.0),
        "density_area": rng.uniform(0.0, 1.0),
        "park_distance": rng.uniform(0.0, 1.0),
        "vulnerable_group_density": rng.uniform(0.0, 1.0),
        # отрицательный коэффициент (чем выше рейтинг, тем меньше "need")
        "social_infrastructure_rating": rng.uniform(0.0, 1.0),
    }

def sample_transport_multiplier(rng):
    # Монотонно возрастающее: m1<=m2<=m3<=m4
    # Держим разумный диапазон
    m1 = rng.uniform(0.5, 1.0)
    m2 = rng.uniform(m1, 1.3)
    m3 = rng.uniform(m2, 1.7)
    m4 = rng.uniform(m3, 2.2)
    return {1: float(m1), 2: float(m2), 3: float(m3), 4: float(m4)}

def tune_parameters(data_csv="clinic-data.csv",
                    n_trials=200,
                    seed=42,
                    n_iters=10,
                    capacity=1000,
                    w_clip=(0.25, 6.0),
                    save_best_json="best_params.json"):
    import json
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    df = pd.read_csv(data_csv)
    points = df[["x", "y"]].to_numpy(float)

    # фиксируем одинаковую инициализацию центров, чтобы честно сравнивать параметры
    kmeans = MiniBatchKMeans(n_clusters=250, random_state=seed, batch_size=8192, n_init="auto")
    kmeans.fit(points)
    init_centers = kmeans.cluster_centers_.astype(float)

    best = {"ttt": float("inf"), "feature_coefs": None, "transport_multiplier": None}

    for t in range(1, n_trials + 1):
        fc = sample_feature_coefs(rng)
        tm = sample_transport_multiplier(rng)

        ttt, _ = run_solver_once(
            df=df,
            points=points,
            init_centers=init_centers,
            feature_coefs=fc,
            transport_multiplier=tm,
            n_iters=n_iters,
            capacity=capacity,
            w_clip=w_clip
        )

        if ttt < best["ttt"]:
            best = {"ttt": float(ttt), "feature_coefs": fc, "transport_multiplier": tm}
            print(f"[BEST] trial={t}/{n_trials}  TTT={ttt:,.3f}  fc={fc}  tm={tm}")
        elif t % 20 == 0:
            print(f"[..]   trial={t}/{n_trials}  current_TTT={ttt:,.3f}  best_TTT={best['ttt']:,.3f}")

    with open(save_best_json, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("Best TTT:", best["ttt"])
    print("Best FEATURE_COEFS:", best["feature_coefs"])
    print("Best TRANSPORT_MULTIPLIER:", best["transport_multiplier"])
    print("Saved:", save_best_json)
    return best

best = tune_parameters(n_trials=300, n_iters=10)
print(best)
