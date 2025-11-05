import os, time, pickle, math, gc, warnings
from typing import Dict, Tuple, List
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# Optional: torch autoencoder (very small)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception as e:
    print("[WARN] PyTorch not available, autoencoder will be skipped:", e)
    TORCH_OK = False

# Optional: FAISS (for very fast KNN on user profiles)
try:
    import faiss
    FAISS_OK = True
except Exception as e:
    print("[WARN] FAISS not available, will fall back to brute-force cosine:", e)
    FAISS_OK = False

# --------------------------
# Paths
# --------------------------
YELP_CLEAN = "data/train/clean_yelp.pkl"
OT_CLEAN   = "data/test/clean_opentable.pkl"
TFIDF_DIR  = "artifacts_tfidf"
FAISS_DIR  = "faiss_results"
RES_DIR    = "results"
os.makedirs(RES_DIR, exist_ok=True)

# --------------------------
# Utils
# --------------------------
def log(msg:str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_first_existing(paths: List[str]):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of the expected paths exist:\n" + "\n".join(paths))

def l2_normalize_csr(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """L2 normalize rows of CSR matrix in-place-ish."""
    # Use sklearn normalize for safety
    return normalize(X, norm="l2", copy=False)

# --------------------------
# 0) Sanity checks for teammate outputs
# --------------------------
def quick_validate_artifacts() -> str:
    notes = []
    # TF-IDF vectorizer
    vec_path = os.path.join(TFIDF_DIR, "vectorizer_tfidf.pkl")
    if os.path.exists(vec_path):
        notes.append("OK: artifacts_tfidf/vectorizer_tfidf.pkl found")
    else:
        notes.append("MISSING: artifacts_tfidf/vectorizer_tfidf.pkl")

    # TF-IDF matrix (name mismatch tolerant: underscore OR dot)
    x_paths = [
        os.path.join(TFIDF_DIR, "X_items_yelp_tfidf.npz"),
        os.path.join(TFIDF_DIR, "X_items_yelp.tfidf.npz"),
    ]
    try:
        x_path = load_first_existing(x_paths)
        notes.append(f"OK: TF-IDF items matrix found → {x_path}")
    except FileNotFoundError as e:
        notes.append("MISSING: X_items_yelp_(tfidf).npz not found (underscore or dot variant)")
        x_path = None

    # FAISS KNN outputs (optional)
    for fname in ["user_neighbors.npy", "user_similarities.npy", "user_mapping.pkl"]:
        p = os.path.join(FAISS_DIR, fname)
        notes.append(("OK:" if os.path.exists(p) else "MISSING:") + f" {FAISS_DIR}/{fname}")

    return "\n".join(notes), vec_path, x_path

# --------------------------
# 1) Load data
# --------------------------
def load_clean_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(YELP_CLEAN):
        raise FileNotFoundError(f"{YELP_CLEAN} not found. Run preprocessing.py first.")
    if not os.path.exists(OT_CLEAN):
        raise FileNotFoundError(f"{OT_CLEAN} not found. Run preprocessing.py for OpenTable.")

    df_yelp = pd.read_pickle(YELP_CLEAN)
    df_ot   = pd.read_pickle(OT_CLEAN)

    # Keep only relevant columns
    cols = ['user_id','business_id','rating','rating_norm','review_text','categories','city','name']
    df_yelp = df_yelp[cols].dropna(subset=['user_id','business_id'])
    df_ot   = df_ot[cols].dropna(subset=['user_id','business_id'])
    return df_yelp, df_ot

# --------------------------
# 2) Build item maps & TF-IDF matrices
# --------------------------
def load_tfidf_artifacts(vec_path:str, x_items_path:str):
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    X_items_yelp = sparse.load_npz(x_items_path).tocsr()

    # Build yelp item_id -> row map (aligned with artifacts build order)
    biz_ids_path = os.path.join(TFIDF_DIR, "biz_ids.pkl")
    if not os.path.exists(biz_ids_path):
        raise FileNotFoundError("artifacts_tfidf/biz_ids.pkl missing (created by tfidf_build.py)")
    with open(biz_ids_path, "rb") as f:
        yelp_biz_ids = np.array(pickle.load(f)).astype(str)
    if X_items_yelp.shape[0] != len(yelp_biz_ids):
        raise ValueError("TF-IDF matrix row count and biz_ids length mismatch")
    yelp_biz2row: Dict[str,int] = {bid:i for i,bid in enumerate(yelp_biz_ids)}
    return vectorizer, X_items_yelp, yelp_biz2row

def build_ot_item_matrix(df_ot_items: pd.DataFrame, vectorizer) -> Tuple[sparse.csr_matrix, np.ndarray]:
    # Compose text same as training build: name + categories + review_text
    texts = (df_ot_items["name"].astype(str) + " " +
             df_ot_items["categories"].astype(str) + " " +
             df_ot_items["review_text"].astype(str)).values
    X_ot = vectorizer.transform(texts)
    X_ot = l2_normalize_csr(X_ot.tocsr())
    item_ids = df_ot_items["business_id"].astype(str).values
    return X_ot, item_ids

# --------------------------
# 3) Build USER content profiles (weighted sum of item TF-IDF)
# --------------------------
def user_profiles_from_interactions(df: pd.DataFrame,
                                    item_row_map: Dict[str,int],
                                    X_items: sparse.csr_matrix,
                                    weight_col: str = "rating_norm") -> Tuple[sparse.csr_matrix, np.ndarray]:
    # aggregate per user: sum( weight * item_vector )
    users = df["user_id"].astype(str).values
    items = df["business_id"].astype(str).values
    weights = df[weight_col].astype(np.float32).values

    # Map item ids to rows; skip missing
    item_rows = np.array([item_row_map.get(i, -1) for i in items], dtype=np.int64)
    mask = item_rows >= 0
    users = users[mask]; item_rows = item_rows[mask]; weights = weights[mask]
    if len(users) == 0:
        raise ValueError("No interactions matched the item TF-IDF matrix.")

    # Build CSR of shape (n_unique_users, n_items) with weights, then multiply by X_items
    user_cats = pd.Categorical(users)
    u_idx = user_cats.codes.astype(np.int64)
    n_users = len(user_cats.categories)
    n_items = X_items.shape[0]

    # Sparse "user x item" weight matrix
    W = sparse.csr_matrix((weights, (u_idx, item_rows)), shape=(n_users, n_items), dtype=np.float32)
    # User profile = W * X_items  (shape: n_users x vocab)
    U = W @ X_items
    U = l2_normalize_csr(U.tocsr())
    user_ids = user_cats.categories.astype(str).values
    return U, user_ids

# --------------------------
# 4) Tiny AutoEncoder over user profiles
# --------------------------
class TinyAE(nn.Module):
    def __init__(self, dim_in: int, h1=512, h2=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, dim_in), nn.ReLU(),
        )
    def forward(self, x):  # x: (B, D)
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z

def train_ae_sparse(U_csr: sparse.csr_matrix, epochs=3, batch_size=2048, lr=1e-3, device="cpu"):
    dim = U_csr.shape[1]
    model = TinyAE(dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    # Make a dense minibatch iterator but only for non-empty rows to keep it light
    # We'll sample rows with some probability to reduce time
    row_nnz = np.diff(U_csr.indptr)
    nonempty_idx = np.where(row_nnz > 0)[0]
    if len(nonempty_idx) == 0:
        raise ValueError("All user profiles are empty!")
    # if very large, subsample to ~150k rows max for speed
    max_rows = 150_000
    if len(nonempty_idx) > max_rows:
        rng = np.random.default_rng(42)
        nonempty_idx = rng.choice(nonempty_idx, size=max_rows, replace=False)

    U_sub = U_csr[nonempty_idx].toarray().astype(np.float32)

    ds = torch.utils.data.TensorDataset(torch.from_numpy(U_sub))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for ep in range(1, epochs+1):
        total = 0.0; n = 0
        for (xb,) in dl:
            xb = xb.to(device)
            yb, _ = model(xb)
            loss = crit(yb, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        log(f"[AE] epoch {ep}/{epochs}  loss={total/max(1,n):.6f}")
    return model

def ae_encode_decode(model: TinyAE, U_csr: sparse.csr_matrix, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        Z_list, Y_list = [], []
        for start in range(0, U_csr.shape[0], 4096):
            Sl = U_csr[start:start+4096].toarray().astype(np.float32)
            X = torch.from_numpy(Sl).to(device)
            Y, Z = model(X)
            Y_list.append(Y.cpu().numpy())
            Z_list.append(Z.cpu().numpy())
        Y_full = np.vstack(Y_list)
        Z_full = np.vstack(Z_list)
    return Y_full, Z_full

# --------------------------
# 5) FAISS KNN on user profiles (cosine via normalized IP)
# --------------------------
def knn_on_profiles(U_csr: sparse.csr_matrix, k=20):
    U = U_csr.astype(np.float32).toarray()  # small enough (n_users_test x dim)
    # L2-normalize to use inner product as cosine
    norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-8
    U = U / norms

    if FAISS_OK:
        d = U.shape[1]; nlist = min(2048, max(8, U.shape[0]//200))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(U); index.add(U); index.nprobe = min(16, nlist)
        D, I = index.search(U, k+1)   # include self
    else:
        # brute-force cosine
        S = U @ U.T
        # keep topk+1 (self)
        I = np.argpartition(-S, range(k+1), axis=1)[:, :k+1]
        D = np.take_along_axis(S, I, axis=1)

    # Remove self (index equal to row id)
    row_idx = np.arange(U.shape[0])[:, None]
    # If self not at [*,0], we still keep top-k excluding any exact self match
    keep = I != row_idx
    I_clean = []
    D_clean = []
    for r in range(I.shape[0]):
        mask = keep[r]
        I_r = I[r][mask][:k]
        D_r = D[r][mask][:k]
        I_clean.append(I_r)
        D_clean.append(D_r)
    return np.array(D_clean, dtype=np.float32), np.array(I_clean, dtype=np.int32)


# ----------------------------------------------------------
# Helper: 사용자 벡터(U)와 아이템 벡터(X)를 곱해 점수 행렬 계산
# ----------------------------------------------------------
def score_users_against_items(U_csr: sparse.csr_matrix,
                              X_items: sparse.csr_matrix) -> np.ndarray:
    """
    Compute dense score matrix of user-item similarities:
      Score = U · X_items^T
    For large matrices, it splits into blocks to save memory.
    """
    n_users = U_csr.shape[0]
    n_items = X_items.shape[0]
    out = np.zeros((n_users, n_items), dtype=np.float32)

    block = 2048
    for s in range(0, n_users, block):
        # (block x vocab) * (items x vocab)^T
        Sl = U_csr[s:s + block]
        Sc = Sl @ X_items.T
        out[s:s + block] = Sc.toarray().astype(np.float32)
    return out


# --------------------------
# 6) Scoring & recommend
# --------------------------
def recommend_hybrid(U_raw: sparse.csr_matrix,
                     U_ae_recon: np.ndarray,
                     U_knn_D: np.ndarray, U_knn_I: np.ndarray,
                     X_items: sparse.csr_matrix,
                     topk: int = 10, alpha: float = 0.5, beta: float = 0.2, gamma: float = 0.3
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hybrid scoring:
      S = alpha * (AE recon · items) + beta * (KNN-avg profile · items) + gamma * (raw TF-IDF · items)
    Notes:
      - Ensure KNN-avg rows are 2-D CSR before vstack (avoid 'blocks must be 2-D').
    """
    n_users = U_raw.shape[0]
    # --- a) AE reconstructed profile vs items
    Uae = sparse.csr_matrix(U_ae_recon.astype(np.float32, copy=False))
    S_ae = score_users_against_items(Uae, X_items)  # (n_users x n_items)

    # --- b) Raw TF-IDF profile vs items
    S_tf = score_users_against_items(U_raw, X_items)

    # --- c) KNN-smoothed user profile (mean of neighbor raw profiles)
    rows = []
    for u in range(n_users):
        nbrs = U_knn_I[u]
        # numpy indexing의 안전성: np.ndarray일 때는 .size 사용
        if getattr(nbrs, "size", len(nbrs)) == 0:
            # slice는 1xV CSR로 유지되어 vstack 가능
            rows.append(U_raw[u])
            continue

        # U_raw[nbrs]는 sparse matrix, mean(axis=0)는 numpy matrix/ndarray가 됨
        Ubar = U_raw[nbrs].mean(axis=0)

        # 1) 2-D ndarray 보장
        if hasattr(Ubar, "A"):            # numpy.matrix인 경우
            Ubar = Ubar.A                 # ndarray
        Ubar = np.asarray(Ubar)
        if Ubar.ndim == 1:                # (V,) -> (1, V)
            Ubar = Ubar.reshape(1, -1)

        # 2) CSR 캐스팅 (float32 권장)
        Ubar_csr = sparse.csr_matrix(Ubar.astype(np.float32, copy=False))
        rows.append(Ubar_csr)

    U_knn = sparse.vstack(rows, format="csr")
    S_knn = score_users_against_items(U_knn, X_items)

    # --- hybrid combine
    S = alpha * S_ae + beta * S_knn + gamma * S_tf

    # --- top-k
    top_idx = np.argpartition(-S, range(topk), axis=1)[:, :topk]
    part = np.take_along_axis(S, top_idx, axis=1)
    order = np.argsort(-part, axis=1)
    top_sorted = np.take_along_axis(top_idx, order, axis=1)
    top_scores = np.take_along_axis(S, top_sorted, axis=1)
    return top_sorted, top_scores


# --------------------------
# Main
# --------------------------
def main():
    t0 = time.time()
    notes, vec_path, x_items_path = quick_validate_artifacts()
    with open(os.path.join(RES_DIR, "diagnostics.txt"), "w", encoding="utf-8") as f:
        f.write(notes + "\n")

    log("Load clean frames")
    df_yelp, df_ot = load_clean_frames()

    log("Load TF-IDF artifacts")
    vectorizer, X_items_yelp, yelp_biz2row = load_tfidf_artifacts(vec_path, x_items_path)

    # Build OpenTable unique item frame
    log("Build OpenTable item TF-IDF matrix")
    df_ot_items = df_ot.drop_duplicates("business_id")[["business_id","name","categories","review_text"]].copy()
    X_items_ot, ot_item_ids = build_ot_item_matrix(df_ot_items, vectorizer)

    # Yelp user content profiles (train for AE)
    log("Build Yelp user profiles (TF-IDF)")
    Uy, yelp_user_ids = user_profiles_from_interactions(df_yelp, yelp_biz2row, X_items_yelp, weight_col="rating_norm")

    # Train tiny AE
    if TORCH_OK:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Train AE on Yelp profiles (device={device})")
        ae = train_ae_sparse(Uy, epochs=3, batch_size=2048, lr=1e-3, device=device)
    else:
        ae = None
        log("Skip AE (PyTorch not available)")

    # Build OpenTable user profiles (test-time)
    log("Build OpenTable user profiles (TF-IDF via OT items)")
    # For OT, items must be looked up in X_items_ot; build a map
    ot_biz2row = {bid:i for i,bid in enumerate(ot_item_ids)}
    Uot, ot_user_ids = user_profiles_from_interactions(df_ot, ot_biz2row, X_items_ot, weight_col="rating_norm")

    # AE reconstruct OpenTable profiles (use the trained AE; if not, use raw as fallback)
    if ae is not None:
        log("AE encode/decode OpenTable user profiles")
        Uot_recon, _ = ae_encode_decode(ae, Uot, device=("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        log("AE missing: copy raw profiles as recon")
        Uot_recon = Uot.toarray().astype(np.float32)

    # KNN on OpenTable user profiles
    log("Compute KNN on OpenTable user profiles")
    D_knn, I_knn = knn_on_profiles(Uot, k=20)

    # Recommend against OpenTable items (content space)
    log("Score & recommend (hybrid of AE + KNN + raw TF-IDF)")
    top_idx, top_scores = recommend_hybrid(Uot, Uot_recon, D_knn, I_knn, X_items_ot, topk=10,
                                           alpha=0.5, beta=0.2, gamma=0.3)

    # Save results
    log("Save recommendations")
    # map item row indices to business_id
    top_biz = np.array(ot_item_ids, dtype=object)[top_idx]
    out_rows = []
    for u, uid in enumerate(ot_user_ids):
        for rank in range(top_biz.shape[1]):
            out_rows.append((uid, rank+1, top_biz[u, rank], float(top_scores[u, rank])))
    df_out = pd.DataFrame(out_rows, columns=["user_id","rank","business_id","score"])
    out_path = os.path.join(RES_DIR, "hybrid_recs_opentable.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    log(f"✅ Done. Saved → {out_path}  (elapsed {time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
