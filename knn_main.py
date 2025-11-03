"""
KNN 기반 음식점 추천 시스템 (로컬 환경용 통합 버전)
----------------------------------------------------
📦 구성:
 1️⃣ Yelp JSON → pickle 변환 (load_data)
 2️⃣ Yelp 전처리 (clean_yelp.pkl 생성)
 3️⃣ User-based KNN (FAISS 근사탐색)
 4️⃣ AutoEncoder 입력 희소행렬 (ae_inputs_sparse.npz 생성)

⚙️ 실행 환경:
 - Python 3.9+
 - pip install -r requirements.txt
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix, vstack, save_npz
import faiss
import gc

# ----------------------------------------------------------
# 경로 설정
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "train")
RESULT_DIR = os.path.join(BASE_DIR, "faiss_results")
os.makedirs(RESULT_DIR, exist_ok=True)

BUSINESS_JSON = os.path.join(DATA_DIR, "yelp_academic_dataset_business.json")
REVIEW_JSON = os.path.join(DATA_DIR, "yelp_academic_dataset_review.json")
BUSINESS_PKL = os.path.join(DATA_DIR, "business.pkl")
REVIEW_PKL = os.path.join(DATA_DIR, "review.pkl")
CLEAN_YELP_PKL = os.path.join(DATA_DIR, "clean_yelp.pkl")
AE_INPUTS_NPZ = os.path.join(BASE_DIR, "ae_inputs_sparse.npz")

# ----------------------------------------------------------
# 1️⃣ Yelp JSON → pickle 변환
# ----------------------------------------------------------
def load_or_create_data():
    """Yelp의 business/review JSON을 불러오거나 pickle로 저장"""
    if os.path.exists(BUSINESS_PKL):
        with open(BUSINESS_PKL, "rb") as f:
            df_business = pickle.load(f)
        print("✅ Loaded business.pkl")
    else:
        print("📂 Reading business JSON...")
        df_business = pd.read_json(BUSINESS_JSON, lines=True)
        with open(BUSINESS_PKL, "wb") as f:
            pickle.dump(df_business, f)
        print("💾 Saved business.pkl")

    if os.path.exists(REVIEW_PKL):
        with open(REVIEW_PKL, "rb") as f:
            df_review = pickle.load(f)
        print("✅ Loaded review.pkl")
    else:
        print("📂 Reading review JSON (chunked)...")
        chunks = [chunk for chunk in pd.read_json(REVIEW_JSON, lines=True, chunksize=10000)]
        df_review = pd.concat(chunks, ignore_index=True)
        with open(REVIEW_PKL, "wb") as f:
            pickle.dump(df_review, f)
        print("💾 Saved review.pkl")

    return df_business, df_review


# ----------------------------------------------------------
# 2️⃣ Yelp 전처리
# ----------------------------------------------------------
def preprocess_yelp(save_path=CLEAN_YELP_PKL):
    df_business, df_review = load_or_create_data()
    print(f"✅ Business: {df_business.shape}, Review: {df_review.shape}")

    df_b = df_business[['business_id', 'categories', 'city', 'stars']].copy()
    df_r = df_review[['user_id', 'business_id', 'stars', 'text']].copy()
    df_r.rename(columns={'stars': 'rating', 'text': 'review_text'}, inplace=True)

    used_biz = df_r['business_id'].unique()
    df_b_small = df_b[df_b['business_id'].isin(used_biz)]

    biz_to_cat = dict(zip(df_b_small['business_id'], df_b_small['categories']))
    biz_to_city = dict(zip(df_b_small['business_id'], df_b_small['city']))
    biz_to_star = dict(zip(df_b_small['business_id'], df_b_small['stars']))

    df_r['categories'] = df_r['business_id'].map(biz_to_cat).fillna('unknown').astype(str).str.lower()
    df_r['city'] = df_r['business_id'].map(biz_to_city).fillna('unknown').astype(str).str.lower()
    df_r['biz_star'] = df_r['business_id'].map(biz_to_star)
    df_r['review_text'] = df_r['review_text'].fillna('').astype(str).str.lower()

    scaler = MinMaxScaler()
    df_r['rating_norm'] = scaler.fit_transform(df_r[['rating']])

    df_out = df_r[['user_id', 'business_id', 'rating', 'rating_norm',
                   'review_text', 'categories', 'city', 'biz_star']]
    df_out.to_pickle(save_path)
    print(f"💾 Saved clean_yelp.pkl → {save_path} ({df_out.shape})")

    del df_business, df_review, df_b, df_r, df_b_small
    gc.collect()
    return df_out


# ----------------------------------------------------------
# 3️⃣ User-based KNN (FAISS)
# ----------------------------------------------------------
def run_knn():
    t0 = time.time()
    df = pd.read_pickle(CLEAN_YELP_PKL)
    print(f"✅ Loaded clean_yelp.pkl: {df.shape}")

    ratings = df[['user_id', 'business_id', 'rating']].dropna()
    user_cat = ratings['user_id'].astype('category')
    biz_cat = ratings['business_id'].astype('category')
    user_idx = user_cat.cat.codes.values
    biz_idx = biz_cat.cat.codes.values

    n_users, n_items = len(user_cat.cat.categories), len(biz_cat.cat.categories)
    print(f"👤 {n_users:,} users | 🍽️ {n_items:,} items")

    user_item_sparse = csr_matrix((ratings['rating'].astype(np.float32), (user_idx, biz_idx)),
                                  shape=(n_users, n_items))
    print(f"🧮 Sparse Matrix: {user_item_sparse.shape}  Nonzero={user_item_sparse.nnz:,}")

    svd = TruncatedSVD(n_components=64, random_state=42)
    user_vectors = svd.fit_transform(user_item_sparse).astype('float32')

    d, nlist, nprobe = user_vectors.shape[1], 4000, 8
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    print("🔹 Training FAISS index...")
    index.train(user_vectors)
    index.add(user_vectors)
    index.nprobe = nprobe

    if faiss.get_num_gpus() > 0:
        print("⚡ GPU detected → moving index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index.nprobe = nprobe

    K = 6
    D, I = index.search(user_vectors, k=K)

    np.save(os.path.join(RESULT_DIR, "user_neighbors.npy"), I)
    np.save(os.path.join(RESULT_DIR, "user_similarities.npy"), D)

    print(f"💾 Saved results to {RESULT_DIR}")
    print(f"⏱ Total KNN Time: {time.time()-t0:.2f}s")
    return I, D, df, user_item_sparse


# ----------------------------------------------------------
# 4️⃣ AutoEncoder 입력 희소행렬 생성
# ----------------------------------------------------------
def create_autoencoder_input(neighbors, scores, df, user_item, save_path=AE_INPUTS_NPZ):
    t_total = time.time()
    user_cat = df['user_id'].astype('category')
    biz_cat = df['business_id'].astype('category')
    n_users, n_items = len(user_cat.cat.categories), len(biz_cat.cat.categories)

    batch_size = 10000
    num_batches = (n_users + batch_size - 1) // batch_size
    batch_outputs = []

    for b in range(num_batches):
        start, end = b * batch_size, min((b + 1) * batch_size, n_users)
        rows, cols, data = [], [], []
        print(f"▶ Batch {b+1}/{num_batches} ({start:,}~{end:,})")

        for local_i, u in enumerate(range(start, end)):
            neighbor_ids = neighbors[u, 1:]
            sim_scores = scores[u, 1:]
            weights = sim_scores / np.sum(sim_scores) if np.sum(sim_scores) > 0 else np.ones_like(sim_scores)/len(sim_scores)
            weighted = user_item[neighbor_ids].multiply(weights[:, np.newaxis])
            mean_vec = np.array(weighted.sum(axis=0))
            nz_cols = np.nonzero(mean_vec)[1]
            nz_vals = mean_vec[0, nz_cols]
            rows.extend([local_i]*len(nz_cols))
            cols.extend(nz_cols)
            data.extend(nz_vals)

        batch_sparse = csr_matrix((data, (rows, cols)), shape=(end-start, n_items))
        batch_outputs.append(batch_sparse)

    ae_inputs_sparse = vstack(batch_outputs).tocsr()
    save_npz(save_path, ae_inputs_sparse)
    print(f"✅ Saved AutoEncoder input → {save_path}")
    print(f"Total shape: {ae_inputs_sparse.shape}  ⏱{time.time()-t_total:.2f}s")


# ----------------------------------------------------------
# 실행 메인
# ----------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 KNN 파이프라인 시작 (로컬 환경)\n")
    df_clean = preprocess_yelp()
    neighbors, scores, df, user_item_sparse = run_knn()
    create_autoencoder_input(neighbors, scores, df, user_item_sparse)
    print("\n🏁 전체 파이프라인 완료!")
