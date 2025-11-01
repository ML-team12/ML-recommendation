import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from load_data import load_or_create_data

# 파일 경로 지정
CLEAN_YELP_PICKLE = "data/train/clean_yelp.pkl"
CLEAN_OPENTABLE_PICKLE = "data/test/clean_opentable.pkl"
OPENTABLE_PATH = "data/test/opentable.csv"

# Yelp 데이터 전처리 (학습용)
def preprocess_yelp(save_path=CLEAN_YELP_PICKLE):
    df_business, df_review, df_user = load_or_create_data()

    # 주요 컬럼 선택
    df_b = df_business[['business_id', 'name', 'city', 'state', 'postal_code', 'categories', 'stars']].copy()
    df_r = df_review[['user_id', 'business_id', 'stars', 'text']].copy()
    df_u = df_user[['user_id', 'review_count', 'average_stars']].copy()

    # 컬럼명 통일
    df_r.rename(columns={'stars': 'rating', 'text': 'review_text'}, inplace=True)
    df_b['categories'] = df_b['categories'].fillna('').astype(str).str.lower()
    df_r['review_text'] = df_r['review_text'].fillna('').astype(str).str.lower()

    # 데이터 병합
    df = df_r.merge(df_b, on='business_id', how='left')
    df = df.merge(df_u, on='user_id', how='left')

    # 결측치 처리
    df['categories'].replace('', 'unknown', inplace=True)
    df['review_text'].fillna('', inplace=True)

    # 평점 정규화
    scaler = MinMaxScaler()
    df['rating_norm'] = scaler.fit_transform(df[['rating']])

    # 컬럼 통일
    df_out = df[['user_id', 'business_id', 'rating', 'rating_norm', 'review_text', 'categories', 'city', 'name']]

    df_out.to_pickle(save_path)
    print(f"Saved Yelp clean data → {save_path} (shape={df_out.shape})")
    return df_out


# OpenTable 데이터 전처리 (테스트용)
def preprocess_opentable(opentable_path=OPENTABLE_PATH, save_path=CLEAN_OPENTABLE_PICKLE):
    if opentable_path.endswith('.csv'):
        df_ot = pd.read_csv(opentable_path)
    else:
        df_ot = pd.read_json(opentable_path, lines=True)

    # 컬럼명 소문자화
    df_ot.columns = [c.lower().strip() for c in df_ot.columns]

    # 컬럼명 표준화
    rename_map = {}
    for c in df_ot.columns:
        if 'uid' in c:
            rename_map[c] = 'user_id'
        elif 'item' in c:
            rename_map[c] = 'business_id'
        elif c in ['rating', 'overall', 'stars']:
            rename_map[c] = 'rating'
        elif c in ['food', 'service', 'ambience', 'value']:
            rename_map[c] = c
    df_ot.rename(columns=rename_map, inplace=True)

    # 컬럼 체크
    if 'user_id' not in df_ot.columns or 'business_id' not in df_ot.columns:
        raise ValueError("OpenTable 데이터에 user_id 또는 business_id 컬럼이 없습니다.")

    # 다중 기준 평균 rating
    criteria_cols = [c for c in ['food', 'service', 'ambience', 'value'] if c in df_ot.columns]
    if 'rating' not in df_ot.columns and len(criteria_cols) > 0:
        df_ot['rating'] = df_ot[criteria_cols].mean(axis=1)

    # 평점 정규화
    scaler = MinMaxScaler()
    df_ot['rating_norm'] = scaler.fit_transform(df_ot[['rating']])

    # pseudo_content → review_text
    def make_pseudo_text(row):
        tokens = []
        for c in criteria_cols:
            val = row.get(c, np.nan)
            if not pd.isna(val):
                tokens.append(f"{c}_{int(round(val))}")
        return " ".join(tokens)

    df_ot['review_text'] = df_ot.apply(make_pseudo_text, axis=1)
    df_ot['categories'] = 'unknown'  # OpenTable에는 카테고리 정보 없으므로 placeholder
    df_ot['city'] = 'unknown'
    df_ot['name'] = 'unknown'

    # 컬럼 순서 통일
    df_out = df_ot[['user_id', 'business_id', 'rating', 'rating_norm', 'review_text', 'categories', 'city', 'name']]

    df_out.to_pickle(save_path)
    print(f"Saved OpenTable clean data → {save_path} (shape={df_out.shape})")
    return df_out


if __name__ == "__main__":
    # Yelp 데이터
    df_yelp = preprocess_yelp()
    # OpenTable 데이터
    if os.path.exists(OPENTABLE_PATH):
        df_opentable = preprocess_opentable(OPENTABLE_PATH)
    else:
        print(f"OpenTable 파일을 찾을 수 없습니다: {OPENTABLE_PATH}")

    # 컬럼 전체 출력
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_yelp.head())
    print(df_opentable.head())
