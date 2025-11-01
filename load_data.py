import pandas as pd
import pickle
import os

# pickle 파일 경로 지정
business_pickle = "data/train/business.pkl"
review_pickle = "data/train/review.pkl"
user_pickle = "data/train/user.pkl"

def load_or_create_data():
    # Business data
    if os.path.exists(business_pickle):
        print("load business.pkl")
        with open(business_pickle, "rb") as f:
            df_business = pickle.load(f)
    else:
        print("load business JSON")
        df_business = pd.read_json("data/train/yelp_academic_dataset_business.json", lines=True)
        with open(business_pickle, "wb") as f:
            pickle.dump(df_business, f)
        print("save business.pkl")

    # Review data
    if os.path.exists(review_pickle):
        print("load review.pkl")
        with open(review_pickle, "rb") as f:
            df_review = pickle.load(f)
    else:
        print("load review JSON (chunksize=10000)")
        list_review = []
        for chunk in pd.read_json("data/train/yelp_academic_dataset_review.json", lines=True, chunksize=10000):
            list_review.append(chunk)
        df_review = pd.concat(list_review, ignore_index=True)
        with open(review_pickle, "wb") as f:
            pickle.dump(df_review, f)
        print("save review.pkl")

    # User data
    if os.path.exists(user_pickle):
        print("load user.pkl")
        with open(user_pickle, "rb") as f:
            df_user = pickle.load(f)
    else:
        print("load user JSON")
        df_user = pd.read_json("data/train/yelp_academic_dataset_user.json", lines=True)
        with open(user_pickle, "wb") as f:
            pickle.dump(df_user, f)
        print("save user.pkl")

    return df_business, df_review, df_user


if __name__ == "__main__":
    df_business, df_review, df_user = load_or_create_data()
    print(df_business.columns)
    print(df_review.columns)
    print(df_user.columns)
