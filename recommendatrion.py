import pandas as pd
import json

json_business = "data/train/yelp_academic_dataset_business.json"
df_business = pd.read_json(json_business, lines = True)

json_review = "data/train/yelp_academic_dataset_review.json"
chunck_review = pd.read_json(json_review, lines = True, chunksize=10000)
df_review = next(chunck_review)

json_user = "data/train/yelp_academic_dataset_user.json"
df_user = pd.read_json(json_user, lines = True)

print(df_business.columns)
print(df_review.columns)
print(df_user.columns)