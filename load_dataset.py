import os
import pandas as pd
import numpy as np

# Set up relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "youtube.csv")
CLEANED_DATA_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "cleaned_youtube_data.csv")

# Load only necessary columns
columns_to_keep = [
    "video_id", "title", "channel_title", "category_id",
    "publish_date", "time_frame", "views", "likes", "comment_count"
]

chunk_size = 100000
df_list = []

print("🔄 Loading dataset in chunks...")
for chunk in pd.read_csv(RAW_DATA_PATH, usecols=columns_to_keep, chunksize=chunk_size, low_memory=False):
    df_list.append(chunk)

df = pd.concat(df_list, ignore_index=True)
print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

df["publish_date"] = pd.to_datetime(df["publish_date"], errors='coerce')
df["publish_year"] = df["publish_date"].dt.year
df["publish_month"] = df["publish_date"].dt.month
df["publish_day"] = df["publish_date"].dt.day
df["category_id"] = df["category_id"].astype(str)
df.dropna(inplace=True)

df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"✅ Cleaned dataset saved at: {CLEANED_DATA_PATH}")
