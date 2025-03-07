import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Set up relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "youtube.csv")
CLEANED_DATA_PATH = os.path.join(
    PROJECT_ROOT, "dataset", "cleaned_youtube_data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "backend",
                          "multi_output_predictor.pkl")

columns_to_keep = [
    "video_id", "title", "channel_title", "category_id",
    "publish_date", "time_frame", "views", "likes", "comment_count"
]

print("🔄 Loading dataset in chunks...")
chunk_size = 100000
df_list = []
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

# -------------------------------
# Model Training
# -------------------------------
df["engagement_rate"] = df["likes"] / df["views"]
df["comment_rate"] = df["comment_count"] / df["views"]
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

features = ["category_id", "publish_year", "publish_month", "publish_day"]
targets = ["views", "engagement_rate", "comment_rate"]

encoder = LabelEncoder()
df["category_id"] = encoder.fit_transform(df["category_id"])

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Multi-Output Model Trained Successfully!")
print(f"📊 Mean Absolute Error (MAE): {mae}")
print(f"📈 R² Score: {r2}")

joblib.dump(model, MODEL_PATH)
print(f"💾 Multi-Output Model saved at: {MODEL_PATH}")
