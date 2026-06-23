import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load updated dataset
df = pd.read_csv("dataset/vendor_performance.csv")

# Sort data
df = df.sort_values(["vendor_id", "month"])

# Create trend feature (derived)
df["score_change"] = df.groupby("vendor_id")["performance_score"].diff().fillna(0)

# Encode target variable
label_encoder = LabelEncoder()
df["risk_encoded"] = label_encoder.fit_transform(df["risk_level"])

# ---------------------------
# FEATURES USED FOR TRAINING
# ---------------------------
feature_columns = [
    "performance_score",
    "on_time_delivery_rate",
    "average_delivery_delay",
    "order_fulfillment_rate",
    "quality_score",
    "defect_rate",
    "return_rate",
    "complaint_count",
    "response_time",
    "service_rating",
    "month",
    "score_change"
]

X = df[feature_columns]
y = df["risk_encoded"]

# Train Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X, y)

# Save model and encoder
joblib.dump(model, "model/vendor_risk_model.pkl")
joblib.dump(label_encoder, "model/risk_encoder.pkl")

print("Random Forest model trained with extended feature set!")
