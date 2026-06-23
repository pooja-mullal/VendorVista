from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# ===== LOAD DATA & MODEL =====
df = pd.read_csv("dataset/vendor_performance.csv")
model = joblib.load("model/vendor_risk_model.pkl")
risk_encoder = joblib.load("model/risk_encoder.pkl")


# ===== HOME PAGE =====
@app.route("/")
def home():
    total_vendors = df["vendor_id"].nunique()
    vendors = sorted(df["vendor_name"].unique())

    return render_template(
        "index.html",
        total_vendors=total_vendors,
        vendors=vendors
    )


# ===== REDIRECT FROM DROPDOWN =====
@app.route("/vendor")
def vendor_redirect():
    vendor_name = request.args.get("vendor_name")
    return vendor_view(vendor_name)


# ===== VENDOR DASHBOARD =====
@app.route("/vendor/<vendor_name>")
def vendor_view(vendor_name):
    months_ahead = request.args.get("months_ahead", type=int)

    vendor_data = df[df["vendor_name"] == vendor_name]

    # Time series data
    months = vendor_data["month"].tolist()
    scores = vendor_data["performance_score"].tolist()

    # ===== RISK DISTRIBUTION =====
    risk_counts_series = vendor_data["risk_level"].value_counts()
    risk_labels = risk_counts_series.index.tolist()
    risk_counts = risk_counts_series.values.tolist()

    # ✅ BACKEND SAFEGUARD (IMPORTANT)
    if not risk_labels or len(risk_labels) == 0:
        risk_labels = ["Low", "Medium", "High"]
        risk_counts = [0, 0, 0]

    # Average score
    avg_score = round(vendor_data["performance_score"].mean(), 2)

    # Prediction block
    prediction = None
    if months_ahead:
        predicted_score, risk = predict_future_risk(vendor_data, months_ahead)
        prediction = {
            "score": predicted_score,
            "risk": risk,
            "months": months_ahead
        }

    return render_template(
        "vendor.html",
        vendor_name=vendor_name,
        months=months,
        scores=scores,
        risk_labels=risk_labels,
        risk_counts=risk_counts,
        avg_score=avg_score,
        prediction=prediction
    )


# ===== ML + HYBRID PREDICTION =====
def predict_future_risk(vendor_data, months_ahead):

    # Sort by month
    vendor_data = vendor_data.sort_values("month")

    # Latest available record
    latest = vendor_data.iloc[-1]

    # Trend calculation
    recent_scores = vendor_data["performance_score"].tail(5)
    score_change = recent_scores.diff().mean()
    if pd.isna(score_change):
        score_change = 0

    # Predict future score
    predicted_score = latest["performance_score"] + (score_change * months_ahead)
    predicted_score = max(0, min(100, predicted_score))

    # Prepare ML input (same order as training)
    X_pred = [[
        predicted_score,
        latest["on_time_delivery_rate"],
        latest["average_delivery_delay"],
        latest["order_fulfillment_rate"],
        latest["quality_score"],
        latest["defect_rate"],
        latest["return_rate"],
        latest["complaint_count"],
        latest["response_time"],
        latest["service_rating"],
        latest["month"] + months_ahead,
        score_change
    ]]

    # ML prediction
    risk_encoded = model.predict(X_pred)[0]
    ml_risk = risk_encoder.inverse_transform([risk_encoded])[0]

    # ===== HYBRID DECISION LOGIC =====
    if predicted_score >= 80:
        final_risk = "Low"
    elif predicted_score < 55:
        final_risk = "High"
    else:
        final_risk = ml_risk

    return round(predicted_score, 2), final_risk


# ===== PREDICT BUTTON HANDLER =====
@app.route("/predict", methods=["POST"])
def predict():
    vendor_name = request.form.get("vendor_name")
    months_ahead = request.form.get("months_ahead")

    return redirect(
        url_for(
            "vendor_view",
            vendor_name=vendor_name,
            months_ahead=months_ahead
        )
    )


# ===== RUN APP =====
if __name__ == "__main__":
    app.run(debug=True)
