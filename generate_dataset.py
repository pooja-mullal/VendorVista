import pandas as pd
import random

vendors = [f"V{str(i).zfill(3)}" for i in range(1, 51)]

prefix = [
    "Alpha", "Beta", "Gamma", "Delta", "Orion", "Nova", "Apex", "Zenith",
    "Vertex", "Nimbus", "Atlas", "Phoenix", "Omega", "Titan", "Aurora"
]

suffix = [
    "Logistics", "Supplies", "Industries", "Traders", "Enterprises",
    "Manufacturing", "Solutions", "Services", "Corporation"
]

vendor_names = [
    f"{random.choice(prefix)} {random.choice(suffix)}"
    for _ in range(50)
]

data = []

for vid, vname in zip(vendors, vendor_names):

    base_performance = random.randint(65, 90)

    for month in range(1, 25):  # 24 months

        # --- DELIVERY FEATURES (A) ---
        on_time_delivery_rate = random.randint(70, 98)
        avg_delivery_delay = round(random.uniform(0.5, 6.0), 2)
        order_fulfillment_rate = random.randint(75, 100)

        # --- QUALITY FEATURES (B) ---
        quality_score = round(random.uniform(6.0, 9.5), 1)
        defect_rate = round(random.uniform(0.5, 6.0), 2)
        return_rate = round(random.uniform(0.5, 5.0), 2)

        # --- RELIABILITY & SERVICE FEATURES (D) ---
        complaint_count = random.randint(0, 10)
        response_time = round(random.uniform(2.0, 48.0), 1)
        service_rating = round(random.uniform(6.0, 9.5), 1)

        # --- PERFORMANCE SCORE CALCULATION ---
        performance_score = (
            0.35 * on_time_delivery_rate +
            0.25 * quality_score * 10 +
            0.20 * service_rating * 10 -
            0.10 * defect_rate * 10 -
            0.05 * avg_delivery_delay * 5 -
            0.05 * complaint_count
        )

        performance_score = round(
            max(40, min(95, performance_score)), 2
        )

        # --- RISK LEVEL ---
        if performance_score >= 80:
            risk_level = "Low"
        elif performance_score >= 55:
            risk_level = "Medium"
        else:
            risk_level = "High"

        data.append([
            vid, vname, month,
            on_time_delivery_rate,
            avg_delivery_delay,
            order_fulfillment_rate,
            quality_score,
            defect_rate,
            return_rate,
            complaint_count,
            response_time,
            service_rating,
            performance_score,
            risk_level
        ])

df = pd.DataFrame(data, columns=[
    "vendor_id",
    "vendor_name",
    "month",
    "on_time_delivery_rate",
    "average_delivery_delay",
    "order_fulfillment_rate",
    "quality_score",
    "defect_rate",
    "return_rate",
    "complaint_count",
    "response_time",
    "service_rating",
    "performance_score",
    "risk_level"
])

df.to_csv("dataset/vendor_performance.csv", index=False)

print("Updated dataset with delivery, quality, and service features generated!")
