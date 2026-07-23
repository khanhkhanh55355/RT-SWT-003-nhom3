import pandas as pd

df = pd.read_csv("data/sampled_100.csv")

# full_ground_truth = tất cả 100 cặp, có Manual Scenario
df[["ID", "domain", "User Story", "Manual Scenario"]].to_csv(
    "data/full_ground_truth.csv", index=False
)
print(f"Saved {len(df)} rows")
