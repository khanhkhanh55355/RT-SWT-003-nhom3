import pandas as pd

# ── Cấu hình ──────────────────────────────────────────
INPUT_PATH          = "data/sampled_100.csv"
PILOT_SAMPLE_PATH   = "data/pilot_sample.csv"
PILOT_GT_PATH       = "data/pilot_ground_truth.csv"
RANDOM_SEED         = 42
N_PILOT             = 20   # 20% của 100

# ── Đọc 100 cặp ───────────────────────────────────────
df = pd.read_csv(INPUT_PATH)

# ── Random sample 20 cặp ──────────────────────────────
pilot = df.sample(n=N_PILOT, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"Pilot: {len(pilot)} cặp")
print(pilot["domain"].value_counts())

# ── pilot_sample.csv — chỉ có input cho LLM ───────────
pilot[["ID", "domain", "User Story"]].to_csv(
    PILOT_SAMPLE_PATH, index=False
)

# ── pilot_ground_truth.csv — có thêm Gherkin chuẩn ───
pilot[["ID", "domain", "User Story", "Manual Scenario"]].to_csv(
    PILOT_GT_PATH, index=False
)

print(f"Đã lưu: {PILOT_SAMPLE_PATH}")
print(f"Đã lưu: {PILOT_GT_PATH}")