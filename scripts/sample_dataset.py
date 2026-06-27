import pandas as pd

# ── Cấu hình ──────────────────────────────────────────
INPUT_PATH  = "data/raw/full_dataset.csv"
OUTPUT_PATH = "data/sampled_100.csv"   # ← đổi chỗ này
RANDOM_SEED = 42
N_TOTAL     = 100

# ── Đọc pool 348 cặp ──────────────────────────────────
df = pd.read_csv(INPUT_PATH)
print(f"Pool: {len(df)} cặp")
print(df["domain"].value_counts())

# ── Tính số cặp mỗi domain (tỷ lệ thuận) ─────────────
domain_counts = df["domain"].value_counts()
total_pool    = len(df)

samples = []
allocated = 0
domains   = domain_counts.index.tolist()

for i, domain in enumerate(domains):
    if i < len(domains) - 1:
        n = round(domain_counts[domain] / total_pool * N_TOTAL)
    else:
        n = N_TOTAL - allocated
    allocated += n
    part = df[df["domain"] == domain].sample(n=n, random_state=RANDOM_SEED)
    samples.append(part)
    print(f"  {domain}: {n} cặp")

df_100 = pd.concat(samples).reset_index(drop=True)
print(f"\nTổng sampled: {len(df_100)} cặp")

df_100.to_csv(OUTPUT_PATH, index=False)
print(f"Đã lưu: {OUTPUT_PATH}")