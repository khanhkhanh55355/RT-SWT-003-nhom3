import pandas as pd

# ── Đường dẫn ─────────────────────────────────────────
INPUT_PATH  = "data/raw/Requirements.xlsx"
OUTPUT_PATH = "data/raw/full_dataset.csv"   # lưu vào raw/

# ── Hàm kiểm tra ──────────────────────────────────────
def is_connextra(text):
    if not isinstance(text, str): return False
    t = text.lower()
    return "as a" in t and "i want" in t

def is_valid_gherkin(text):
    if not isinstance(text, str): return False
    t = text.lower()
    return "given" in t and "when" in t and "then" in t

def assign_domain(id_val):
    if 1 <= id_val <= 150:   return "Digital Asset Management"
    elif 151 <= id_val <= 275: return "Brand Management"
    elif 276 <= id_val <= 425: return "Marketing Operations Platform"
    elif 426 <= id_val <= 500: return "Marketing Compliance"

# ── Chạy ──────────────────────────────────────────────
df = pd.read_excel(INPUT_PATH)
df["domain"] = df["ID"].apply(assign_domain)

df_filtered = df[
    df["User Story"].apply(is_connextra) &
    df["Manual Scenario"].apply(is_valid_gherkin)
].copy()

print(f"Sau lọc: {len(df_filtered)} cặp hợp lệ")
print(df_filtered["domain"].value_counts())

df_filtered.to_csv(OUTPUT_PATH, index=False)
print(f"Đã lưu: {OUTPUT_PATH}")