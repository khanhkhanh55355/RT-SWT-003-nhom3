import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ===== Paths =====
OUTPUT_PATH = "../results/pilot_llm_output.csv"
GT_PATH = "../data/pilot_ground_truth.csv"
FIG_PATH = "../figures/fig1_distribution.png"
SUMMARY_PATH = "../results/summary.csv"

# ===== Load data =====
df_output = pd.read_csv(OUTPUT_PATH)
df_gt = pd.read_csv(GT_PATH)

df_output["id"] = df_output["id"].astype(str)
df_gt["ID"] = df_gt["ID"].astype(str)

df = df_output.merge(df_gt, left_on="id", right_on="ID", how="inner")

print(f"Merged rows: {len(df)}")
print(df[["id", "text", "User Story", "llm_output", "Manual Scenario"]].head())

# ===== RQ1: Cosine Similarity =====
expert = df["Manual Scenario"].fillna("").astype(str).tolist()
generated = df["llm_output"].fillna("").astype(str).tolist()

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(expert + generated)

expert_vecs = vectors[:len(expert)]
generated_vecs = vectors[len(expert):]

cosine_scores = []
for i in range(len(df)):
    score = cosine_similarity(expert_vecs[i], generated_vecs[i])[0][0]
    cosine_scores.append(score)

df["cosine_similarity"] = cosine_scores

median_cosine = float(np.median(cosine_scores))
mean_cosine = float(np.mean(cosine_scores))

# Wilcoxon test against threshold 0.85
threshold_cosine = 0.85
diff = np.array(cosine_scores) - threshold_cosine

try:
    wilcoxon_result = wilcoxon(diff, alternative="greater")
    wilcoxon_p = float(wilcoxon_result.pvalue)
except Exception as e:
    wilcoxon_p = np.nan
    print("Wilcoxon error:", e)

rq1_decision = (
    "Reject H0"
    if median_cosine >= threshold_cosine and wilcoxon_p < 0.05
    else "Fail to reject H0"
)

# ===== RQ2: Executable Syntax Rate =====
def is_executable_gherkin(text):
    text = str(text)
    has_scenario = "Scenario:" in text
    has_given = "Given " in text or "\nGiven " in text
    has_when = "When " in text or "\nWhen " in text
    has_then = "Then " in text or "\nThen " in text
    return has_scenario and has_given and has_when and has_then

df["parser_status"] = df["llm_output"].apply(
    lambda x: "PASS" if is_executable_gherkin(x) else "FAIL"
)

pass_count = int((df["parser_status"] == "PASS").sum())
total_count = int(len(df))
fail_count = total_count - pass_count
executable_rate = pass_count / total_count if total_count > 0 else 0

# Binomial exact test against threshold 80%
binom_result = binomtest(
    k=pass_count,
    n=total_count,
    p=0.80,
    alternative="greater"
)

binom_p = float(binom_result.pvalue)

rq2_decision = (
    "Reject H0"
    if executable_rate >= 0.80 and binom_p < 0.05
    else "Fail to reject H0"
)

# ===== Histogram =====
os.makedirs("../figures", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.hist(df["cosine_similarity"], bins=10, edgecolor="black")
plt.axvline(0.85, linestyle="--", label="Threshold = 0.85")
plt.title("Distribution of Cosine Similarity")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()

# ===== Save detailed metric output =====
DETAIL_PATH = "../results/pilot_metric_output.csv"
df.to_csv(DETAIL_PATH, index=False)

# ===== Summary =====
summary = pd.DataFrame([
    {
        "RQ": "RQ1",
        "Metric": "Cosine Semantic Similarity",
        "Threshold": ">= 0.85",
        "Result": round(median_cosine, 4),
        "p_value": round(wilcoxon_p, 6) if not np.isnan(wilcoxon_p) else "NA",
        "Decision": rq1_decision,
    },
    {
        "RQ": "RQ2",
        "Metric": "Executable Syntax Rate",
        "Threshold": ">= 80%",
        "Result": f"{executable_rate:.2%}",
        "p_value": round(binom_p, 6),
        "Decision": rq2_decision,
    },
])

summary.to_csv(SUMMARY_PATH, index=False)

print("\n===== RQ1 Result =====")
print(f"Median Cosine Similarity: {median_cosine:.4f}")
print(f"Mean Cosine Similarity: {mean_cosine:.4f}")
print(f"Wilcoxon p-value: {wilcoxon_p}")
print(f"Decision: {rq1_decision}")

print("\n===== RQ2 Result =====")
print(f"PASS: {pass_count}")
print(f"FAIL: {fail_count}")
print(f"Executable Rate: {executable_rate:.2%}")
print(f"Binomial p-value: {binom_p}")
print(f"Decision: {rq2_decision}")

print("\nSaved files:")
print(f"- {DETAIL_PATH}")
print(f"- {SUMMARY_PATH}")
print(f"- {FIG_PATH}")