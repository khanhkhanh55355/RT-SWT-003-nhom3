# Notes — quyết định kỹ thuật & error log

## Nhóm

- **Tên nhóm:** RT-SWT-003-Nhóm 3
- **Seed pilot sample:** 42

## Quyết định kỹ thuật

| Mục | Quyết định |
|-----|------------|
| Model LLM | `gpt-4o-2024-11-20` |
| Temperature | `0` |
| Script chạy batch | `scripts/run_experiment.py` |
| Script tính metric | `scripts/compute_metric.py` |
| Embedding cosine (MS) | `text-embedding-3-small` (hoặc `text-embedding-3-large` qua `OPENAI_EMBEDDING_MODEL`) |

## Pilot [MỚI T6]

- [x] Tạo `data/pilot_sample.csv` (10–20% N, ghi seed ở trên)
- [ ] Annotate `data/pilot_ground_truth.csv` + IAA
- [x] Chạy `scripts/test_api.py` (gate E3)
- [x] Chạy experiment pilot → `results/pilot_llm_output.csv`
- [ ] Phân tích → `results/pilot_analysis.ipynb`

## Full run [MỚI T7–T8]

- [x] Annotate `data/full_ground_truth.csv`
- [x] Chạy full experiment → `results/full_llm_output.csv` (LR — `scripts/run_full.py`)
- [ ] Phân tích → `results/full_analysis.ipynb`
- [ ] Tổng hợp → `results/summary.csv`
- [ ] Vẽ `figures/fig1_distribution.png`, `figures/fig2_comparison.png`

## Error log

| Ngày | Mô tả lỗi | Cách xử lý |
|------|-----------|------------|
| 2026-07-21 | Full run simulated cũ (121 dòng fake) | Xóa log cũ, chạy lại bằng API thật |
| 2026-07-22 | Checkpoint cũ khiến `run_full.py` skip toàn bộ | Xóa output/checkpoint cũ, chạy lại từ đầu |

## Full Run Log [LR — Phạm Hoàng Đức Minh]

### Lần 3 — 2026-07-22 (prompt Gherkin strict v2)

#### Cấu hình
- Prompt: executable BDD Gherkin, 10 rules + few-shot example (re-order pages)
- User message: `User Story:\n{text}`
- Pilot: N=20 → `results/pilot_llm_output.csv`
- Full: N=100 → `results/full_llm_output.csv`

#### Kết quả
- Pilot: **20/20 success**, chi phí ~$0.03
- Full: **100/100 success**, chi phí ~$0.15

---

### Lần 2 (chính thức) — 2026-07-22

**Commit:** `a7cc375`

#### Cấu hình
- Dataset: `data/sampled_100.csv` (N=100)
- Model: `gpt-4o-2024-11-20`, temperature=0
- Prompt: QA expert BDD (`BDD_SYSTEM_PROMPT` trong `scripts/run_experiment.py`) — có few-shot example (Page Builder reorder)
- Checkpoint interval: 50
- Script: `scripts/run_full.py` → `scripts/run_experiment.py`

#### Kết quả
- Output: `results/full_llm_output.csv` — **100/100 success**, 0 INVALID
- Log: `results/full_api_log.txt`, `results/full_api_log.jsonl`
- Thời gian chạy: ~00:44–00:46 UTC (2026-07-22)
- Tổng chi phí API: **~$0.166** (`cumulative_cost_usd` = 0.1659275)

#### Ghi chú cho MS (cosine similarity)
- Embedding: `text-embedding-3-small` (đổi sang `text-embedding-3-large` qua `OPENAI_EMBEDDING_MODEL` trong `.env`)
- MS dùng `scripts/ms_pilot_analysis.py` / `scripts/ms_analysis.py` — không dùng `all-MiniLM-L6-v2`

---

### Lần 1 — 2026-07-21 (đã thay thế)

- Prompt profile cũ: `pilot`
- Tổng chi phí API: ~$0.12
- Superseded bởi lần 2 (prompt BDD mới + re-run sạch)

## Data Preparation Log [DG — Hồ Ngọc Bảo Trân]

**Ngày:** 2026-06-27

### Kiểm tra file gốc (Requirements.xlsx)
- Tổng số cặp: 500
- Các cột: ID, User Story, Requirements, Manual Scenario
- Null values: 0

### Kết quả kiểm tra tiêu chí lọc
- Connextra format: 351 / 500
- Gherkin hợp lệ (Given/When/Then): 496 / 500
- **Cả hai tiêu chí đạt: 348 / 500** ← pool để sample

### Vấn đề: Không có cột domain
File gốc không có cột domain. Gán thủ công theo paper:
- ID 1–150 → Digital Asset Management
- ID 151–275 → Brand Management
- ID 276–425 → Marketing Operations Platform
- ID 426–500 → Marketing Compliance

### Seed
- Random seed = 42

### Kết quả lọc thực tế
- Digital Asset Management: 115 cặp hợp lệ
- Brand Management: 97 cặp hợp lệ
- Marketing Operations Platform: 90 cặp hợp lệ
- Marketing Compliance: 46 cặp hợp lệ
- **Tổng pool: 348 cặp** → đủ để sample 100

### Kết quả sampling

**Working dataset (N=100):** data/sampled_100.csv
- Digital Asset Management: 33 cặp
- Brand Management: 28 cặp
- Marketing Operations Platform: 26 cặp
- Marketing Compliance: 13 cặp

**Pilot (N=20):** data/pilot_sample.csv + data/pilot_ground_truth.csv
- Random seed: 42
- Sample từ sampled_100.csv, không stratified thêm