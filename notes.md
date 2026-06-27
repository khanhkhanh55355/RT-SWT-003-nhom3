# Notes — quyết định kỹ thuật & error log

## Nhóm

- **Tên nhóm:** _(điền)_
- **Seed pilot sample:** _(điền seed khi tạo `data/pilot_sample.csv`, ~10–20% N)_

## Quyết định kỹ thuật

| Mục | Quyết định |
|-----|------------|
| Model LLM | `gpt-4o-2024-11-20` |
| Temperature | `0` |
| Script chạy batch | `scripts/run_experiment.py` |
| Script tính metric | `scripts/compute_metric.py` |

## Pilot [MỚI T6]

- [ ] Tạo `data/pilot_sample.csv` (10–20% N, ghi seed ở trên)
- [ ] Annotate `data/pilot_ground_truth.csv` + IAA
- [ ] Chạy `scripts/test_api.py` (gate E3)
- [ ] Chạy experiment pilot → `results/pilot_llm_output.csv`
- [ ] Phân tích → `results/pilot_analysis.ipynb`

## Full run [MỚI T7–T8]

- [ ] Annotate `data/full_ground_truth.csv`
- [ ] Chạy full experiment → `results/full_llm_output.csv`
- [ ] Phân tích → `results/full_analysis.ipynb`
- [ ] Tổng hợp → `results/summary.csv`
- [ ] Vẽ `figures/fig1_distribution.png`, `figures/fig2_comparison.png`

## Error log

| Ngày | Mô tả lỗi | Cách xử lý |
|------|-----------|------------|
| | | |
