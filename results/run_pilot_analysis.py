import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

FINAL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "full_llm_output.csv")
JSONL_LOG_PATH = os.path.join(os.path.dirname(__file__), "full_api_log.jsonl")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    # Load outputs
    if not os.path.exists(FINAL_OUTPUT_PATH):
        print("Missing final output:", FINAL_OUTPUT_PATH)
        return
    df_output = pd.read_csv(FINAL_OUTPUT_PATH)
    print(f"📊 Đã load thành công {len(df_output)} hàng dữ liệu thực nghiệm.")

    # INVALID ratio
    total_rows = len(df_output)
    invalid_rows = df_output[df_output['status'] == 'INVALID'] if 'status' in df_output.columns else pd.DataFrame()
    invalid_count = len(invalid_rows)
    invalid_ratio = invalid_count / total_rows if total_rows > 0 else 0
    print(f"⚠️ Số lượng mẫu bị lỗi (INVALID): {invalid_count}/{total_rows} ({invalid_ratio:.2%})")
    if invalid_ratio > 0.20:
        print("❌ CẢNH BÁO: Tỷ lệ lỗi vượt quá ngưỡng cho phép (> 20%). Cần kiểm tra lại log API!")
    else:
        print("✅ Đạt yêu cầu hệ thống. Tỷ lệ lỗi nằm trong tầm kiểm soát (< 20%).")

    # Load JSONL logs
    log_records = []
    if os.path.exists(JSONL_LOG_PATH):
        with open(JSONL_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log_records.append(json.loads(line))
                except Exception as e:
                    print('Skipping malformed JSONL line', e)
    else:
        print(f'Missing log file: {JSONL_LOG_PATH}')

    df_logs = pd.DataFrame(log_records) if log_records else pd.DataFrame()

    total_cost = df_logs['cost_usd'].sum() if 'cost_usd' in df_logs.columns else 0.0
    avg_latency = df_logs['latency_sec'].mean() if 'latency_sec' in df_logs.columns else np.nan
    total_tokens = 0
    if 'usage' in df_logs.columns:
        try:
            total_tokens = df_logs['usage'].apply(lambda x: x.get('total_tokens', 0) if isinstance(x, dict) else 0).sum()
        except Exception:
            total_tokens = 0

    print("💰 --- BÁO CÁO TÀI CHÍNH & HIỆU NĂNG ---")
    print(f"- Tổng chi phí phát sinh: ${total_cost:.6f} USD")
    print(f"- Tổng số Token tiêu hao: {int(total_tokens):,} tokens")
    print(f"- Thời gian phản hồi trung bình (Average Latency): {avg_latency:.2f} giây")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    if not df_logs.empty and 'latency_sec' in df_logs.columns:
        sns.histplot(data=df_logs, x='latency_sec', kde=True, ax=axes[0], color='teal', bins=15)
        axes[0].set_title('Phân phối Thời gian phản hồi (Latency Distribution)')
        axes[0].set_xlabel('Thời gian (giây)')
        axes[0].set_ylabel('Số lượng mẫu')
    else:
        axes[0].text(0.5, 0.5, 'No latency data', ha='center')

    if 'status' in df_output.columns:
        sns.countplot(data=df_output, x='status', ax=axes[1], palette='Set2')
        axes[1].set_title('Thống kê trạng thái Output của thực nghiệm')
        axes[1].set_xlabel('Trạng thái')
        axes[1].set_ylabel('Số lượng')
    else:
        axes[1].text(0.5, 0.5, 'No status column', ha='center')

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'pilot_performance_metrics.png')
    plt.savefig(fig_path, bbox_inches='tight')
    print('Saved figure to', fig_path)

if __name__ == '__main__':
    main()
