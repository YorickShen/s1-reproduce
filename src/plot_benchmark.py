import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = PROJECT_ROOT / "outputs" / "eval_results.jsonl"
OUTPUT_IMG = PROJECT_ROOT / "outputs" / "s1_benchmark_visual.png"

def plot_dashboard():
    # 真实实测数据（基于本次 63-Step 四大试金石）
    categories = ["Algebra\n(s1K-325)", "Combinatorics\n(s1K-170)", "Geometry\n(s1K-53)", "Number Theory\n(Factor)"]
    baseline_tokens = [1536, 1536, 1536, 1536]
    s1_tokens = [1247, 1250, 1250, 1248]

    baseline_correct = [0, 0, 0, 0]      # 0/4 PASS
    s1_correct = [1, 0, 0, 1]            # 2/4 PASS (50%)

    # 设置学术绘图风格
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- 图 1：各题思考 Token 消耗对比 ---
    x = range(len(categories))
    width = 0.35
    b_bars = ax1.bar([i - width/2 for i in x], baseline_tokens, width=width, label='Baseline (Unconstrained)', color='#9e9e9e', alpha=0.9)
    s1_bars = ax1.bar([i + width/2 for i in x], s1_tokens, width=width, label='s1 (Budget Forcing)', color='#1f77b4', alpha=0.95)

    ax1.set_ylabel('Thinking Tokens', fontsize=12, fontweight='bold')
    ax1.set_title('Test-Time Compute: Thinking Tokens per Question', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_ylim(0, 1800)

    # --- 图 2：宏观准确率绝对差距对比 (50% vs 0%) ---
    models = ['Baseline\n(0-Shot / Greedy)', 's1 (63-Step)\nBudget Forcing']
    accs = [0.0, 50.0]
    colors = ['#9e9e9e', '#2ca02c']
    bars = ax2.bar(models, accs, color=colors, width=0.45, alpha=0.95)

    ax2.set_ylabel('Benchmark Pass@1 Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Benchmark Accuracy: 0% vs 50% (+50% Absolute Gain)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # 标注数值与 PASS 标签
    for bar, acc, text in zip(bars, accs, ["0/4 (0%)", "2/4 (50%) \n[PASS: Q1, Q4]"]):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 3, text, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"[*] [可视化大功告成] 高清大图已保存至: {OUTPUT_IMG}")

if __name__ == "__main__":
    plot_dashboard()