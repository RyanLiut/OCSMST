"""λ 权重扫描折线图 + 最优 λ 案例数柱状图（论文级 PDF）。

数据：工作簿「消融实验-融合权重（无norm-cooc_minmax）」表
      blend_Emb (Qwen3-Embdding-8B)（cooc_minmax）列，10 个 case（6/12 无数据自动跳过）。
- 折线：每个 λ 的宏平均 Top-1 F1；
- 柱状：每个 λ 作为"该 case 最优 λ"的案例数（并列最优按 1/k 均分，故柱高总和 = 10）。

输出：ARR_Oct/main_evaluation/blend_cooc_minmax/lambda_f1_and_best_lambda.pdf（及同名 png 预览）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'ARR_Oct/main_evaluation/blend_cooc_minmax'
XLSX = Path(__file__).resolve().parents[4] / '实验记录_完整.xlsx'
SHEET = '消融实验-融合权重（无norm-cooc_minmax）'
COL = 'blend_Emb (Qwen3-Embdding-8B)（cooc_minmax）'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,
})


def main():
    df = pd.read_excel(XLSX, SHEET)
    df['case'] = df['case'].ffill().astype(int)
    wide = df.pivot_table(index='case', columns='权重', values=COL)
    wide = wide.dropna(how='all')
    lambdas = [round(w, 1) for w in wide.columns]
    mean_f1 = wide.mean(axis=0).to_numpy()

    best = wide.max(axis=1)
    counts = np.zeros(len(lambdas))
    for case in wide.index:
        ties = np.isclose(wide.loc[case].to_numpy(), best[case])
        counts[ties] += 1.0 / ties.sum()

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax2 = ax.twinx()

    bars = ax2.bar(lambdas, counts, width=0.065, color='0.78', edgecolor='0.45',
                   linewidth=0.5, zorder=1, label='Datasets with optimal $\\lambda$')
    ax2.axvline(0.5, color='#D62728', linestyle=(0, (4, 3)), linewidth=0.9, zorder=2)

    line, = ax.plot(lambdas, mean_f1, color='#1f4e79', linewidth=1.4, zorder=3,
                    label='Average $F_1$')
    idx_star = lambdas.index(0.1)
    others = [(l, v) for k, (l, v) in enumerate(zip(lambdas, mean_f1)) if k != idx_star]
    ax.plot(*zip(*others), linestyle='none', marker='o', markersize=3.4,
            color='#1f4e79', zorder=4)
    star, = ax.plot([0.1], [mean_f1[idx_star]], linestyle='none', marker='*',
                    markersize=10, color='#1f4e79', zorder=5)

    peak = int(np.argmax(mean_f1))
    ax.annotate(f'$\\lambda={lambdas[peak]:.1f}$',
                xy=(lambdas[peak], mean_f1[peak]),
                xytext=(lambdas[peak] + 0.13, mean_f1[peak] + 0.045),
                fontsize=7.5, color='#1f4e79',
                arrowprops=dict(arrowstyle='-', color='#1f4e79', lw=0.6,
                                shrinkA=1, shrinkB=2))

    ax.set_xlabel('$\\lambda$')
    ax.set_ylabel('Average $F_1$')
    ax2.set_ylabel('Datasets with optimal $\\lambda$')
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks(lambdas)
    ax.set_ylim(0.28, 0.80)
    ax2.set_ylim(0, 9.0)
    ax2.set_yticks([0, 2, 4, 6, 8])
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', color='0.88', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top',):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_position(('outward', 0))

    ax.legend(handles=[line, bars], loc='upper right', frameon=False,
              handlelength=1.6, borderaxespad=0.2, labelspacing=0.35)

    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'lambda_f1_and_best_lambda.{ext}',
                    dpi=300 if ext == 'png' else None, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)
    print('λ 平均 F1:', {l: round(float(m), 4) for l, m in zip(lambdas, mean_f1)})
    print('最优 λ 案例数（并列均分）:', {l: round(float(c), 2) for l, c in zip(lambdas, counts)})
    print('已保存 →', OUT / 'lambda_f1_and_best_lambda.pdf')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
