"""
不确定性熵值与SOH估计误差相关性分析脚本
验证：
1. 各熵指标与绝对误差的皮尔逊/斯皮尔曼相关系数
2. 按熵分位数统计误差分布
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import argparse


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"true_soh", "pred_soh_text", "clbp_entropy_norm",
                "llm_entropy_numeric", "combined_uncertainty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少列: {missing}")
    # 过滤无效预测
    df = df[np.isfinite(df["pred_soh_text"])].copy()
    df["abs_error"] = np.abs(df["true_soh"] - df["pred_soh_text"])
    return df


def correlation_analysis(df: pd.DataFrame, entropy_col: str, error_col: str = "abs_error"):
    """计算皮尔逊和斯皮尔曼相关系数"""
    valid = df[[entropy_col, error_col]].dropna()
    valid = valid[np.isfinite(valid[entropy_col])]
    if len(valid) < 10:
        return None

    x = valid[entropy_col].values
    y = valid[error_col].values

    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        "n":          len(valid),
        "pearson_r":  pearson_r,
        "pearson_p":  pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def quantile_analysis(df: pd.DataFrame, entropy_col: str,
                      error_col: str = "abs_error",
                      error_threshold: float = 0.005):
    """按熵值分位数统计误差分布"""
    valid = df[[entropy_col, error_col]].dropna()
    valid = valid[np.isfinite(valid[entropy_col])].copy()
    if len(valid) < 10:
        return None

    ent = valid[entropy_col].values
    q25, q50, q75 = np.percentile(ent, [25, 50, 75])

    # 自定义分段：低/中/高
    bins = [
        ("低  (≤ Q25={:.3f})".format(q25),  valid[valid[entropy_col] <= q25]),
        ("中  (Q25~Q75={:.3f}~{:.3f})".format(q25, q75),
         valid[(valid[entropy_col] > q25) & (valid[entropy_col] <= q75)]),
        ("高  (> Q75={:.3f})".format(q75),  valid[valid[entropy_col] > q75]),
    ]

    rows = []
    for label, subset in bins:
        if len(subset) == 0:
            continue
        pct = (subset[error_col] <= error_threshold).mean() * 100
        rows.append({
            "区间":                   label,
            "样本数":                 len(subset),
            f"误差≤{error_threshold}占比(%)": round(pct, 1),
            "平均绝对误差":           round(subset[error_col].mean(), 5),
            "中位绝对误差":           round(subset[error_col].median(), 5),
        })
    return pd.DataFrame(rows)


def decile_analysis(df: pd.DataFrame, entropy_col: str,
                    error_col: str = "abs_error",
                    error_threshold: float = 0.005,
                    n_bins: int = 10):
    """十分位数细粒度分析"""
    valid = df[[entropy_col, error_col]].dropna()
    valid = valid[np.isfinite(valid[entropy_col])].copy()
    if len(valid) < n_bins:
        return None

    labels = [f"D{i+1}" for i in range(n_bins)]
    valid["decile"] = pd.qcut(valid[entropy_col], q=n_bins, labels=labels, duplicates="drop")

    rows = []
    for d, grp in valid.groupby("decile", observed=True):
        lo = grp[entropy_col].min()
        hi = grp[entropy_col].max()
        pct = (grp[error_col] <= error_threshold).mean() * 100
        rows.append({
            "十分位": d,
            "熵范围": f"[{lo:.4f}, {hi:.4f}]",
            "样本数": len(grp),
            f"误差≤{error_threshold}占比(%)": round(pct, 1),
            "均值绝对误差": round(grp[error_col].mean(), 5),
        })
    return pd.DataFrame(rows)


def sig_flag(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def print_report(df: pd.DataFrame, error_threshold: float = 0.005):
    sep = "=" * 70

    entropy_cols = {
        "CLBP熵 (clbp_entropy_norm)":     "clbp_entropy_norm",
        "LLM数值熵 (llm_entropy_numeric)": "llm_entropy_numeric",
        "综合不确定性 (combined_uncertainty)": "combined_uncertainty",
    }

    print(sep)
    print("  BatteryGPT — 不确定性熵值 vs SOH估计误差  相关性分析报告")
    print(sep)
    print(f"  有效样本数: {len(df)}")
    print(f"  误差阈值  : {error_threshold}")
    print(f"  true_SOH  : [{df['true_soh'].min():.4f}, {df['true_soh'].max():.4f}]")
    print(f"  |误差|均值: {df['abs_error'].mean():.5f}   中位: {df['abs_error'].median():.5f}")
    print()

    # ── 1. 相关系数 ──────────────────────────────────────────────
    print(sep)
    print("【1】皮尔逊 / 斯皮尔曼 相关系数")
    print(sep)
    header = f"  {'指标':<32} {'Pearson r':>10} {'p值':>10} {'显著':>5}  " \
             f"{'Spearman ρ':>11} {'p值':>10} {'显著':>5}"
    print(header)
    print("  " + "-" * 90)

    for name, col in entropy_cols.items():
        res = correlation_analysis(df, col)
        if res is None:
            print(f"  {name:<32}  数据不足")
            continue
        sf_p = sig_flag(res["pearson_p"])
        sf_s = sig_flag(res["spearman_p"])
        print(f"  {name:<32} "
              f"{res['pearson_r']:>10.4f} {res['pearson_p']:>10.2e} {sf_p:>5}  "
              f"{res['spearman_r']:>11.4f} {res['spearman_p']:>10.2e} {sf_s:>5}")

    print("\n  显著性标记: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05")
    print()

    # ── 2. 分位数统计 ────────────────────────────────────────────
    for name, col in entropy_cols.items():
        print(sep)
        print(f"【2】分位数误差统计 — {name}")
        print(sep)
        tbl = quantile_analysis(df, col, error_threshold=error_threshold)
        if tbl is None:
            print("  数据不足，跳过")
            continue
        print(tbl.to_string(index=False))
        print()

    # ── 3. 十分位细粒度（重点：综合不确定性）────────────────────
    print(sep)
    print("【3】十分位细粒度分布 — 综合不确定性 (combined_uncertainty)")
    print(sep)
    tbl = decile_analysis(df, "combined_uncertainty", error_threshold=error_threshold)
    if tbl is not None:
        print(tbl.to_string(index=False))
    print()

    # ── 4. 误差阈值敏感性 ────────────────────────────────────────
    print(sep)
    print("【4】不同误差阈值下 综合不确定性 低/高熵组 对比")
    print(sep)
    valid = df[["combined_uncertainty", "abs_error"]].dropna()
    valid = valid[np.isfinite(valid["combined_uncertainty"])]
    med_ent = valid["combined_uncertainty"].median()
    low_grp  = valid[valid["combined_uncertainty"] <= med_ent]["abs_error"]
    high_grp = valid[valid["combined_uncertainty"] >  med_ent]["abs_error"]

    thresholds = [0.002, 0.005, 0.010, 0.020, 0.050]
    print(f"  中位熵分割点: {med_ent:.4f}  低熵组 N={len(low_grp)}  高熵组 N={len(high_grp)}")
    print(f"\n  {'阈值':>8}  {'低熵组占比(%)':>14}  {'高熵组占比(%)':>14}  {'差值':>8}")
    print("  " + "-" * 52)
    for thr in thresholds:
        lo_pct = (low_grp  <= thr).mean() * 100
        hi_pct = (high_grp <= thr).mean() * 100
        print(f"  {thr:>8.3f}  {lo_pct:>14.1f}  {hi_pct:>14.1f}  {lo_pct - hi_pct:>8.1f}")
    print()

    # ── 5. Mann-Whitney U 检验（低熵 vs 高熵组误差是否显著不同） ─
    print(sep)
    print("【5】Mann-Whitney U 检验（低熵 vs 高熵 绝对误差）")
    print(sep)
    u_stat, u_p = stats.mannwhitneyu(low_grp, high_grp, alternative="less")
    print(f"  H0: 低熵组误差 ≥ 高熵组误差 (单侧检验)")
    print(f"  U统计量={u_stat:.1f}  p值={u_p:.3e}  {sig_flag(u_p)}")
    if u_p < 0.05:
        print("  ✅ 结论：低熵组误差显著小于高熵组（置信度95%）")
    else:
        print("  ❌ 结论：未发现显著差异")
    print()
    print(sep)
    print("  分析完成")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="不确定性熵值与SOH误差相关性分析")
    parser.add_argument("--csv",   type=str,   default="soh_predictions_full.csv")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="误差阈值（默认0.005）")
    parser.add_argument("--output", type=str, default=None,
                        help="可选：将报告保存到文本文件")
    args = parser.parse_args()

    df = load_data(args.csv)
    
    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, "w", encoding="utf-8") as f:
            sys.stdout = f
            print_report(df, error_threshold=args.threshold)
        sys.stdout = original_stdout
        print(f"报告已保存至: {args.output}")
    else:
        print_report(df, error_threshold=args.threshold)


if __name__ == "__main__":
    main()