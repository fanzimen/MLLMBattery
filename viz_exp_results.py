import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = pathlib.Path(__file__).resolve().parent
EXP_CSV = BASE_DIR / "exp_results_new.csv"

plt.rc("font", family="Times New Roman")


def safe_read_csv(path: pathlib.Path):
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return None

def collect_per_cell_metrics(df_exp: pd.DataFrame):
    """
    返回 per_cell_metrics[(data, model)] = {"mae": [...], "rmse": [...]}
    数据来源：每个 result_dir 下的 summary.csv (file_name,num_samples,mae,rmse)
    """
    per_cell = defaultdict(lambda: {"mae": [], "rmse": []})

    for _, row in df_exp.iterrows():
        data_name = row["data"]
        model_name = row["model"]
        result_dir = pathlib.Path(row["result_dir"])
        summary_path = result_dir / "summary.csv"
        df_sum = safe_read_csv(summary_path)
        if df_sum is None or df_sum.empty:
            print(f"[WARN] summary.csv missing or empty in {result_dir}")
            continue
        df_sum = df_sum.loc[:, ~df_sum.columns.str.contains("^Unnamed")]

        key = (data_name, model_name)
        per_cell[key]["mae"].extend(df_sum["mae"].tolist())
        per_cell[key]["rmse"].extend(df_sum["rmse"].tolist())

    return per_cell
# ============================================================
# 1) dataset 级：每个 dataset 一张箱线图，展示所有模型的指标
# ============================================================

def rename_model_for_plot(model: str) -> str:
    """统一论文中的模型名称显示."""
    if model == "TimeLLM_MLP":
        return "Ours"
    if model == "BiGRU_Transformer":
        return "BiGRU_Trans."
    # 去掉 Baseline / _Baseline / 前后空格
    model = model.replace("_Baseline", "").replace("Baseline", "").strip()
    return model


def plot_dataset_overall_box(df_exp: pd.DataFrame,
                             per_cell_metrics,
                             save_root: pathlib.Path):
    """
    对每个 dataset (data 列)：
    - 在同一张图中画所有模型基于 per-cell mae 的箱线图
    - 在同一张图中画所有模型基于 per-cell rmse 的箱线图
    """
    datasets = df_exp["data"].unique()

    # 想要的显示顺序（用 rename 后的名字来排）
    desired_order = ["Ours", "PINN", "LSTM",  
                     "Transformer", "CNN","BiGRU_Trans."]

    for data_name in datasets:
        sub = df_exp[df_exp["data"] == data_name].copy()
        if sub.empty:
            continue

        models_raw = sorted(sub["model"].unique())
        models_renamed = [rename_model_for_plot(m) for m in models_raw]

        # 构建 (原名 -> 显示名) 和 (显示名 -> 原名) 映射
        raw_to_show = dict(zip(models_raw, models_renamed))
        show_to_raw = {}
        for r, s in raw_to_show.items():
            # 如果重名，以第一次出现为准
            show_to_raw.setdefault(s, r)

        # 根据 desired_order 重排显示名，过滤掉本 dataset 中不存在的模型
        models = [m for m in desired_order if m in show_to_raw]

        mae_data = []
        rmse_data = []

        for show_name in models:
            raw_name = show_to_raw[show_name]
            key = (data_name, raw_name)
            if key not in per_cell_metrics:
                continue
            mae_vals = np.array(per_cell_metrics[key]["mae"], dtype=float)
            rmse_vals = np.array(per_cell_metrics[key]["rmse"], dtype=float)
            if len(mae_vals) == 0 or len(rmse_vals) == 0:
                continue
            mae_data.append(mae_vals)
            rmse_data.append(rmse_vals)

        if not mae_data:
            print(f"[WARN] No per-cell metrics for dataset {data_name}")
            continue

        dataset_dir = save_root / data_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        def _box_plot(data_list, ylabel, filename):
            plt.figure(figsize=(6, 4))
            positions = np.arange(1, len(models) + 1)
            bplot = plt.boxplot(
                data_list,
                positions=positions,
                widths=0.5,
                patch_artist=True,
                showfliers=False,
            )
            palette = [                
                "#4690cc", '#e09f20', '#7770c0', '#7f7f7f',
                "#03875de3","#a55059"]
            colors = [palette[i % len(palette)] for i in range(len(models))]
            for patch, c in zip(bplot["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.7)
            for median in bplot["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

            plt.xticks(positions, models, rotation=0, fontsize=13)
            plt.ylabel(ylabel, fontsize=13)
            plt.yticks(fontsize=13)

            # plt.title(f"{ylabel} across models")
            plt.grid(axis="y", linestyle="--", alpha=0.3)
            plt.tight_layout()
            out_path = dataset_dir / filename
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"[INFO] Saved: {out_path}")

        _box_plot(mae_data, "MAE", f"{data_name}_overall_MAE_box.png")
        _box_plot(rmse_data, "RMSE", f"{data_name}_overall_RMSE_box.png")

# ============================================================
# 2) cell 级：同一条退化曲线上，多模型 true/pred/err 对比
# ============================================================

def collect_cell_results_across_models(df_exp: pd.DataFrame):
    """
    返回一个嵌套字典：
    cell_results[(data, cell_name)][model] = results_csv_path

    data 来自 df_exp['data']
    cell_name 来自各 result_dir 下的 summary.csv 的 file_name（去掉 .csv）
    """
    cell_results = defaultdict(dict)  # {(data, cell_name): {model: csv_path}}

    for _, row in df_exp.iterrows():
        data_name = row["data"]
        model_name = row["model"]
        result_dir = pathlib.Path(row["result_dir"])
        summary_path = result_dir / "summary.csv"
        df_sum = safe_read_csv(summary_path)
        if df_sum is None or df_sum.empty:
            print(f"[WARN] summary.csv missing in {result_dir}")
            continue
        df_sum = df_sum.loc[:, ~df_sum.columns.str.contains("^Unnamed")]

        for _, r in df_sum.iterrows():
            file_name = r["file_name"]  # 如 XJTU_R2.5_battery-3.csv
            cell_name = file_name.replace(".csv", "")
            results_csv = result_dir / f"{cell_name}_results.csv"
            if results_csv.exists():
                cell_results[(data_name, cell_name)][model_name] = results_csv
            else:
                print(f"[WARN] {results_csv} not found for {data_name}-{cell_name}-{model_name}")

    return cell_results

def plot_scatter_true_vs_pred_per_dataset(df_exp: pd.DataFrame,
                                          per_cell_metrics,
                                          save_root: pathlib.Path):
    """
    对每个 dataset，汇总某个指定模型（一般是 Ours）的所有 cell 的 true/pred 点，
    绘制 True SOH vs Prediction 散点图 + y=x 红虚线。

    这里默认使用 Ours（TimeLLM_MLP）模型的结果。
    """
    target_raw_name = "TimeLLM_MLP"  # 我们的方法
    # 找出所有 dataset
    datasets = df_exp["data"].unique()

    for data_name in datasets:
        # 找到这个 dataset + Ours 对应的 result_dir
        sub = df_exp[(df_exp["data"] == data_name) &
                     (df_exp["model"] == target_raw_name)]
        if sub.empty:
            print(f"[WARN] No Ours result for dataset {data_name}")
            continue

        # 默认取第一条（一般只有一条）
        result_dir = pathlib.Path(sub.iloc[0]["result_dir"])
        # 汇总该 result_dir 下所有 *_results.csv
        all_results_files = sorted(result_dir.glob("*_results.csv"))
        if not all_results_files:
            print(f"[WARN] No *_results.csv in {result_dir} for scatter plot.")
            continue

        true_all = []
        pred_all = []

        for csv_path in all_results_files:
            df_res = safe_read_csv(csv_path)
            if df_res is None or df_res.empty:
                continue
            if not {"true_soh", "pred_soh"}.issubset(df_res.columns):
                continue
            # 有重复行可以直接用所有点
            true_all.append(df_res["true_soh"].values)
            pred_all.append(df_res["pred_soh"].values)

        if not true_all:
            print(f"[WARN] No valid true/pred for dataset {data_name}")
            continue

        true_all = np.concatenate(true_all)
        pred_all = np.concatenate(pred_all)

        # 绘图
        dataset_dir = save_root / data_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_path = dataset_dir / f"{data_name}_scatter_ours.png"

        plt.figure(figsize=(4, 4))
        plt.scatter(true_all, pred_all,
                    s=10, c="#4C72B0", alpha=0.7, edgecolors="none")

        # y=x 参考线
        vmin = min(true_all.min(), pred_all.min())
        vmax = max(true_all.max(), pred_all.max())
        margin = 0.02 * (vmax - vmin)
        x_line = np.linspace(vmin - margin, vmax + margin, 100)
        plt.plot(x_line, x_line, "r--", linewidth=1.5)

        plt.xlim(vmin - margin, vmax + margin)
        plt.ylim(vmin - margin, vmax + margin)
        plt.xlabel("True SOH")
        plt.ylabel("Prediction")
        plt.grid(True, linestyle="--", alpha=0.3)
        # 标题只写 batch / dataset 名
        plt.title(data_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved: {out_path}")

def plot_multi_model_curves_for_cell(
    data_name: str,
    cell_name: str,
    model_to_csv: dict,
    save_root: pathlib.Path,
):
    """
    输入：某个 dataset 下某一条退化曲线 (cell_name)，以及所有模型的 *_results.csv 路径
    输出：
    - 一张图：True SOH + 各模型 Pred SOH 曲线
    - 一张图：各模型 |error| 曲线
    保存到：test_results_new/{data_name}/{cell_name}/soh_multi_model.png 等
    """
    # 读取所有模型的结果，并对 index 对齐
    dfs = {}
    for model, csv_path in model_to_csv.items():
        df_res = safe_read_csv(csv_path)
        if df_res is None or df_res.empty:
            continue
        if not {"index", "true_soh", "pred_soh"}.issubset(df_res.columns):
            continue
        df_res = df_res.loc[:, ["index", "true_soh", "pred_soh"]]
        # 去重 index
        df_res = (
            df_res.groupby("index")[["true_soh", "pred_soh"]]
            .mean()
            .reset_index()
            .sort_values("index")
        )
        dfs[model] = df_res

    if not dfs:
        print(f"[WARN] No valid results for {data_name}-{cell_name}")
        return

    # 确定统一的 index
    all_idx = sorted(set(np.concatenate([df["index"].values for df in dfs.values()])))
    all_idx = np.array(all_idx)

    # 构造 true_soh（以第一个模型为准）
    first_model = list(dfs.keys())[0]
    df0 = dfs[first_model]
    # 用插值或 reindex，这里简单用 merge
    true_soh = np.interp(
        all_idx,
        df0["index"].values,
        df0["true_soh"].values
    )

    # 为各模型构造 pred 和 abs error
    model_pred = {}
    model_err = {}
    for model, dfm in dfs.items():
        pred = np.interp(
            all_idx,
            dfm["index"].values,
            dfm["pred_soh"].values
        )
        model_pred[rename_model_for_plot(model)] = pred
        model_err[rename_model_for_plot(model)] = np.abs(pred - true_soh)

    ymin = min(true_soh.min(), *(p.min() for p in model_pred.values()))
    ymax = max(true_soh.max(), *(p.max() for p in model_pred.values()))
    margin = max(0.01, 0.05 * (ymax - ymin))

    # 想要的显示顺序（显示名）
    desired_order = ["BiGRU_Trans.", "Transformer", "CNN",
                     "LSTM", "PINN", "Ours"]

    # 当前 cell 实际有哪些模型（model_pred 的 key 已经是重命名后的）
    available_models = set(model_pred.keys())
    models_sorted = [m for m in desired_order if m in available_models]

    palette = [                
        "#4690cc", '#e09f20', '#7770c0', '#7f7f7f',
        "#03875de3","#a55059"]
    colors = {m: palette[i % len(palette)] for i, m in enumerate(models_sorted)}

    cell_dir = save_root / data_name / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)

    # 1) SOH 多模型曲线
    plt.figure(figsize=(6, 4))
    plt.plot(all_idx, true_soh, label="True SOH", color="black", linewidth=1.3, linestyle='--')
    for m in models_sorted:
        plt.plot(
            all_idx,
            model_pred[m],
            label=m,
            color=colors[m],
            linewidth=1.2,
            linestyle='-',
            alpha=0.7,
        )
        if m == "Ours":
            plt.plot(
                all_idx,
                model_pred[m],
                label=None,
                color=colors[m],
                linewidth=2.0,
                linestyle='-',
                alpha=1.0,
            )
    plt.xlabel("Cycle index", fontsize=13)
    plt.ylabel("SOH", fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(ymin - margin, ymax + margin)
    # plt.grid(True, linestyle="--", alpha=0.3)
    # plt.title(f"{cell_name}")
    plt.legend(frameon=False)
    
    plt.tight_layout()
    out_path = cell_dir / "soh_multi_model.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {out_path}")

    # 2) 误差多模型曲线
    plt.figure(figsize=(6, 4))
    for m in models_sorted:
        plt.plot(
            all_idx,
            model_err[m],
            label=m,
            color=colors[m],
            linewidth=1.2,
        )
    plt.xlabel("Cycle index")
    plt.ylabel("Absolute error")
    # plt.grid(True, linestyle="--", alpha=0.3)
    plt.title(f"{cell_name}")
    plt.legend(frameon=False, fontsize=13)
    plt.tight_layout()
    out_path = cell_dir / "error_multi_model.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved: {out_path}")


# ============================================================
# 3) 主入口
# ============================================================

def main():
    if not EXP_CSV.exists():
        print(f"[ERROR] exp_results.csv not found at {EXP_CSV}")
        return

    df_exp = pd.read_csv(EXP_CSV)
    df_exp = df_exp.loc[:, ~df_exp.columns.str.contains("^Unnamed")]

    # 先从各 summary.csv 收集 per-cell 指标
    per_cell_metrics = collect_per_cell_metrics(df_exp)

    # (1) dataset 级：箱线图（每个 dataset 一张）
    plot_dataset_overall_box(
        df_exp,
        per_cell_metrics=per_cell_metrics,
        save_root=BASE_DIR / "test_results_new",
    )

    # (1.5) dataset 级：True vs Prediction 散点图（Ours）
    plot_scatter_true_vs_pred_per_dataset(
        df_exp,
        per_cell_metrics=per_cell_metrics,
        save_root=BASE_DIR / "test_results_new",
    )

    # (2) cell 级：同一退化曲线上，多模型 true/pred/error
    cell_results = collect_cell_results_across_models(df_exp)

    for (data_name, cell_name), model_to_csv in cell_results.items():
        print(f"[INFO] Plotting multi-model curves for {data_name} - {cell_name}")
        plot_multi_model_curves_for_cell(
            data_name=data_name,
            cell_name=cell_name,
            model_to_csv=model_to_csv,
            save_root=BASE_DIR / "test_results_new",
        )


if __name__ == "__main__":
    main()