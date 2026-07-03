import os
import re
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_RESULT_DIR = os.path.join(_PROJECT_ROOT, "results", "analogy")


MODELS = ["qwen-1.5b", "qwen-3b", "qwen-7b"]
MODEL_PRETTY = {
    "qwen-1.5b": "Qwen-2.5 1.5B",
    "qwen-3b": "Qwen-2.5 3B",
    "qwen-7b": "Qwen-2.5 7B",
}
COND_PRETTY = {"base": "Baseline", "meta": "ML (Meta-training)"}
COND_ORDER = ["base", "meta"]


def _fmt(mean: float, std: float) -> str:
    """Format 'mean ± std', or just 'mean' when std is undefined (single value)."""
    if pd.isna(mean):
        return "N/A"
    return f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f} ± {std:.2f}"


def _load_results(result_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(result_dir, "results.csv"))
    df["seed"] = df["seed"].astype(str)
    return df


def _parse_run_name(path: str):
    """Extract (model, ft_type, model_type, seed) from an errors/train_logs filename.

    e.g. 'qwen-3b_lora_base_seed_1048_eval_meta.json'
         'qwen-3b_lora_meta_seed_1048.csv'
    -> ('qwen-3b', 'lora', 'base'/'meta', '1048')
    """
    name = os.path.basename(path)
    m = re.match(r"(qwen-[\d.]+b)_(\w+?)_(base|meta)_seed_(\d+)", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def summary_table(result_dir: str = DEFAULT_RESULT_DIR) -> pd.DataFrame:
    df = _load_results(result_dir)

    rows = []
    for model in MODELS:
        stats = {}
        for cond in COND_ORDER:
            accs = df[(df["model"] == model) & (df["model_type"] == cond)]["accuracy"]
            stats[cond] = (accs.mean(), accs.std(), len(accs))
        base_m = stats["base"][0]
        meta_m = stats["meta"][0]
        rows.append({
            "model": model,
            "n_seeds": stats["base"][2],
            "baseline": _fmt(*stats["base"][:2]),
            "meta": _fmt(*stats["meta"][:2]),
            "delta": None if pd.isna(base_m) or pd.isna(meta_m) else round(meta_m - base_m, 2),
            "baseline_mean": base_m,
            "meta_mean": meta_m,
        })
    table = pd.DataFrame(rows)

    print(f"\nCore Accuracy Summary — Baseline vs Meta-training "
          f"(mean ± std over {table['n_seeds'].iloc[0]} seeds)")
    print("-" * 78)
    print(f"{'Model':<16}{'Baseline':<20}{'ML (Meta)':<20}{'Δ (ML − Base)':<15}")
    print("-" * 78)
    for _, r in table.iterrows():
        delta = "N/A" if r["delta"] is None else f"+{r['delta']:.2f}" if r["delta"] >= 0 else f"{r['delta']:.2f}"
        print(f"{MODEL_PRETTY[r['model']]:<16}{r['baseline']:<20}{r['meta']:<20}{delta:<15}")
    print("-" * 78)

    out = os.path.join(result_dir, "tables")
    os.makedirs(out, exist_ok=True)
    table.drop(columns=["baseline_mean", "meta_mean"]).to_csv(
        os.path.join(out, "summary_table.csv"), index=False
    )
    return table


def per_relation_table(result_dir: str = DEFAULT_RESULT_DIR) -> pd.DataFrame:
    err_dir = os.path.join(result_dir, "errors")
    records = []
    for path in glob.glob(os.path.join(err_dir, "*_eval_meta.json")):
        parsed = _parse_run_name(path)
        if parsed is None:
            continue
        model, _ft, cond, seed = parsed
        with open(path) as fh:
            data = json.load(fh)
        for rel, acc in data.get("per_relation_accuracy", {}).items():
            records.append({"model": model, "model_type": cond,
                            "seed": seed, "relation": rel, "accuracy": acc})

    if not records:
        print("\n[per_relation_table] No per-relation data found in errors/.")
        return pd.DataFrame()

    long = pd.DataFrame(records)
    agg = (long.groupby(["model", "model_type", "relation"])["accuracy"]
                .agg(["mean", "std"]).reset_index())

    relations = sorted(long["relation"].unique())
    print(f"\nPer-relation Accuracy — Baseline vs Meta-training (mean ± std over seeds)")
    print("-" * (28 + 22 * len(relations)))
    header = f"{'Model':<16}{'Method':<12}"
    for rel in relations:
        header += f"{rel[:20]:<22}"
    print(header)
    print("-" * (28 + 22 * len(relations)))
    for model in MODELS:
        for cond in COND_ORDER:
            line = f"{MODEL_PRETTY[model]:<16}{COND_PRETTY[cond].split(' ')[0]:<12}"
            for rel in relations:
                sub = agg[(agg["model"] == model) & (agg["model_type"] == cond)
                          & (agg["relation"] == rel)]
                if sub.empty:
                    line += f"{'N/A':<22}"
                else:
                    line += f"{_fmt(sub['mean'].iloc[0], sub['std'].iloc[0]):<22}"
            print(line)
    print("-" * (28 + 22 * len(relations)))

    out = os.path.join(result_dir, "tables")
    os.makedirs(out, exist_ok=True)
    agg.to_csv(os.path.join(out, "per_relation_table.csv"), index=False)

    _per_relation_heatmap(agg, relations, result_dir)
    return agg


def _per_relation_heatmap(agg: pd.DataFrame, relations, result_dir: str) -> None:
    pivot = agg.pivot_table(index=["model", "model_type"], columns="relation",
                            values="mean")
    delta_rows = {}
    for model in MODELS:
        try:
            base = pivot.loc[(model, "base")]
            meta = pivot.loc[(model, "meta")]
        except KeyError:
            continue
        delta_rows[MODEL_PRETTY[model]] = (meta - base)
    if not delta_rows:
        return
    delta = pd.DataFrame(delta_rows).T[relations]

    plt.figure(figsize=(1.6 * len(relations) + 3, 1.0 * len(delta) + 2))
    lim = np.nanmax(np.abs(delta.values)) if delta.size else 1
    sns.heatmap(delta, annot=True, fmt=".1f",
                cmap=sns.diverging_palette(240, 10, as_cmap=True),
                center=0, vmin=-lim, vmax=lim,
                cbar_kws={"label": "Δ Accuracy (%)  (ML − Baseline)"})
    plt.title("Per-relation improvement from Meta-training")
    plt.xlabel("Relation")
    plt.ylabel("Model")
    plt.tight_layout()
    plots = os.path.join(result_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    plt.savefig(os.path.join(plots, "per_relation_delta_heatmap.png"), dpi=200,
                bbox_inches="tight")
    plt.close()


def accuracy_barplot(result_dir: str = DEFAULT_RESULT_DIR) -> None:
    df = _load_results(result_dir)
    means = df.groupby(["model", "model_type"])["accuracy"].mean().unstack()
    stds = df.groupby(["model", "model_type"])["accuracy"].std().unstack()
    means = means.reindex(index=MODELS, columns=COND_ORDER)
    stds = stds.reindex(index=MODELS, columns=COND_ORDER)

    x = np.arange(len(MODELS))
    width = 0.36
    colors = {"base": "tab:blue", "meta": "tab:red"}

    plt.figure(figsize=(8, 5))
    for i, cond in enumerate(COND_ORDER):
        plt.bar(x + (i - 0.5) * width, means[cond], width,
                yerr=stds[cond], capsize=4, label=COND_PRETTY[cond],
                color=colors[cond], alpha=0.9)
    plt.xticks(x, [MODEL_PRETTY[m] for m in MODELS])
    plt.ylabel("Accuracy (%)")
    plt.title("Analogy accuracy: Baseline vs Meta-training")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plots = os.path.join(result_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    plt.savefig(os.path.join(plots, "accuracy_barplot.png"), dpi=200)
    plt.close()


def _seed_curves(df, log_dir, model, cond):
    """All per-seed validation curves for a (model, cond), as (progress[0..1], acc%) pairs."""
    curves = []
    seeds = sorted(df[(df["model"] == model) & (df["model_type"] == cond)]["seed"].unique())
    for seed in seeds:
        log_path = os.path.join(log_dir, f"{model}_lora_{cond}_seed_{seed}.csv")
        if not os.path.exists(log_path):
            continue
        log = pd.read_csv(log_path)
        if "eval/acc" not in log.columns or len(log) == 0:
            continue
        acc = log["eval/acc"].to_numpy(dtype=float)
        if np.nanmax(acc) <= 1.0:                    # stored as fraction -> percent
            acc = acc * 100.0
        n = len(acc)
        # Prefer a real training-progress column; fall back to even checkpoint spacing.
        xraw = None
        for cand in ("step", "global_step", "num_examples", "epoch"):
            if cand in log.columns:
                xraw = log[cand].to_numpy(dtype=float)
                break
        if xraw is None or n < 2 or xraw.max() == xraw.min():
            prog = np.linspace(0.0, 1.0, n)
        else:
            prog = (xraw - xraw.min()) / (xraw.max() - xraw.min())
        curves.append((prog, acc))
    return curves


def learning_curves(result_dir: str = DEFAULT_RESULT_DIR, n_grid: int = 100) -> None:
    log_dir = os.path.join(result_dir, "train_logs")
    df = _load_results(result_dir)
    grid = np.linspace(0.0, 1.0, n_grid)
    colors = {"base": "tab:blue", "meta": "tab:red"}

    # Aggregate to mean +/- std over seeds on a common progress grid.
    agg, ymax = {}, 0.0
    for model in MODELS:
        for cond in COND_ORDER:
            curves = _seed_curves(df, log_dir, model, cond)
            if not curves:
                continue
            stacked = np.vstack([np.interp(grid, p, a) for p, a in curves])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            agg[(model, cond)] = (mean, std, len(curves))
            ymax = max(ymax, float(np.nanmax(mean + std)))

    # One row of panels, shared y-axis for honest cross-size comparison.
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, MODELS):
        for cond in COND_ORDER:
            if (model, cond) not in agg:
                continue
            mean, std, k = agg[(model, cond)]
            x = grid * 100.0
            ax.plot(x, mean, color=colors[cond], lw=2,
                    label=f"{COND_PRETTY[cond]} (n={k})")
            ax.fill_between(x, mean - std, mean + std, color=colors[cond],
                            alpha=0.18, linewidth=0)
        ax.set_title(MODEL_PRETTY[model])
        ax.set_xlabel("Training progress (%)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].set_ylim(0, min(100, ymax * 1.05))
    axes[0].legend(loc="lower right", frameon=True)
    fig.suptitle("Learning curves — mean ± std over seeds", y=1.02, fontsize=13)
    fig.tight_layout()

    plots = os.path.join(result_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    fig.savefig(os.path.join(plots, "learning_curves.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def error_analysis(result_dir: str = DEFAULT_RESULT_DIR) -> pd.DataFrame:
    err_dir = os.path.join(result_dir, "errors")
    records = []
    for path in glob.glob(os.path.join(err_dir, "*_eval_meta.json")):
        parsed = _parse_run_name(path)
        if parsed is None:
            continue
        model, _ft, cond, seed = parsed
        with open(path) as fh:
            data = json.load(fh)
        errors = data.get("errors", [])
        by_rel = {}
        for e in errors:
            by_rel[e.get("relation", "?")] = by_rel.get(e.get("relation", "?"), 0) + 1
        records.append({
            "model": model, "model_type": cond, "seed": seed,
            "overall_accuracy": data.get("overall_accuracy"),
            "n_errors": data.get("n_errors", len(errors)),
        })

    if not records:
        print("\n[error_analysis] No error files found.")
        return pd.DataFrame()

    err = pd.DataFrame(records)
    agg = (err.groupby(["model", "model_type"])
              .agg(mean_errors=("n_errors", "mean"),
                   mean_acc=("overall_accuracy", "mean"))
              .reset_index())

    print("\nError counts — mean number of wrong predictions per run")
    print("-" * 60)
    print(f"{'Model':<16}{'Method':<14}{'Avg # errors':<14}{'Avg acc':<10}")
    print("-" * 60)
    for model in MODELS:
        for cond in COND_ORDER:
            sub = agg[(agg["model"] == model) & (agg["model_type"] == cond)]
            if sub.empty:
                continue
            print(f"{MODEL_PRETTY[model]:<16}{COND_PRETTY[cond].split(' ')[0]:<14}"
                  f"{sub['mean_errors'].iloc[0]:<14.1f}{sub['mean_acc'].iloc[0]:<10.2f}")
    print("-" * 60)

    out = os.path.join(result_dir, "tables")
    os.makedirs(out, exist_ok=True)
    agg.to_csv(os.path.join(out, "error_analysis.csv"), index=False)
    return agg


def _render_table_image(df: pd.DataFrame, path: str, title: str = None,
                        highlight_last_col: bool = False,
                        header_color: str = "#2f4b7c",
                        row_colors=("#ffffff", "#eef2f8"),
                        dpi: int = 200) -> None:
    """Render a DataFrame as a clean table image (header band, zebra rows)."""
    df = df.astype(str)
    n_rows, n_cols = df.shape

    content_len = [max([len(c)] + [len(v) for v in df[c].values]) for c in df.columns]
    total = sum(content_len)
    col_widths = [w / total for w in content_len]

    fig_w = min(max(sum(content_len) * 0.135, 5), 18)
    fig_h = 0.55 + 0.45 * (n_rows + 1) + (0.5 if title else 0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   colWidths=col_widths, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#c8d0dc")
        cell.set_linewidth(0.7)
        if r == 0: 
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(row_colors[(r - 1) % 2])
            if highlight_last_col and c == n_cols - 1:
                cell.set_text_props(color="#0a7d2c", fontweight="bold")

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def tables_to_images(result_dir: str = DEFAULT_RESULT_DIR) -> None:
    """Read the CSVs written by the table functions and render them as PNGs."""
    tdir = os.path.join(result_dir, "tables")
    pdir = os.path.join(result_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    spath = os.path.join(tdir, "summary_table.csv")
    if os.path.exists(spath):
        s = pd.read_csv(spath)
        disp = pd.DataFrame({
            "Model": s["model"].map(MODEL_PRETTY).fillna(s["model"]),
            "Baseline": s["baseline"],
            "ML (Meta)": s["meta"],
            "Δ (ML − Base)": s["delta"].map(
                lambda d: f"+{d:.2f}" if float(d) >= 0 else f"{d:.2f}"),
        })
        _render_table_image(
            disp, os.path.join(pdir, "summary_table.png"),
            title="Core Accuracy — Baseline vs Meta-training  (mean ± std, 5 seeds)",
            highlight_last_col=True)

    ppath = os.path.join(tdir, "per_relation_table.csv")
    if os.path.exists(ppath):
        pr = pd.read_csv(ppath)
        pr["cell"] = pr.apply(
            lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}"
            if pd.notna(r["std"]) else f"{r['mean']:.2f}", axis=1)
        wide = pr.pivot_table(index=["model", "model_type"], columns="relation",
                              values="cell", aggfunc="first").reset_index()
        wide["_m"] = wide["model"].map({m: i for i, m in enumerate(MODELS)})
        wide["_c"] = wide["model_type"].map({"base": 0, "meta": 1})
        wide = wide.sort_values(["_m", "_c"]).drop(columns=["_m", "_c"])
        wide.insert(0, "Model", wide.pop("model").map(MODEL_PRETTY))
        wide.insert(1, "Method", wide.pop("model_type").map(
            {"base": "Baseline", "meta": "ML"}))
        _render_table_image(
            wide, os.path.join(pdir, "per_relation_table.png"),
            title="Per-relation Accuracy — Baseline vs Meta-training (mean ± std)")

    epath = os.path.join(tdir, "error_analysis.csv")
    if os.path.exists(epath):
        e = pd.read_csv(epath)
        e["_m"] = e["model"].map({m: i for i, m in enumerate(MODELS)})
        e["_c"] = e["model_type"].map({"base": 0, "meta": 1})
        e = e.sort_values(["_m", "_c"])
        disp = pd.DataFrame({
            "Model": e["model"].map(MODEL_PRETTY),
            "Method": e["model_type"].map({"base": "Baseline", "meta": "ML"}),
            "Avg # errors": e["mean_errors"].map(lambda v: f"{v:.1f}"),
            "Avg accuracy (%)": e["mean_acc"].map(lambda v: f"{v:.2f}"),
        })
        _render_table_image(
            disp, os.path.join(pdir, "error_analysis_table.png"),
            title="Error Counts — Baseline vs Meta-training")


if __name__ == "__main__":
    RESULT_DIR = DEFAULT_RESULT_DIR
    os.makedirs(os.path.join(RESULT_DIR, "plots"), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, "tables"), exist_ok=True)

    summary_table(RESULT_DIR)
    per_relation_table(RESULT_DIR)
    error_analysis(RESULT_DIR)
    accuracy_barplot(RESULT_DIR)
    learning_curves(RESULT_DIR)
    tables_to_images(RESULT_DIR)  

    print(f"\nPlots written to  {RESULT_DIR}/plots/")
    print(f"Tables written to {RESULT_DIR}/tables/")
