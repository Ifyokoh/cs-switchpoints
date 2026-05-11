"""Exploratory data analysis of code-switching pairs.

Reads data/switchpairs/*_pairs.tsv and writes plots (SVG) and frequency
tables (CSV) to outputs/. Summary statistics are written to
outputs/eda_summary.txt.

Multi-dataset plots are combined into a single SVG with one subplot per
dataset so language pairs can be compared side by side. POS heatmaps are
kept as separate SVGs per dataset because each has a different tag set and
scale.

Sections:
    1. Sentence length distribution and relationship to switch rate
    2. Within/boundary split — switch vs same-language pairs (core result)
    3. POS pair frequencies at switch points
    4. Dependency role frequencies at switch points

Run:
    python scripts/eda.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS = ["hi_en", "te_en", "tr_en"]

DATASET_LABELS = {
    "hi_en": "Hindi–English",
    "te_en": "Telugu–English",
    "tr_en": "Turkish–English",
}


def load_pairs(filepath: Path) -> pd.DataFrame:
    """Load a switchpairs TSV file into a DataFrame.
    Args:
        filepath: Path to a *_pairs.tsv file.
    Returns:
        DataFrame with string-typed switch and relation columns, and
        integer tok_i_id and tok_j_id columns.
    """
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    df["tok_i_id"] = pd.to_numeric(df["tok_i_id"])
    df["tok_j_id"] = pd.to_numeric(df["tok_j_id"])
    return df


def section_1_sentence_length(datasets: dict[str, pd.DataFrame], outputs_dir: Path) -> str:
    """Analyse sentence length distribution and its relationship to switch rate.

    Sentence length is measured as the number of adjacent pairs per sentence.
    Longer sentences have more pair positions, so raw switch counts are not
    comparable across lengths. This section checks whether the switch *rate*
    (switches / total pairs) also changes with sentence length.

    Produces:
        sentence_length_distribution.svg — combined histogram (3 subplots).

    Args:
        datasets: Mapping from dataset prefix to full DataFrame.
        outputs_dir: Directory for output SVGs.

    Returns:
        Summary string for eda_summary.txt.
    """
    def per_sentence_stats(group: pd.DataFrame) -> pd.Series:
        total = len(group)
        switch_count = (group["switch"] == "true").sum()
        return pd.Series({
            "n_pairs": total,
            "switch_count": switch_count,
            "switch_rate": switch_count / total,
        })

    all_stats: dict[str, pd.DataFrame] = {}
    for prefix, df in datasets.items():
        all_stats[prefix] = df.groupby("sent_id").apply(per_sentence_stats).reset_index()

    n = len(datasets)

    # combined histogram of sentence lengths
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (prefix, stats) in zip(axes, all_stats.items()):
        label = DATASET_LABELS[prefix]
        max_len = int(stats["n_pairs"].max())
        ax.hist(stats["n_pairs"], bins=range(1, max_len + 2), color="steelblue", edgecolor="white")
        ax.set_xlabel("Sentence length (adjacent pairs)")
        ax.set_ylabel("Number of sentences")
        ax.set_title(label)
    fig.suptitle("Sentence length distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(outputs_dir / "sentence_length_distribution.svg")
    plt.close(fig)

    lines = ["\n=== Section 1: Sentence length ==="]
    for prefix, stats in all_stats.items():
        label = DATASET_LABELS[prefix]
        correlation = stats["n_pairs"].corr(stats["switch_rate"])
        lines.extend([
            f"\n  [{label}]",
            f"  Sentences                     : {len(stats):,}",
            f"  Min length (pairs)            : {int(stats['n_pairs'].min())}",
            f"  Max length (pairs)            : {int(stats['n_pairs'].max())}",
            f"  Mean length (pairs)           : {stats['n_pairs'].mean():.1f}",
            f"  Median length (pairs)         : {stats['n_pairs'].median():.1f}",
            f"  Sentences with 0 switches     : {int((stats['switch_count'] == 0).sum()):,}",
            f"  Mean switch rate/sentence     : {stats['switch_rate'].mean():.3f}",
            f"  Corr(length, switch rate)     : {correlation:.3f}",
        ])
    return "\n".join(lines)


def section_2_within_boundary_chart(datasets: dict[str, pd.DataFrame], outputs_dir: Path) -> str:
    """Plot within/boundary proportions for switch vs same-language pairs.

    This is the core descriptive result. A grouped bar chart per dataset
    compares the within/boundary split for code-switch pairs against
    same-language pairs, showing whether switches are over- or under-represented
    at syntactic boundaries relative to the baseline.

    Produces:
        within_boundary_by_switch.svg — combined grouped bar chart (3 subplots).

    Args:
        datasets: Mapping from dataset prefix to full DataFrame.
        outputs_dir: Directory for output SVG.

    Returns:
        Summary string for eda_summary.txt.
    """
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    lines = ["\n=== Section 2: Within/boundary split by switch status ==="]

    for ax, (prefix, df) in zip(axes, datasets.items()):
        label = DATASET_LABELS[prefix]
        filtered = df[(df["switch"] != "unknown") & (df["relation"] != "unknown")]

        rows = []
        for switch_val, group_label in [("true", "Switch"), ("false", "Same-lang")]:
            group = filtered[filtered["switch"] == switch_val]
            total = len(group)
            if total == 0:
                continue
            within_pct = 100 * (group["relation"] == "within").sum() / total
            boundary_pct = 100 * (group["relation"] == "boundary").sum() / total
            rows.append((group_label, within_pct, boundary_pct, total))

        group_labels = [r[0] for r in rows]
        within_pcts = [r[1] for r in rows]
        boundary_pcts = [r[2] for r in rows]

        x = range(len(group_labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], within_pcts, width, label="within", color="steelblue")
        ax.bar([i + width / 2 for i in x], boundary_pcts, width, label="boundary", color="coral")
        ax.set_xticks(list(x))
        ax.set_xticklabels(group_labels)
        ax.set_title(label)
        ax.legend(fontsize=8)

        if ax is axes[0]:
            ax.set_ylabel("Percentage (%)")

        for group_label, within_pct, boundary_pct, total in rows:
            lines.append(
                f"  {label} — {group_label} (n={total:,}): "
                f"within={within_pct:.1f}%  boundary={boundary_pct:.1f}%"
            )

    fig.suptitle("Within vs boundary: switch pairs vs same-language pairs", fontsize=13)
    fig.tight_layout()
    fig.savefig(outputs_dir / "within_boundary_by_switch.svg")
    plt.close(fig)

    return "\n".join(lines)


def section_3_pos_pairs(datasets: dict[str, pd.DataFrame], outputs_dir: Path) -> str:
    """Analyse POS pair frequencies at code-switch points.

    Identifies which UPOS combinations (left token → right token) are most
    common when a switch occurs. A combined top-10 bar chart allows cross-dataset
    comparison. Individual heatmaps per dataset show the full tag-pair matrix.

    Produces:
        pos_top10.svg — combined top-10 bar chart (3 subplots).

    Args:
        datasets: Mapping from dataset prefix to full DataFrame.
        outputs_dir: Directory for output files.

    Returns:
        Summary string for eda_summary.txt.
    """
    n = len(datasets)
    lines = ["\n=== Section 3: POS pairs at switch points ==="]

    all_pos_counts: dict[str, pd.DataFrame] = {}
    all_switch_dfs: dict[str, pd.DataFrame] = {}

    for prefix, df in datasets.items():
        switch_df = df[df["switch"] == "true"].copy()
        pos_counts = (
            switch_df
            .groupby(["upos_i", "upos_j"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        all_pos_counts[prefix] = pos_counts
        all_switch_dfs[prefix] = switch_df

    # combined top-10 bar chart
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (prefix, pos_counts) in zip(axes, all_pos_counts.items()):
        label = DATASET_LABELS[prefix]
        switch_df = all_switch_dfs[prefix]
        top10 = pos_counts.head(10).copy()
        top10["pair"] = top10["upos_i"] + " → " + top10["upos_j"]
        ax.barh(top10["pair"][::-1], top10["count"][::-1], color="steelblue")
        ax.set_xlabel("Number of switch pairs")
        ax.set_title(f"{label}\n(n={len(switch_df):,} switches)")
    fig.suptitle("Top-10 POS pairs at switch points", fontsize=13)
    fig.tight_layout()
    fig.savefig(outputs_dir / "pos_top10.svg")
    plt.close(fig)

    for prefix, pos_counts in all_pos_counts.items():
        label = DATASET_LABELS[prefix]
        switch_df = all_switch_dfs[prefix]
        top5 = pos_counts.head(5)
        lines.append(f"\n  [{label}] — total switch pairs: {len(switch_df):,}")
        lines.append("  Top 5 POS pairs:")
        for _, row in top5.iterrows():
            lines.append(f"    {row['upos_i']} → {row['upos_j']}: {int(row['count']):,}")

    return "\n".join(lines)


def section_4_deprel(datasets: dict[str, pd.DataFrame], outputs_dir: Path) -> str:
    """Analyse dependency relation frequencies at switch points.

    Shows which syntactic roles appear most at switch points and whether each
    role is associated more with within-constituent or boundary switches.

    Produces:
        deprel_by_relation.svg — combined within vs boundary by deprel_i (3 subplots).

    Args:
        datasets: Mapping from dataset prefix to full DataFrame.
        outputs_dir: Directory for output files.

    Returns:
        Summary string for eda_summary.txt.
    """
    n = len(datasets)
    lines = ["\n=== Section 4: Dependency roles at switch points ==="]

    all_comparisons: dict[str, pd.DataFrame] = {}
    all_switch_dfs: dict[str, pd.DataFrame] = {}

    for prefix, df in datasets.items():
        switch_df = df[(df["switch"] == "true") & (df["relation"] != "unknown")].copy()
        all_switch_dfs[prefix] = switch_df

        deprel_counts = (
            switch_df
            .groupby("deprel_i")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
)
        within_counts = (
            switch_df[switch_df["relation"] == "within"]
            .groupby("deprel_i")
            .size()
            .rename("within")
        )
        boundary_counts = (
            switch_df[switch_df["relation"] == "boundary"]
            .groupby("deprel_i")
            .size()
            .rename("boundary")
        )
        comparison = (
            pd.concat([within_counts, boundary_counts], axis=1)
            .fillna(0)
            .astype(int)
        )
        comparison["total"] = comparison.sum(axis=1)
        comparison = comparison.sort_values("total", ascending=False).head(10)
        comparison = comparison.drop(columns="total")
        all_comparisons[prefix] = comparison

    # combined deprel by-relation chart
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (prefix, comparison) in zip(axes, all_comparisons.items()):
        label = DATASET_LABELS[prefix]
        switch_df = all_switch_dfs[prefix]
        comparison.plot(
            kind="barh",
            ax=ax,
            color=["steelblue", "coral"],
            legend=(ax is axes[0]),
        )
        ax.set_xlabel("Number of switch pairs")
        ax.set_title(f"{label}\n(n={len(switch_df):,} switches)")
        ax.invert_yaxis()
    fig.suptitle("Dependency role (deprel_i) at switch points — within vs boundary", fontsize=13)
    fig.tight_layout()
    fig.savefig(outputs_dir / "deprel_by_relation.svg")
    plt.close(fig)

    for prefix, switch_df in all_switch_dfs.items():
        label = DATASET_LABELS[prefix]
        comparison = all_comparisons[prefix]
        lines.append(
            f"\n  [{label}] — total switch pairs (relation known): {len(switch_df):,}"
        )
        lines.append("  Top 5 deprel_i at switch points:")
        for deprel, row in comparison.head(5).iterrows():
            within = int(row.get("within", 0))
            boundary = int(row.get("boundary", 0))
            total = within + boundary
            lines.append(
                f"    {deprel}: total={total}  within={within}  boundary={boundary}"
            )

    return "\n".join(lines)


def main() -> None:
    """Run all EDA sections and write outputs to the outputs/ directory."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--switchpairs-dir",
        type=Path,
        default=repo_root / "data" / "switchpairs",
        help="Directory containing the *_pairs.tsv files.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory for plots and tables.",
    )
    args = parser.parse_args()

    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # load all datasets
    datasets: dict[str, pd.DataFrame] = {}
    for prefix in DATASETS:
        filepath = args.switchpairs_dir / f"{prefix}_pairs.tsv"
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.", file=sys.stderr)
            continue
        datasets[prefix] = load_pairs(filepath)
        print(f"Loaded {prefix}: {len(datasets[prefix]):,} pairs")

    if not datasets:
        print("No datasets found. Exiting.", file=sys.stderr)
        sys.exit(1)

    summary_lines = ["EDA Summary", "=" * 60]

    print("\nSection 1: sentence length...")
    summary_lines.append(section_1_sentence_length(datasets, args.outputs_dir))

    print("Section 2: within/boundary chart...")
    summary_lines.append(section_2_within_boundary_chart(datasets, args.outputs_dir))

    print("Section 3: POS pairs...")
    summary_lines.append(section_3_pos_pairs(datasets, args.outputs_dir))

    print("Section 4: deprel...")
    summary_lines.append(section_4_deprel(datasets, args.outputs_dir))

    summary_path = args.outputs_dir / "eda_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\nSummary written to {summary_path}")
    print(f"Plots and tables saved to {args.outputs_dir}/")


if __name__ == "__main__":
    main()
