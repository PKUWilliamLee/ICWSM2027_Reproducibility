from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMP = ROOT / "outputs" / "derived" / "empirical"
SIM = ROOT / "outputs" / "derived" / "simulation"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Tinos", "Liberation Serif", "serif"],
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9.2,
    "ytick.labelsize": 9.2,
    "legend.fontsize": 8.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {"full": "#1f77b4", "info": "#e67e22", "state": "#2ca02c", "ablation": "#d62728"}


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def annual_theme_plots() -> None:
    df = pd.read_csv(EMP / "annual_primary_theme_shares.csv")
    themes = [
        ("Schoolwork Burden", "fig1a_1_schoolwork_burden"),
        ("Shadow Education", "fig1a_2_shadow_education"),
        ("Refund Disputes", "fig1a_3_refund_disputes"),
        ("After-school Services", "fig1a_4_after_school_services"),
        ("Family Pressure", "fig1a_5_family_pressure"),
        ("Student Well-being", "fig1a_6_student_well_being"),
        ("Admissions Equity", "fig1a_7_admissions_equity"),
    ]
    for theme, stem in themes:
        fig, ax = plt.subplots(figsize=(2.15, 1.55))
        ax.plot(df["year"], df[theme] * 100.0, linewidth=1.2)
        ax.axvline(2021.56, linestyle="--", linewidth=0.8, color="#555555")
        ax.set_title(theme, fontsize=9.2)
        ax.set_ylabel("Share (%)")
        ax.set_xlim(df["year"].min(), df["year"].max())
        ax.set_xticks([2011, 2016, 2021, 2025])
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.45)
        save(fig, stem)


def cooccurrence() -> None:
    df = pd.read_csv(EMP / "cooccurrence_selected_edges.csv")
    df = df.sort_values(["post_share", "pre_share"], ascending=False).reset_index(drop=True)
    y = np.arange(len(df))[::-1]
    fig = plt.figure(figsize=(6.95, 3.08))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.55, 3.95, 0.78], wspace=0.02)
    ax_label = fig.add_subplot(gs[0, 0]); ax = fig.add_subplot(gs[0, 1]); ax_delta = fig.add_subplot(gs[0, 2], sharey=ax)
    for yi, row in zip(y, df.itertuples(index=False)):
        ax.axhline(yi, color="#dddddd", linewidth=0.45, zorder=0)
        ax.plot([row.pre_share, row.post_share], [yi, yi], color="#555555", linewidth=0.9, zorder=1)
    ax.scatter(df["pre_share"], y, marker="o", s=34, color=COLORS["full"], label="Pre-policy", zorder=3)
    ax.scatter(df["post_share"], y, marker="s", s=34, color=COLORS["info"], label="Post-policy", zorder=3)
    ax.set_ylim(-0.6, len(df)-0.4); ax.set_xlim(0, max(df["pre_share"].max(), df["post_share"].max()) + 1.8)
    ax.set_xlabel("Share among two-theme messages (%)"); ax.set_yticks([]); ax.grid(axis="x", color="#e5e5e5", linewidth=0.45)
    ax.legend(loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.005))
    ax_label.set_xlim(0, 1); ax_label.set_ylim(ax.get_ylim()); ax_label.axis("off")
    for yi, edge in zip(y, df["edge"]): ax_label.text(0.99, yi, edge, ha="right", va="center", fontsize=9.1)
    ax_delta.set_xlim(0, 1); ax_delta.set_ylim(ax.get_ylim()); ax_delta.axis("off")
    ax_delta.text(0.05, len(df)-0.15, r"$\Delta$", ha="left", va="bottom", fontweight="bold")
    for yi, ch in zip(y, df["change_pp"]): ax_delta.text(0.05, yi, f"{ch:+.1f} pp", ha="left", va="center")
    save(fig, "fig1b_cooccurrence_dumbbell")


def itsa() -> None:
    long = pd.read_csv(EMP / "itsa_counterfactual_effects_long.csv")
    order = ["Schoolwork Burden", "Shadow Education", "Refund Disputes", "After-school Services", "Family Pressure", "Student Well-being", "Admissions Equity"]
    ybase = np.arange(len(order))[::-1]; ymap = dict(zip(order, ybase))
    offsets = {0: 0.27, 12: 0.09, 24: -0.09, 36: -0.27}; markers = {0: "o", 12: "s", 24: "^", 36: "D"}
    colors = {0: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c", 36: "#d62728"}
    fig, ax = plt.subplots(figsize=(6.85, 3.32))
    for horizon in [0, 12, 24, 36]:
        sub = long[long["horizon"].eq(horizon)].set_index("Theme").loc[order].reset_index()
        yy = np.array([ymap[t] for t in sub["Theme"]], float) + offsets[horizon]
        xerr = np.vstack([sub["estimate"] - sub["low"], sub["high"] - sub["estimate"]])
        ax.errorbar(sub["estimate"], yy, xerr=xerr, fmt=markers[horizon], color=colors[horizon], markersize=4.8, linewidth=0.95, capsize=2.2, label=f"{horizon} months")
    ax.axvline(0, linestyle="--", linewidth=0.85, color="#333333")
    ax.set_yticks(ybase, order); ax.set_xlabel("Fitted difference from continued pre-policy trajectory (percentage points)")
    ax.set_xlim(-36, 18); ax.grid(axis="x", color="#e5e5e5", linewidth=0.45)
    ax.legend(loc="lower left", frameon=False, ncol=4, bbox_to_anchor=(0.0, 1.005))
    save(fig, "fig2_itsa_effects")


def mechanism() -> None:
    h = pd.read_csv(SIM / "p3_information_state_by_horizon.csv")
    h["series"] = h["series"].replace({"Full policy": "Combined measures", "Policy information": "Policy information", "Implementation state": "Added implementation conditions"})
    style = {"Combined measures": (COLORS["full"], "-", "o"), "Policy information": (COLORS["info"], "--", "s"), "Added implementation conditions": (COLORS["state"], "-.", "^")}
    fig, ax = plt.subplots(figsize=(3.34, 2.48))
    for series, (color, ls, marker) in style.items():
        sub = h[h["series"].eq(series)].sort_values("horizon_months")
        ax.plot(sub["horizon_months"], sub["effect_magnitude"], color=color, linestyle=ls, marker=marker, markersize=4.8, linewidth=1.25, label=series)
    ax.set_xticks([0, 6, 12, 24, 36]); ax.set_xlabel("Months since August 2021"); ax.set_ylabel("Semantic-change magnitude"); ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.45); ax.legend(frameon=False, loc="best", fontsize=8.0)
    save(fig, "fig3a_information_state")

    comp = pd.read_csv(SIM / "p3_component_ratios_long.csv")
    order = ["A", "B", "C"]; labels = {"A": "Homework\nregulation", "B": "Tutoring\nregulation", "C": "After-school\nservices"}
    ybase = np.arange(3)[::-1]; ymap = dict(zip(order, ybase))
    styles = {"Standalone magnitude": (0.24, "o", COLORS["full"]), "State-ablation magnitude": (0.08, "s", COLORS["info"]), "Standalone projection": (-0.08, "^", COLORS["state"]), "State-ablation projection": (-0.24, "D", COLORS["ablation"])}
    names = {"Standalone magnitude": "Single-measure magnitude", "State-ablation magnitude": "Unchanged-condition magnitude", "Standalone projection": "Single-measure projection", "State-ablation projection": "Unchanged-condition projection"}
    fig, ax = plt.subplots(figsize=(4.30, 3.0))
    for measure, (offset, marker, color) in styles.items():
        sub = comp[comp["measure"].eq(measure)].set_index("component").loc[order].reset_index()
        yy = np.array([ymap[c] for c in sub["component"]], float) + offset
        ax.scatter(sub["value"], yy, marker=marker, s=32, color=color, label=names[measure], zorder=3)
    ax.axvline(1.0, linestyle="--", linewidth=0.85, color="#333333")
    ax.set_yticks(ybase, [labels[c] for c in order]); ax.set_xlabel("Ratio to combined-measures benchmark"); ax.set_xlim(-0.03, 1.22)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.45); ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, columnspacing=0.5, handletextpad=0.2, fontsize=7.2)
    save(fig, "fig3b_component_ratios")


def main() -> None:
    annual_theme_plots(); cooccurrence(); itsa(); mechanism()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()
