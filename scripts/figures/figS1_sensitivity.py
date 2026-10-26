#!/usr/bin/env python3
"""What each part of the method is actually worth, on the same dated corpus.

Three quantities have to be read together. Removing a capability rarely loses
the published year outright, because the safety rules are built to avoid that;
what it does is widen the set the method must report. A configuration that
looks sharper in the middle panel and worse in the right-hand one is not an
improvement, it is a method throwing the answer away.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

import figure_style as fs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT))

from tools import years as yr  # noqa: E402

# Printed name and group for every configuration the audit runs, in the order
# they are drawn.
CONFIGS = [
    ("baseline", "Baseline", "Baseline"),
    ("cubit_at_2_2_degrees", "Cubit at 2.2 degrees", "Conventions varied"),
    ("delta_t_applied", "Fixed Delta T shift", "Conventions varied"),
    ("britton_offset_minus_half", "Zodiac 0.5 degrees lower",
     "Conventions varied"),
    ("britton_offset_plus_half", "Zodiac 0.5 degrees higher",
     "Conventions varied"),
    ("tolerance_narrow", "Narrow distance tolerance", "Conventions varied"),
    ("tolerance_wide", "Wide distance tolerance", "Conventions varied"),
    ("moon_night_window", "Moon tested over two nights", "Conventions varied"),
    ("no_mars_family", "No Mars", "Physical evidence removed"),
    ("no_jupiter_family", "No Jupiter", "Physical evidence removed"),
    ("no_saturn_family", "No Saturn", "Physical evidence removed"),
    ("no_lunar_eclipse_family", "No lunar eclipses",
     "Physical evidence removed"),
    ("no_sun_family", "No solar-year clock",
     "Physical evidence removed"),
    ("no_sirius_scheme", "No Sirius dates", "Physical evidence removed"),
    ("no_inferred_summary_month", "No inferred summary month",
     "Reading rules removed"),
    ("no_restored_month_context", "No restored month context",
     "Reading rules removed"),
    ("no_extended_month_heading", "No added month-heading forms",
     "Reading rules removed"),
    ("no_restoration_reading", "No reading across brackets",
     "Reading rules removed"),
    ("no_first_appearance", "No first-appearance rule",
     "Reading rules removed"),
    ("no_acronychal_rising", "No acronychal risings", "Reading rules removed"),
    ("no_scorpius_alias", "No Scorpius as Scorpio", "Reading rules removed"),
    ("no_agent_suggested_rules", "No rules from the coding agent",
     "Reading rules removed"),
    ("no_station_relative_to_star", "No station-against-star rule",
     "Reading rules removed"),
    ("no_planet_to_planet_distance", "No planet-against-planet rule",
     "Reading rules removed"),
    ("no_month_context", "No month context at all", "Calendar context removed"),
    ("contradictions_before_information", "Contradictions before bits",
     "Ordering and strict set changed"),
    ("contradictions_bound_the_leading_set",
     "Contradictions shrink the strict set", "Ordering and strict set changed"),
    ("independent_joint_atoms", "Joint positions split", "Scoring changed"),
    ("count_weighting", "Count every clause equally", "Scoring changed"),
    ("equal_weighting_with_tiers", "Equal weights, tiers kept",
     "Scoring changed"),
    ("information_weighting", "Bits without tiers", "Scoring changed"),
    ("longitude_only_distances_supporting",
     "Longitude-only distances supporting", "Scoring changed"),
]
GROUP_COLOUR = {
    "Baseline": fs.ORANGE,
    "Conventions varied": fs.GREY,
    "Physical evidence removed": fs.BLUE,
    "Reading rules removed": fs.GREEN,
    "Calendar context removed": fs.GREEN,
    "Ordering and strict set changed": fs.INK,
    "Scoring changed": fs.INK,
}


def validate_configurations(summary: pd.DataFrame) -> None:
    """Refuse to silently omit a completed sensitivity configuration."""
    expected = {key for key, _, _ in CONFIGS}
    actual = set(summary.index)
    missing = expected - actual
    unexpected = actual - expected
    if not summary.index.is_unique or missing or unexpected:
        raise ValueError(
            "Sensitivity configurations and figure labels do not match. "
            f"Missing from data: {sorted(missing)}; "
            f"missing from figure: {sorted(unexpected)}; "
            f"duplicate data labels: {not summary.index.is_unique}. "
            "Use the completed sensitivity audit and label every configuration."
        )


def layout():
    """Row positions, with an italic header row opening each group."""
    rows, ticks, labels, colours, weights = [], [], [], [], []
    y, seen = 0.0, None
    for key, label, group in CONFIGS:
        if group != seen and group != "Baseline":
            ticks.append(y)
            labels.append(group)
            colours.append(GROUP_COLOUR[group])
            weights.append("italic")
            y += 0.92
        rows.append((key, group, y))
        ticks.append(y)
        labels.append(label)
        colours.append(GROUP_COLOUR[group])
        weights.append("bold" if group == "Baseline" else "normal")
        y += 1.0
        seen = group
    return rows, ticks, labels, colours, weights, y


def panel(ax, rows, values, baseline, xlabel, left, height):
    ax.axvline(baseline, color=fs.ORANGE, linewidth=0.8,
               linestyle=(0, (3.0, 2.2)), zorder=1)
    for key, group, y in rows:
        colour = GROUP_COLOUR[group]
        ax.plot([left, values[key]], [y, y], color=colour, alpha=0.32,
                linewidth=1.0, zorder=2)
        ax.plot([values[key]], [y], "o", markersize=4.6, color=colour,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    ax.set_ylim(height - 0.55, -0.75)
    ax.set_xlabel(xlabel, fontsize=8.6)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, right=False)
    ax.tick_params(axis="x", which="both", top=False)


def main() -> Path:
    fs.use()
    summary = pd.read_parquet(
        ROOT / "data" / "benchmark" / "sensitivity_summary.parquet").set_index("config")
    validate_configurations(summary)
    rows, ticks, labels, colours, weights, height = layout()

    fig, axes = plt.subplots(
        1, 3, figsize=(fs.FULL_WIDTH, max(4.15, height * 0.145)),
        gridspec_kw={"width_ratios": [1.0, 0.92, 0.62], "wspace": 0.085})

    # Reported in equivalence classes, the same unit as the headline result,
    # because a lattice of years the evidence cannot separate is one physical
    # explanation rather than many candidates.
    leading = (summary.leading_group_rate * 100).to_dict()
    panel(axes[0], rows, leading, leading["baseline"],
          "Published year in\nfirst group (%)", 0.0, height)
    axes[0].set_xlim(0, 104)
    axes[0].set_xticks([0, 40, 80])
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels, fontsize=7.6)
    for tick, colour, weight in zip(axes[0].get_yticklabels(), colours,
                                    weights):
        tick.set_color(colour)
        if weight == "italic":
            tick.set_style("italic")
        elif weight == "bold":
            tick.set_fontweight("bold")

    group = summary.median_leading_group.to_dict()
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.75, 1500)
    panel(axes[1], rows, group, group["baseline"],
          "Years in that group\n(median)", 0.75, height)
    axes[1].set_xticks([1, 10, 100, yr.CANDIDATE_YEAR_COUNT])
    axes[1].set_xticklabels(["1", "10", "100", str(yr.CANDIDATE_YEAR_COUNT)])
    axes[1].set_yticks([])

    recall = (summary.strict_recall * 100).to_dict()
    panel(axes[2], rows, recall, recall["baseline"],
          "Published year in\nstrict set (%)", 84.0, height)
    axes[2].set_xlim(84, 101.5)
    axes[2].set_xticks([88, 94, 100])
    axes[2].set_yticks([])

    for ax, letter in zip(axes, "abc"):
        fs.panel_label(ax, letter, x=0.0, y=1.045)
    fig.subplots_adjust(left=0.300, right=0.995, top=0.962, bottom=0.115)
    return fs.save(fig, OUT / "figS1_sensitivity")


if __name__ == "__main__":
    print(main())
