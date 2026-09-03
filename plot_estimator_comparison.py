"""Reproduce the paper's estimator-comparison plot from the result CSV."""

from __future__ import annotations

from pathlib import Path
import csv
import string

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("CP", "DFA", "HFD", "PSD", "SVA")
H_PANELS = (0.2, 0.5, 0.8)
L_VALUES = (1000, 2000, 5000)
BOUND_SETTINGS = ((0.10, 0.05), (0.10, 0.005), (0.05, 0.005))
BASE_COLORS = (
    (0.1725, 0.6275, 0.1725),
    (0.1216, 0.4667, 0.7059),
    (1.0, 0.4980, 0.0549),
    (0.8392, 0.1529, 0.1569),
    (0.5804, 0.4039, 0.7412),
)


def _load_plot_data(results_path: str | Path):
    """Load, validate, and order data exactly as visualised in the figure."""
    with Path(results_path).open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    if not data:
        raise ValueError("Result file is empty.")
    required = {
        "method", "H_true", "L", "mean_estimate", "standard_deviation"
    }
    for p, epsilon in BOUND_SETTINGS:
        required.add(f"universal_profile_points_p{p:g}_epsilon{epsilon:g}")
    missing = required.difference(data[0])
    if missing:
        raise ValueError(f"Result file is missing columns: {sorted(missing)}")

    means, standard_deviations = {}, {}
    for H in H_PANELS:
        means[H], standard_deviations[H] = {}, {}
        for L in L_VALUES:
            selected = {
                row["method"]: row
                for row in data
                if np.isclose(float(row["H_true"]), H) and int(row["L"]) == L
            }
            if set(selected) != set(METHODS):
                raise ValueError(f"Expected methods {METHODS} for H={H}, L={L}.")
            means[H][L] = np.array(
                [float(selected[method]["mean_estimate"]) for method in METHODS]
            )
            standard_deviations[H][L] = np.array(
                [float(selected[method]["standard_deviation"]) for method in METHODS]
            )

    # Shape: setting x method x L. This is exactly the bottom-panel order.
    universal_points = []
    for p, epsilon in BOUND_SETTINGS:
        column = f"universal_profile_points_p{p:g}_epsilon{epsilon:g}"
        setting = np.empty((len(METHODS), len(L_VALUES)))
        for method_index, method in enumerate(METHODS):
            for L_index, L in enumerate(L_VALUES):
                values = {
                    float(row[column])
                    for row in data
                    if row["method"] == method and int(row["L"]) == L
                }
                if len(values) != 1:
                    raise ValueError(
                        f"Expected one universal value for method={method}, L={L}."
                    )
                setting[method_index, L_index] = values.pop()
        universal_points.append(setting)
    return means, standard_deviations, universal_points


def plot_estimator_comparison(
    results_path: str | Path = "estimator_comparison_results.csv",
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Reproduce the estimator-comparison figure from the generated CSV.

    Parameters
    ----------
    results_path : path-like, default="estimator_comparison_results.csv"
        CSV produced by :func:`generate_estimator_comparison_results`.
    output_path : path-like, optional
        Optional image or PDF destination. The file type follows the suffix.
    show : bool, default=True
        Display the figure using Matplotlib.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing nine estimator panels and three universal-BEB
        panels.

    Raises
    ------
    FileNotFoundError
        If ``results_path`` does not exist.
    ValueError
        If required columns or method/H/L combinations are missing, duplicated,
        or inconsistent.

    Notes
    -----
    Panel order is fixed to ``H=(0.2, 0.5, 0.8)`` by rows and
    ``L=(1000, 2000, 5000)`` by columns. The bottom panels display
    ``L * B_univ`` in the order documented in the paper.
    """
    means, stds, extra_data = _load_plot_data(results_path)

    # Original design parameters.
    bar_thickness = 0.8
    row_spacing = 0.6
    bar_group_distance = 2
    axis_length_percentage = 0.3
    pad_frac = 0.08
    plot_width = 1.4
    central_line_width = 4.0
    error_end_marker_width = 4.5
    error_end_marker_frac = 0.45
    x_label_distance = 25
    x_label_font_size = 32
    ytick_fontsize = 30
    x_axis_labels = ("av. Hurst exponent", "number of data points")

    custom_xlims = {
        (0.2, 1000): (0.18, 0.22), (0.2, 2000): (0.18, 0.22),
        (0.2, 5000): (0.18, 0.22), (0.5, 1000): (0.48, 0.52),
        (0.5, 2000): (0.48, 0.52), (0.5, 5000): (0.48, 0.52),
        (0.8, 1000): (0.78, 0.82), (0.8, 2000): (0.78, 0.82),
        (0.8, 5000): (0.78, 0.82),
    }
    num_xticks = {
        (0.2, 1000): 4, (0.2, 2000): 4, (0.2, 5000): 5,
        (0.5, 1000): 4, (0.5, 2000): 4, (0.5, 5000): 5,
        (0.8, 1000): 4, (0.8, 2000): 4, (0.8, 5000): 5,
    }
    bottom_titles = (
        r"$(p,\varepsilon) = (0.10, 0.05)$",
        r"$(p,\varepsilon) = (0.10, 0.005)$",
        r"$(p,\varepsilon) = (0.05, 0.005)$",
    )

    # The original global centred span includes estimates, standard deviations,
    # and the custom axis limits before the displayed fraction is selected.
    half_width = 0.0
    for H in H_PANELS:
        for L in L_VALUES:
            for mean, sigma in zip(means[H][L], stds[H][L]):
                half_width = max(half_width, abs(mean - H) + sigma)
            x0, x1 = custom_xlims[(H, L)]
            half_width = max(half_width, abs(x0 - H), abs(x1 - H))
    half_width = (half_width * (1.0 + pad_frac) or 0.02)
    displayed_half_width = half_width * axis_length_percentage

    number_of_methods = len(METHODS)
    top_height = number_of_methods * (bar_thickness + row_spacing)
    centers = np.arange(number_of_methods) * (
        bar_thickness + bar_group_distance
    )
    bottom_height = centers[-1] + bar_thickness
    total_height = 3 * top_height + bottom_height + 1

    figure = plt.figure(
        figsize=(20 * plot_width, total_height), dpi=500
    )
    figure.subplots_adjust(left=0.07)
    grid = plt.GridSpec(
        4, 3,
        hspace=row_spacing,
        wspace=0.2,
        height_ratios=[top_height] * 3 + [bottom_height],
        width_ratios=[plot_width] * 3,
    )
    plt.rcParams.update({
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "xtick.major.pad": 10,
        "ytick.major.pad": 10,
    })

    letters = iter(string.ascii_lowercase)
    for row, H in enumerate(H_PANELS):
        for column, L in enumerate(L_VALUES):
            axis = figure.add_subplot(grid[row, column])
            axis.set_title(
                rf"$\mathbf{{{next(letters)})}}$  H = {H}, L = {L}",
                fontsize=35,
                pad=10,
            )
            x_min = H - displayed_half_width
            x_max = H + displayed_half_width
            axis.set_xlim(x_min, x_max)
            ticks = np.linspace(x_min, x_max, num_xticks[(H, L)])
            axis.set_xticks(ticks)
            axis.set_xticklabels([f"{tick:.2f}" for tick in ticks])
            axis.axvline(
                H, linestyle="--", color="black",
                linewidth=central_line_width, zorder=2,
            )
            y_positions = np.arange(number_of_methods) * (
                bar_thickness + row_spacing
            )

            for method_index, (mean, sigma, color) in enumerate(
                zip(means[H][L], stds[H][L], BASE_COLORS)
            ):
                y = y_positions[method_index]
                axis.barh(
                    y, mean - H, left=H, height=bar_thickness,
                    color=color, edgecolor="black", zorder=3,
                )
                marker_half_height = bar_thickness * error_end_marker_frac
                axis.plot(
                    [mean, mean],
                    [y - marker_half_height, y + marker_half_height],
                    color="black", linewidth=error_end_marker_width, zorder=7,
                )
                left = max(mean - sigma, x_min)
                right = min(mean + sigma, x_max)
                axis.plot([left, right], [y, y], color="black", linewidth=1.5, zorder=6)
                cap_height = bar_thickness * 0.35
                axis.plot([left, left], [y-cap_height, y+cap_height], color="black", linewidth=1.5, zorder=6)
                axis.plot([right, right], [y-cap_height, y+cap_height], color="black", linewidth=1.5, zorder=6)

            axis.set_yticks(y_positions)
            axis.set_yticklabels(METHODS, fontsize=ytick_fontsize)
            axis.invert_yaxis()
            axis.set_xlabel(
                x_axis_labels[0], labelpad=x_label_distance,
                fontsize=x_label_font_size,
            )

    offsets = np.linspace(-bar_thickness, bar_thickness, 3)
    for column, values in enumerate(extra_data):
        axis = figure.add_subplot(grid[3, column])
        axis.set_title(
            rf"$\mathbf{{{next(letters)})}}$   {bottom_titles[column]}",
            fontsize=25, pad=10,
        )
        for method_index in range(number_of_methods):
            for L_index in range(3):
                axis.barh(
                    centers[method_index] + offsets[L_index],
                    values[method_index, L_index],
                    height=bar_thickness,
                    color=BASE_COLORS[method_index],
                    edgecolor="black",
                )
        axis.set_yticks(centers)
        axis.set_yticklabels(METHODS, fontsize=ytick_fontsize)
        axis.invert_yaxis()
        axis.set_xlabel(
            x_axis_labels[1], labelpad=x_label_distance,
            fontsize=x_label_font_size,
        )
        x0, x1 = axis.get_xlim()
        if column == 1:
            padding = (x1 - x0) * 0.01
            axis.set_xlim(x0, max(x1, float(np.max(values[0])) + 8 * padding))
            for L_index, L in enumerate(L_VALUES):
                axis.text(
                    values[0, L_index] + 2 * padding,
                    centers[0] + offsets[L_index],
                    f"L = {L}", va="center", ha="left",
                    fontsize=x_label_font_size,
                )
        else:
            axis.set_xlim(x0, x1 * 1.1)

    if output_path is not None:
        figure.savefig(output_path, dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    return figure


if __name__ == "__main__":
    plot_estimator_comparison()
