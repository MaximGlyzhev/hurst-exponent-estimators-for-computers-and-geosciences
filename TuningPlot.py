"""Visualise the tuning-parameter sensitivity of DFA, HFD, and SVA.

The input CSV must contain one row per simulated Hurst parameter and tuning-
parameter configuration. Required columns are ``method``, ``H_true``,
``mean``, ``std``, ``f``, ``n``, ``l_DFA``, and ``l_min``. For every method
and configuration, the script computes the maximum absolute estimation error
and maximum standard deviation over the selected Hurst parameters. A
farthest-point procedure then selects representative configurations for a
readable four-panel plot.

The script writes ``TuningPlot.pdf`` and ``TuningPlot.png`` according to the
settings below and subsequently displays the figure. Plot construction and
the representative-point selection reproduce the figure used in the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

# =============================
# USER SETTINGS
# =============================
CSV_PATH = "dfa_sva_hfd_parameters_optimised.csv"
FONT_SIZE = 16
FONT_WEIGHT = "bold"
DPI = 600
SAVE_PDF = True
SAVE_PNG = True
OUTPUT_NAME = "TuningPlot"
K_POINTS = 7

# Filters
F_MIN, F_MAX_EXCL = 0.05, 0.5
N_MIN, N_MAX = 3, 20
L_MIN, L_MAX = 3, 20
H_VALUES = np.round(np.arange(0.1, 1.0, 0.1), 1)
METHODS = ["DFA", "HFD", "SVA"]

# Base spacing between subplot rows and columns.
HSPACE = 0.42
WSPACE = 0.22

# =============================
# LEGEND LAYOUT CONTROLS
# =============================
LEGEND_WIDTH = 0.92
LEGEND_GAP = 0.08
LEGEND_BOTTOM_PAD = 0.02

LEGEND_HEIGHT_MIN = 0.13
LEGEND_HEIGHT_MAX = 0.28

LEGEND_NCOL_START = 5
LEGEND_NCOL_MIN = 2

LEGEND_BORDERPAD = 0.9
LEGEND_LABELSPACING = 0.8
LEGEND_COLUMNSPACING = 1.2
LEGEND_HANDLELENGTH = 2.0
LEGEND_HANDLETEXTPAD = 0.6

# =============================
# FORCE-IN POINTS (stars)
# =============================
# NOTE: l means l_min for HFD/SVA, and l_DFA for DFA
SPECIAL_POINTS = {
    "HFD": (0.05, 10, 3),
    "SVA": (0.05, 10, 3),
    "DFA": (0.10, 10, 20),
}

# =============================
# COLORS AND MARKERS
# =============================
STYLE = {
    "DFA": {"color": (0.1216, 0.4667, 0.7059), "marker": "o"},
    "HFD": {"color": (1.0, 0.4980, 0.0549), "marker": "s"},
    "SVA": {"color": (0.5804, 0.4039, 0.7412), "marker": "^"},
}

# =============================
# GLOBAL FONT
# =============================
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "font.weight": FONT_WEIGHT,
    "axes.titleweight": FONT_WEIGHT,
    "axes.labelweight": FONT_WEIGHT,
    "legend.fontsize": FONT_SIZE,
})

# =============================
# HELPERS
# =============================
def method_specific_lcol(method: str) -> str:
    """Return the input column containing the method's minimum scale."""
    return "l_DFA" if method == "DFA" else "l_min"


def pick_representative_points(df_xy: pd.DataFrame, k: int) -> pd.DataFrame:
    """Select up to ``k`` well-separated configurations in error--SD space.

    The two plotting coordinates are independently normalised. Selection
    starts with the point closest to the lower-left corner and iteratively
    adds the point farthest from the selected set. Rows and columns are
    returned unchanged.

    Parameters
    ----------
    df_xy : pandas.DataFrame
        Aggregated configurations containing ``max_abs_error`` and
        ``max_std``.
    k : int
        Maximum number of representative configurations.

    Returns
    -------
    pandas.DataFrame
        Selected rows from ``df_xy``.
    """
    if len(df_xy) <= k:
        return df_xy.copy()

    x = df_xy["max_abs_error"].to_numpy()
    y = df_xy["max_std"].to_numpy()

    def norm(a):
        mn, mx = float(np.min(a)), float(np.max(a))
        return np.zeros_like(a) if mx - mn < 1e-12 else (a - mn) / (mx - mn)

    pts = np.column_stack([norm(x), norm(y)])
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    chosen = [start]
    d2 = np.full(len(df_xy), np.inf)

    for _ in range(1, k):
        last = chosen[-1]
        diff = pts - pts[last]
        d2 = np.minimum(d2, np.sum(diff * diff, axis=1))
        chosen.append(int(np.argmax(d2)))

    return df_xy.iloc[chosen].copy()


def expand_axis_margins(ax, right=0.22, top=0.18, left=0.05, bottom=0.05):
    """Expand axis limits by fractions of the current data ranges."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = xmax - xmin
    yr = ymax - ymin
    ax.set_xlim(xmin - xr * left, xmax + xr * right)
    ax.set_ylim(ymin - yr * bottom, ymax + yr * top)


def annotate_no_overlap(ax, xs, ys, labels, *, fontsize=None, fontweight=None):
    """Place point labels while penalising overlap and boundary violations.

    Candidate offsets are scored in display coordinates. Label--label
    overlap is penalised most strongly, followed by overlap with the point,
    overflow beyond the axes, and finally the offset distance. No leader
    lines are drawn.
    """
    if fontsize is None:
        fontsize = FONT_SIZE - 3
    if fontweight is None:
        fontweight = FONT_WEIGHT

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer=renderer)

    PAD = 3.0  # Pixel padding used to keep labels inside the axes.

    # Candidate label offsets in points, including longer offsets near edges.
    offsets = [
        (12, 10), (14, 0), (12, -10), (0, 14), (0, -14), (-14, 0),
        (-12, 10), (-12, -10),
        (18, 12), (18, -12), (-18, 12), (-18, -12),
        (26, 0), (-26, 0), (0, 26), (0, -26),
        (34, 16), (34, -16), (-34, 16), (-34, -16),
        (42, 0), (-42, 0), (0, 42), (0, -42),
        (52, 18), (52, -18), (-52, 18), (-52, -18),
        (64, 0), (-64, 0), (0, 64), (0, -64),
        (78, 24), (78, -24), (-78, 24), (-78, -24),
        (92, 0), (-92, 0), (0, 92), (0, -92),
    ]

    placed_bbs = []

    def outside_penalty(bb):
        dx_left = max(0.0, (ax_bb.x0 + PAD) - bb.x0)
        dx_right = max(0.0, bb.x1 - (ax_bb.x1 - PAD))
        dy_bottom = max(0.0, (ax_bb.y0 + PAD) - bb.y0)
        dy_top = max(0.0, bb.y1 - (ax_bb.y1 - PAD))
        return dx_left + dx_right + dy_bottom + dy_top

    # Marker hitbox radius in pixels.
    POINT_R_PX = 10.0

    for x, y, lab in zip(xs, ys, labels):
        # Point location in display coordinates.
        px, py = ax.transData.transform((x, y))
        point_bb = Bbox.from_extents(px - POINT_R_PX, py - POINT_R_PX, px + POINT_R_PX, py + POINT_R_PX)

        best = None
        best_score = 1e30

        for dx, dy in offsets:
            ann = ax.annotate(
                lab, (x, y),
                textcoords="offset points", xytext=(dx, dy),
                fontsize=fontsize, fontweight=fontweight,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
                clip_on=True
            )
            fig.canvas.draw()

            bb = ann.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
            overlaps_labels = sum(bb.overlaps(pbb) for pbb in placed_bbs)
            out = outside_penalty(bb)
            overlaps_point = 1 if bb.overlaps(point_bb) else 0

            score = (
                overlaps_labels * 1e9 +
                overlaps_point * 1e8 +
                out * 1e6 +
                (dx * dx + dy * dy)
            )

            if score < best_score:
                best_score = score
                best = (dx, dy)

            ann.remove()

        dx, dy = best

        ann = ax.annotate(
            lab, (x, y),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=fontsize, fontweight=fontweight,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
            clip_on=True
        )

        # Clamp the final annotation inside the axes.
        fig.canvas.draw()
        bb = ann.get_window_extent(renderer=renderer).expanded(1.02, 1.10)

        shift_x = 0.0
        shift_y = 0.0
        if bb.x0 < ax_bb.x0 + PAD:
            shift_x = (ax_bb.x0 + PAD) - bb.x0
        elif bb.x1 > ax_bb.x1 - PAD:
            shift_x = (ax_bb.x1 - PAD) - bb.x1

        if bb.y0 < ax_bb.y0 + PAD:
            shift_y = (ax_bb.y0 + PAD) - bb.y0
        elif bb.y1 > ax_bb.y1 - PAD:
            shift_y = (ax_bb.y1 - PAD) - bb.y1

        if shift_x != 0.0 or shift_y != 0.0:
            px_to_pt = 72.0 / fig.dpi
            ann.set_position((dx + shift_x * px_to_pt, dy + shift_y * px_to_pt))
            fig.canvas.draw()

        placed_bbs.append(ann.get_window_extent(renderer=renderer).expanded(1.02, 1.10))


def add_common_legend_auto(fig, *, legend_width):
    """Create a common bottom legend and resize it until its content fits.

    The routine first reduces the number of columns and then increases the
    reserved legend height within the configured limits.
    """
    legend_row1 = [
        Line2D([0], [0], marker=STYLE["DFA"]["marker"], linestyle="None",
               markerfacecolor=STYLE["DFA"]["color"], markeredgecolor="none",
               markersize=10, label="DFA"),
        Line2D([0], [0], marker=STYLE["HFD"]["marker"], linestyle="None",
               markerfacecolor=STYLE["HFD"]["color"], markeredgecolor="none",
               markersize=10, label="HFD"),
        Line2D([0], [0], marker=STYLE["SVA"]["marker"], linestyle="None",
               markerfacecolor=STYLE["SVA"]["color"], markeredgecolor="none",
               markersize=10, label="SVA"),
        Line2D([0], [0], marker="*", linestyle="None",
               markerfacecolor="gray", markeredgecolor="black",
               markersize=14, label="parameters for comparison"),
        Line2D([0], [0], linestyle="None", label="Point labels: (f,l,n)"),
    ]

    legend_row2 = [
        Line2D([0], [0], linestyle="None",
               label="f: upper cut-off coefficient l: minimal scale n: number of fitting points"),
    ]

    legend_elements = legend_row1 + legend_row2

    ncol = LEGEND_NCOL_START
    height = LEGEND_HEIGHT_MIN

    while True:
        bottom_reserved = LEGEND_BOTTOM_PAD + height + LEGEND_GAP
        plt.subplots_adjust(bottom=bottom_reserved)

        for ax in list(fig.axes):
            if getattr(ax, "_is_legend_ax", False):
                fig.delaxes(ax)

        legend_left = 0.5 - legend_width / 2.0
        legend_ax = fig.add_axes([legend_left, LEGEND_BOTTOM_PAD, legend_width, height])
        legend_ax._is_legend_ax = True
        legend_ax.axis("off")

        leg = legend_ax.legend(
            handles=legend_elements,
            loc="center",
            ncol=ncol,
            frameon=True,
            fancybox=False,
            borderaxespad=0.0,
            borderpad=LEGEND_BORDERPAD,
            labelspacing=LEGEND_LABELSPACING,
            columnspacing=LEGEND_COLUMNSPACING,
            handlelength=LEGEND_HANDLELENGTH,
            handletextpad=LEGEND_HANDLETEXTPAD,
        )

        frame = leg.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(1.8)
        frame.set_alpha(1.0)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        leg_bb = leg.get_window_extent(renderer=renderer)
        ax_bb = legend_ax.get_window_extent(renderer=renderer)
        safety_px = 2.0

        fits_width = (leg_bb.x0 >= ax_bb.x0 - safety_px) and (leg_bb.x1 <= ax_bb.x1 + safety_px)
        fits_height = (leg_bb.y0 >= ax_bb.y0 - safety_px) and (leg_bb.y1 <= ax_bb.y1 + safety_px)

        if fits_width and fits_height:
            break

        if not fits_width and ncol > LEGEND_NCOL_MIN:
            ncol -= 1
            continue

        if not fits_height and height < LEGEND_HEIGHT_MAX:
            extra_px = (leg_bb.height - ax_bb.height) + 10.0
            extra_fig = extra_px / fig.bbox.height
            height = min(LEGEND_HEIGHT_MAX, height + max(0.02, extra_fig))
            continue

        break


# =============================
# LOAD + FILTER
# =============================
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]
df["abs_error"] = (df["mean"] - df["H_true"]).abs()

df = df[
    (df["f"] >= F_MIN) &
    (df["f"] < F_MAX_EXCL) &
    (df["n"].between(N_MIN, N_MAX)) &
    (df["H_true"].isin(H_VALUES))
].copy()

selected_by_method = {}

for method in METHODS:
    sub = df[df["method"] == method].copy()
    lcol = method_specific_lcol(method)
    sub = sub[sub[lcol].between(L_MIN, L_MAX)].copy()

    if sub.empty:
        selected_by_method[method] = (pd.DataFrame(), lcol, [])
        continue

    grouped = (
        sub.groupby(["f", "n", lcol], as_index=False)
           .agg(max_abs_error=("abs_error", "max"), max_std=("std", "max"))
    )

    sel = pick_representative_points(grouped, K_POINTS)

    star_indices = []
    if method in SPECIAL_POINTS:
        f0, n0, l0 = SPECIAL_POINTS[method]
        m = (np.isclose(grouped["f"], f0)) & (grouped["n"] == n0) & (grouped[lcol] == l0)
        spec = grouped[m]
        if not spec.empty:
            spec_row = spec.iloc[[0]].copy()
            already = ((np.isclose(sel["f"], f0)) & (sel["n"] == n0) & (sel[lcol] == l0)).any()
            if not already:
                sel = pd.concat([sel, spec_row], ignore_index=True)
            star_indices = list(sel.index[
                (np.isclose(sel["f"], f0)) & (sel["n"] == n0) & (sel[lcol] == l0)
            ])

    selected_by_method[method] = (sel, lcol, star_indices)

# =============================
# PLOTTING (2 x 2)
# =============================
FIG_WIDTH = 16
FIG_HEIGHT = 14.0

HSPACE_PLOT = HSPACE + 0.10
WSPACE_PLOT = WSPACE + 0.06

fig, axes = plt.subplots(
    2, 2,
    figsize=(FIG_WIDTH, FIG_HEIGHT),
    dpi=DPI,
    gridspec_kw={"hspace": HSPACE_PLOT, "wspace": WSPACE_PLOT}
)

plt.subplots_adjust(
    bottom=LEGEND_BOTTOM_PAD + LEGEND_HEIGHT_MIN + LEGEND_GAP,
    wspace=WSPACE_PLOT,
    hspace=HSPACE_PLOT
)

method_axes = {
    "DFA": axes[0, 0],
    "HFD": axes[0, 1],
    "SVA": axes[1, 0],
}
ax_all = axes[1, 1]

# =============================
# PANEL LABELS OUTSIDE THE TOP-LEFT CORNERS
# =============================
def add_panel_label_outside(ax, text):
    """Place a panel label immediately outside an axes' top-left corner."""
    ax.text(
        -0.13, 1.10, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FONT_SIZE + 8,
        fontweight="black",
        clip_on=False
    )

# --- individual plots
for method in METHODS:
    ax = method_axes[method]
    sel, lcol, star_idx = selected_by_method[method]

    ax.set_title(method)
    ax.set_xlabel("Maximum absolute estimation error")
    ax.set_ylabel("Maximum standard deviation")

    if method == "DFA":
        add_panel_label_outside(ax, "a)")
    elif method == "HFD":
        add_panel_label_outside(ax, "b)")
    elif method == "SVA":
        add_panel_label_outside(ax, "c)")

    if sel.empty:
        continue

    st = STYLE[method]
    ax.scatter(
        sel["max_abs_error"], sel["max_std"],
        marker=st["marker"], color=st["color"],
        s=70, zorder=3
    )

    expand_axis_margins(ax, right=0.28, top=0.20, left=0.06, bottom=0.06)

    xs = sel["max_abs_error"].to_numpy()
    ys = sel["max_std"].to_numpy()
    labels = [
        f"({r.f:.2f},{int(getattr(r, lcol))},{int(r.n)})"
        for r in sel.itertuples(index=False)
    ]

    annotate_no_overlap(ax, xs, ys, labels)

    if star_idx:
        ax.scatter(
            sel.loc[star_idx, "max_abs_error"], sel.loc[star_idx, "max_std"],
            marker="*", color=st["color"],
            s=220, edgecolors="black", linewidths=0.7, zorder=6
        )

# --- combined plot
ax_all.set_title("All methods")
ax_all.set_xlabel("Maximum absolute estimation error")
ax_all.set_ylabel("Maximum standard deviation")
add_panel_label_outside(ax_all, "d)")

for method in METHODS:
    sel, _, star_idx = selected_by_method[method]
    if sel.empty:
        continue

    st = STYLE[method]
    ax_all.scatter(
        sel["max_abs_error"], sel["max_std"],
        marker=st["marker"], color=st["color"],
        s=70, zorder=3
    )

    if star_idx:
        ax_all.scatter(
            sel.loc[star_idx, "max_abs_error"], sel.loc[star_idx, "max_std"],
            marker="*", color=st["color"],
            s=220, edgecolors="black", linewidths=0.7, zorder=6
        )

expand_axis_margins(ax_all, right=0.22, top=0.18, left=0.06, bottom=0.06)

# =============================
# COMMON LEGEND (auto wrap + auto height)
# =============================
add_common_legend_auto(fig, legend_width=LEGEND_WIDTH)

# =============================
# SAVE
# =============================
if SAVE_PDF:
    plt.savefig(f"{OUTPUT_NAME}.pdf", dpi=DPI, bbox_inches="tight")
if SAVE_PNG:
    plt.savefig(f"{OUTPUT_NAME}.png", dpi=DPI, bbox_inches="tight")

plt.show()

