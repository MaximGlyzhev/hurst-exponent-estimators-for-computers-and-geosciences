"""Create the CP robustness figure used in the accompanying paper.

The three panel rows compare the maximum absolute CP estimation errors for
additive height noise, lateral sampling-position errors, and finite-bit height
quantisation. Columns represent profile lengths and bar colours represent
three ranges of true Hurst parameters. Within every input matrix, row index
zero is displayed at the bottom and row index two at the top.

The numerical values below are fixed results of the simulation study. This
module only visualises them; it does not rerun the simulations.
"""

import string

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["xtick.major.pad"] = 10


def plot_cp_robustness_results(
    bit_depth_errors,
    white_noise_errors,
    displacement_errors,
    f1=None,
    f2=None,
    colour1="black",
    colour2="black",
    xlabel="Av. est. error",
    label_offset=0.05,
    dpi=600,
    header_coords=None,
    mod_label_coords=None,
    layer_headers=None,
    layer_mod_labels=None,
    x_spacing=0.3,
    custom_xlims=None,
    num_xticks=5,
    label_font_size=25,
    legend_font_size=22,
    plot_width=5,
    label_width=7.0,
):
    """Plot grouped CP errors for the three measurement modifications.

    Parameters
    ----------
    bit_depth_errors, white_noise_errors, displacement_errors : mapping
        Dictionaries keyed by profile length ``L``. Each value must be a
        ``(3, 3)`` array-like object. Rows represent the three modification
        settings in bottom-to-top order; columns represent the three Hurst-
        parameter groups.
    f1, f2 : float, optional
        Vertical reference values for the two error bounds.
    colour1, colour2 : color, default="black"
        Colours of the reference lines.
    xlabel : str, default="Av. est. error"
        Label applied to every data axis.
    label_offset : float, default=0.05
        Horizontal position of the default row labels.
    dpi : int, default=600
        Figure resolution.
    header_coords, mod_label_coords : tuple or sequence, optional
        Manual positions for row headers and modification labels.
    layer_headers, layer_mod_labels : sequence or mapping, optional
        Custom text for the three rows and their modification settings.
    x_spacing : float, default=0.3
        Horizontal gap between data columns in inches.
    custom_xlims : mapping, optional
        Optional limits indexed first by row and then by profile length.
    num_xticks : int, default=5
        Number of evenly spaced x-axis ticks.
    label_font_size, legend_font_size : int
        Font sizes for axis text and the common legend.
    plot_width, label_width : float
        Widths of the data panels and label column in inches.

    Notes
    -----
    The function displays the figure but deliberately does not save it. This
    preserves the behaviour of the script used to create the paper figure.
    """

    if layer_headers is None:
        layer_headers = [
            ("height", "measurement", "errors"),
            ("x-axis", "displacement", ""),
            ("z-scale", "resolution", ""),
        ]
    if isinstance(layer_headers, dict):
        layer_headers = [layer_headers.get(i, ("", "", "")) for i in range(3)]

    if layer_mod_labels is None:
        layer_mod_labels = [
            [(r"I=0.22 ($\tau = 40$)","", ""), ("I=0.22  ","", ""), ("I=0.0125","", "")],                  # was row 1
            [(r"$p_{nd}=20\%$", "", ""), (r"$p_{nd}=50\%$", "", ""), (r"$p_{nd}=90\%$", "", "")],  # was row 2
            [("32bit",   "", ""), ("16bit", "", ""), ("8bit",  "", "")],                    # was row 0
        ]
    if isinstance(layer_mod_labels, dict):
        layer_mod_labels = [layer_mod_labels.get(i, [("", "", "")] * 3) for i in range(3)]

    L_values = sorted(bit_depth_errors.keys())
    groups         = ['H = 0.1, 0.2',
                      'H = 0.3, 0.4, 0.5, 0.6, 0.7',
                      'H = 0.8, 0.9']
    n_mods         = 3

    bar_height     = 0.25
    offsets        = [-bar_height, 0, bar_height]
    default_colors = ['skyblue', 'salmon', 'lightgreen']
    exceed_color   = 'lightgrey'

    # — Figure + grid setup —
    nrows       = 3
    n_data      = len(L_values)
    ncols       = 1 + n_data
    fig_h       = 24
    fig_w       = label_width + n_data*(plot_width + x_spacing)
    width_ratios = [label_width] + [plot_width]*n_data

    gs = gridspec.GridSpec(nrows, ncols,
                           width_ratios=width_ratios,
                           wspace=x_spacing/plot_width,
                           hspace=0.3)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # — Left label column —
    for row in range(nrows):
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.axis('off')
        ax_lbl.set_ylim(-0.5, n_mods - 0.5)

        # header text
        if header_coords is not None:
            hc_x, hc_y = header_coords
        else:
            hc_x, hc_y = label_offset, n_mods - 0.3
        h0, h1, h2 = layer_headers[row]
        ax_lbl.text(hc_x, hc_y, f"{h0}\n{h1}\n{h2}",
                    fontweight='bold', va='top', ha='left',
                    fontsize=label_font_size, fontfamily='monospace')

        # module labels
        if mod_label_coords is not None:
            coords = mod_label_coords
        else:
            coords = [(label_offset, n_mods - 1.4 - i*0.93) for i in range(n_mods)]
        for (mx, my), txt in zip(coords, layer_mod_labels[row]):
            ax_lbl.text(mx, my,
                        f"{txt[0]:>7}    {txt[1]:>10}    {txt[2]:>5}",
                        va='center', ha='left',
                        fontsize=label_font_size, fontfamily='monospace')

    # — Data axes creation —
    data_axes = []
    for row in range(nrows):
        row_axes = []
        for j, L in enumerate(L_values):
            ax = fig.add_subplot(gs[row, j+1])
            row_axes.append(ax)
        data_axes.append(row_axes)

    # Input row zero is the bottom setting, matching the label coordinates.
    y_pos = np.arange(n_mods)

    def plot_row(ax, mat, pct, L, ridx):
        # x-limits
        if custom_xlims and ridx in custom_xlims and L in custom_xlims[ridx]:
            xmin, xmax = custom_xlims[ridx][L]
        else:
            p = np.percentile(mat, pct)
            xmin, xmax = 0, 1.1 * p
        ax.set_xlim(xmin, xmax)
        if num_xticks > 1:
            ax.set_xticks(np.linspace(xmin, xmax, num_xticks))

        # single solid bars
        for gi, (off, col) in enumerate(zip(offsets, default_colors)):
            for mi, y in enumerate(y_pos):
                v = mat[mi, gi]
                face = exceed_color if v > xmax else col
                lbl = groups[gi] if (mi == 0 and L == L_values[0]) else None
                ax.barh(y + off, v, height=bar_height, color=face, label=lbl)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_xlabel(xlabel, fontsize=label_font_size)

    # Height-measurement errors.
    for ax, L in zip(data_axes[0], L_values):
        plot_row(ax, np.array(white_noise_errors[L]), 80, L, 0)
        ax.set_title(f"L = {L}", fontsize=label_font_size)
        if f1: ax.axvline(f1, linestyle='--', color=colour1)
        if f2: ax.axvline(f2, linestyle='--', color=colour2)

    # Lateral sampling-position errors.
    for ax, L in zip(data_axes[1], L_values):
        plot_row(ax, np.array(displacement_errors[L]), 80, L, 1)
        ax.set_title(f"L = {L}", fontsize=label_font_size)
        if f1: ax.axvline(f1, linestyle='--', color=colour1)
        if f2: ax.axvline(f2, linestyle='--', color=colour2)

    # Finite-bit height resolution.
    for ax, L in zip(data_axes[2], L_values):
        plot_row(ax, np.array(bit_depth_errors[L]), 100, L, 2)
        ax.set_title(f"L = {L}", fontsize=label_font_size)
        if f1: ax.axvline(f1, linestyle='--', color=colour1)
        if f2: ax.axvline(f2, linestyle='--', color=colour2)

    # — Subplot lettering —
    flat_axes = [ax for row in data_axes for ax in row]
    letters   = [f"{ltr})" for ltr in string.ascii_lowercase[:len(flat_axes)]]
    for ax, lbl in zip(flat_axes, letters):
        ax.text(-0.08, 1.02, lbl,
                transform=ax.transAxes,
                fontsize=label_font_size, fontweight='bold',
                va='bottom', ha='left')

    # ————— Legend —————
    blue_handle  = mpatches.Patch(facecolor=default_colors[0], label=groups[0])
    red_handle   = mpatches.Patch(facecolor=default_colors[1], label=groups[1])
    green_handle = mpatches.Patch(facecolor=default_colors[2], label=groups[2])

    exceed_handle = mpatches.Patch(facecolor=exceed_color,
                                   edgecolor='black',
                                   label='Exceeds scale range')

    line_handles = []
    if f1:
        line_handles.append(Line2D([0], [0], color=colour1, linestyle='--',
                                   label='0.025 error bound'))
    if f2:
        line_handles.append(Line2D([0], [0], color=colour2, linestyle='--',
                                   label='0.005 error bound'))

    # Column-major ordering for ncol=2 so the grid looks like:
    # row1: green | blue
    # row2: red   | ...
    all_handles = [
        green_handle,   # col1 row1
        red_handle,     # col1 row2
        blue_handle,    # col2 row1
        exceed_handle,
        *line_handles
    ]

    # center legend under middle data plot
    mid_col = len(L_values) // 2
    ref_ax = data_axes[nrows//2][mid_col]
    bbox = ref_ax.get_position()
    center_x = bbox.x0 + bbox.width/2

    fig.legend(
        all_handles, [h.get_label() for h in all_handles],
        loc='lower center', ncol=2,
        bbox_to_anchor=(center_x, -0.02),
        prop={'size': legend_font_size}
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


# Numerical results used in the paper figure. Within each matrix, rows are
# ordered from the bottom to the top of the corresponding panel.
bit_depth_errors = {
    100: [
        [0.011115019632950196, 0.010290331326487878, 0.02011506814198516],
        [0.01455747042287052,  0.010696532082651977, 0.01499794412958877],
        [0.013107929151150777, 0.010364266896256857, 0.016999869336394857],
    ],
    500: [
        [0.0018607655660374672, 0.0010646422041108439, 0.06304375619895752],
        [0.0017906442999967437, 0.001654270146887571,  0.0065296082240929865],
        [0.002355521035172642,  0.0022051044381542484, 0.006531465496886879],
    ],
    2000: [
        [0.0005392224914006066, 0.02667510690051389,   0.0215029677392935],
        [0.0010731633439555882, 0.0008968093229165808, 0.0037197534530791776],
        [0.0008522098688252266, 0.00081775859321781,   0.002083002076364804],
    ],
}

displacement_errors = {
    100: [
        [0.00835276158673609,  0.010196563939175007, 0.01784164910080477],
        [0.012272595564497307, 0.00983782442634501,  0.017166593340235403],
        [0.011789590677722922, 0.010200274580958124, 0.01645864593083135],
    ],
    500: [
        [0.004468413173473101,  0.00392939947615778,   0.007937174273239922],
        [0.0008440481021774748, 0.0023911363570190414, 0.008235414987781464],
        [0.003392969990702391,  0.0023019381253520654, 0.006655087939142157],
    ],
    2000: [
        [0.003741756620970843,   0.0029717423777856355, 0.005937131044829402],
        [0.00045671379175388216, 0.0009069209271347356, 0.003625685905716991],
        [0.00022070362647794162, 0.0009279775323055572, 0.0037270105960773847],
    ],
}

white_noise_errors = {
    100: [
        [0.014363328325857433, 0.00941641424488604, 0.016429460626753323],  # case 0 (bottom)
        [0.02550347817477036, 0.5933774371836089, 0.8847380374160123],  # case 1 (middle)
        [1, 1, 1],  # case 2 (top)

    ],
    500: [
        [0.0026920790660494853, 0.0031436682047930042, 0.007089705744364405],  # case 0 (bottom)
        [0.01575673095804425, 0.5822431868978057, 0.8775731113973473],     # case 1 (middle)
        [0.017693920424045384, 0.06135123446753077, 0.10698463561217986],  # case 2 (top)

    ],
    2000: [
        [0.0009396553352401388, 0.0006212131034805668, 0.003408848716689672],  # case 0 (bottom)
        [0.013178583516488124, 0.5804242898371983, 0.876268488318031],       # case 1 (middle)
        [0.0065634431933126736, 0.030169533276717897, 0.06836067417610336],  # case 2 (top)


    ],
}



ERROR_BOUND_1 = 0.025
ERROR_BOUND_2 = 0.005

CUSTOM_X_LIMITS = {
    i: {100: (0, 0.07), 500: (0, 0.07), 2000: (0, 0.07)}
    for i in range(3)
}


def main():
    """Display the CP robustness figure with the paper settings."""
    plot_cp_robustness_results(
        bit_depth_errors,
        white_noise_errors,
        displacement_errors,
        f1=ERROR_BOUND_1,
        f2=ERROR_BOUND_2,
        colour1="red",
        colour2="darkgreen",
        label_offset=-0.3,
        dpi=500,
        header_coords=(0.6, 3),
        mod_label_coords=[(0.6, 0), (0.6, 1), (0.6, 2)],
        x_spacing=1.4,
        custom_xlims=CUSTOM_X_LIMITS,
    )


if __name__ == "__main__":
    main()

