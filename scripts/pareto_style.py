import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PARETO_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def apply_pareto_rcparams():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "figure.autolayout": True,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def scatter_pareto(ax, x_data, y_data, label, color):
    return ax.scatter(
        x_data,
        y_data,
        c=color,
        marker="o",
        s=26,
        edgecolors="k",
        linewidths=0.35,
        alpha=0.58,
        label=label,
        zorder=3,
        rasterized=True,
    )


def style_pareto_axis(ax, all_x, all_y, legend_handles=None, legend_title=None):
    ax.set_xlabel("Cost (CNY)", fontweight="bold", fontsize=16)
    ax.set_ylabel("Carbon Emissions (kg CO$_2$)", fontweight="bold", fontsize=16)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(16)
    ax.yaxis.get_offset_text().set_fontsize(16)
    ax.tick_params(axis="both", which="major", length=10, width=2)
    ax.tick_params(axis="both", which="minor", length=6, width=1)
    ax.grid(True, which="both", linestyle="--", alpha=1.0, linewidth=2.0)

    all_x = np.asarray(all_x, dtype=float)
    all_y = np.asarray(all_y, dtype=float)
    x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    if x_max > x_min:
        ax.set_xlim([x_min * 0.8, x_max * 1.2])
    if y_max > y_min:
        y_pad = 0.08 * (y_max - y_min)
        ax.set_ylim([max(0.0, y_min - y_pad), y_max + y_pad])

    if legend_handles is None:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="black", title=legend_title)
    else:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            framealpha=0.9,
            edgecolor="black",
            title=legend_title,
        )
