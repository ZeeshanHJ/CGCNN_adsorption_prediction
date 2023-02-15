
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from easy_mpl import ridge, regplot


def plot_ridge_reg(
        true, prediction,
        title:str,
        ridge_color:str="purple",
        marker_color="skyblue",
        cut=0.1,
        ylim = (0.0, 1.0),
        save=True,
        show = True,
        add_ridge = True,
):

    fig, ax = plt.subplots(figsize=(9, 7))

    if add_ridge:
        ax = ridge(pd.DataFrame(prediction, columns=['']),
                   color=["white"],
                   line_color=[ridge_color],
                   line_width=3.0,
                   title=title,
                   show=False,
                   ax=ax,
                   cut=cut
                   )

        ax[0].set_ylim(ylim)
        ax[0].set_ylabel("GCNN Prediction distribution", fontsize=20, color=ridge_color)
        ax[0].tick_params(axis='y', labelsize=18, color=ridge_color)
        ax[0].tick_params(axis='y', width=2, color=ridge_color)
    else:
        ax = [ax]

    ax[0].set_xlabel("DFT Calculated $\Delta \it{E} $(eV)", fontsize=20, color="black")
    ax[0].tick_params(axis='x', labelsize=18, color="black")
    ax[0].tick_params(axis='x', width=2, color="black")

    ax2 = ax[0].twinx()
    ax2 = regplot(true, prediction,
                  scatter_kws={"alpha": 0.5, 'linewidths': 0.5, 'edgecolors': 'black'},
                  marker_color=marker_color,
                  line_color="black", line_style='--',
                  line_kws={"linewidth": 3.0},
                  marker_size=70,
                  show=False,
                  ax=ax2
                  )

    ax2.set_ylabel("GCNN Predicted $\Delta \it{E} $(eV)", fontsize=20, color=marker_color)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(2)
        if axis in ['left']:
            ax2.spines[axis].set_color(ridge_color)
        elif axis in ['right']:
            ax2.spines[axis].set_color(marker_color)

    # increase tick width
    ax2.yaxis.set_tick_params(width=2, color=marker_color)
    ax2.xaxis.set_tick_params(width=2, color="black")
    ax2.tick_params(axis='y', labelsize=18, color=marker_color)
    if save:
        plt.savefig(os.path.join(os.getcwd(), "results", title), dpi=600, bbox_inches="tight")

    if show:
        plt.show()

    return
