
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

from easy_mpl import plot


def modify_axes_width(axes):
    # increase tick width
    axes.tick_params(width=1.5)

    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(1.5)
    return


def modify_yticklabels(axes):
    yticks = maybe_change_ticks_dtype(axes.get_yticks())
    axes.set_yticklabels(yticks, size=17, weight='bold')
    return


def maybe_change_ticks_dtype(ticks):
    if 'float' in ticks.dtype.name:
        ticks = np.round(ticks, 2)
    else:
        ticks = ticks.astype(int)
    return ticks


ind = [1, 2, 3]
val1 = [0, -1.809, 0]
val2 = [0, -1.72, 0]

plot(ind, val1, '.', show=False, figsize=(4, 6))
ax = plot(ind, val2, '.', show=False)
ax.set_xticks([])
ax.tick_params('y', labelsize=15)
modify_axes_width(ax)
modify_yticklabels(ax)
plt.savefig("results/jali.png", dpi=600, bbox_inches="tight")
plt.tight_layout()
plt.show()