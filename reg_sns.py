
import os

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def maybe_change_ticks_dtype(ticks):
    if 'float' in ticks.dtype.name:
        ticks = np.round(ticks, 2)
    else:
        ticks = ticks.astype(int)
    return ticks


def modify_xticklabels(axes):
    xticks = maybe_change_ticks_dtype(axes.get_xticks())
    axes.set_xticklabels(xticks, size=12, weight='bold')
    return


def modify_yticklabels(axes):
    yticks = maybe_change_ticks_dtype(axes.get_yticks())
    axes.set_yticklabels(yticks, size=12, weight='bold')
    return


def modify_ticklabels(axes):
    modify_xticklabels(axes)
    modify_yticklabels(axes)
    modify_axes_width(axes)
    return


def modify_axes_width(axes):
    # increase tick width
    axes.tick_params(width=2)

    # change all spines
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(2)
    return


def modify_grid_axes_width(gridspec):
    modify_axes_width(gridspec.ax_marg_x)
    modify_axes_width(gridspec.ax_marg_y)
    return


def set_axes_labels(axes):
    axes.set_ylabel(ylabel='CGCNN predicted $\Delta $ E(eV)', fontsize=15, weight='bold')
    axes.set_xlabel(xlabel='DFT predicted $\Delta $ E(eV)', fontsize=15, weight='bold')
    return


def modify_axes_labels(axes):
    modify_ticklabels(axes)
    set_axes_labels(axes)
    return


def modify_legend(axes):
    legend_properties = {'weight': 'bold',
                         'size': 14, }
    axes.legend(prop=legend_properties)
    return


# %%
# Cd Test
# ============
cd_fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_cd_test = pd.read_csv(os.path.join(cd_fpath, "test_Activation Energy_0.csv"),
                         index_col="index")
scaler = MinMaxScaler(feature_range=(df_cd_test['true_Activation Energy'].min(), 4.0))
df_cd_test['true_Activation Energy'] = scaler.fit_transform(
    df_cd_test['true_Activation Energy'].values.reshape(-1,1))
scaler = MinMaxScaler(feature_range=(df_cd_test['pred_Activation Energy'].min(), 4.0))
df_cd_test['pred_Activation Energy'] = scaler.fit_transform(
    df_cd_test['pred_Activation Energy'].values.reshape(-1,1))

grid = sns.jointplot(data=df_cd_test, x='true_Activation Energy',
              y='pred_Activation Energy',
              kind="reg",
              color='darkcyan',
              scatter_kws = {'alpha': 0.5, 'color': 'olive',
                             'edgecolors':'black', 'linewidth':0.5},
              line_kws = {'alpha': 0.5, 'color': 'k'},
              )
ax = grid.ax_joint
modify_axes_labels(ax)
modify_grid_axes_width(grid)
#ax.text(0.1, 4.0, s="$R^2$ 0.99", fontsize=15, weight="bold")
plt.savefig("results/cd_test.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
# Cd Training
# ============
df_cd_training = pd.read_csv(os.path.join(cd_fpath, "training_Activation Energy_0.csv"),
                             index_col="index")
scaler = MinMaxScaler(feature_range=(df_cd_training['true_Activation Energy'].min(), 4.0))
df_cd_training['true_Activation Energy'] = scaler.fit_transform(
    df_cd_training['true_Activation Energy'].values.reshape(-1,1))
scaler = MinMaxScaler(feature_range=(df_cd_training['pred_Activation Energy'].min(), 4.0))
df_cd_training['pred_Activation Energy'] = scaler.fit_transform(
    df_cd_training['pred_Activation Energy'].values.reshape(-1,1))

grid = sns.jointplot(data=df_cd_training, x='true_Activation Energy',
              y='pred_Activation Energy',
              kind="reg",
              color='tab:brown',
              scatter_kws = {'alpha': 0.5, 'color': 'crimson',
                             'edgecolors':'black', 'linewidth':0.5},
              line_kws = {'alpha': 0.5, 'color': 'k'},
              )
ax = grid.ax_joint
modify_axes_labels(ax)
modify_grid_axes_width(grid)
plt.savefig("results/cd_training.png", dpi=600, bbox_inches="tight")
plt.show()

#
# %%
# Cd Training and Test
# =====================
# df_cd_test['hue'] = "Training"
# df_cd_training['hue'] = "Test"
# df = pd.concat([df_cd_test, df_cd_training], axis=0)
#
# g = sns.jointplot(data=df,
#                   x="true_Activation Energy",
#                   y="pred_Activation Energy",
#                   #scatter_kws = {'alpha': 0.5, 'edgecolors': 'k'},
#                   edgecolors = 'k',
#                   hue='hue',
#                   palette='husl')
# ax = g.ax_joint
# modify_axes_labels(ax)
# modify_legend(ax)
# modify_grid_axes_width(g)
# plt.show()
#
# %%
# Pb Test
# ============

pb_fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
df_pb_test = pd.read_csv(os.path.join(pb_fpath, "test_Activation Energy_0.csv"),
                         index_col="index")
scaler = MinMaxScaler(feature_range=(df_pb_test['true_Activation Energy'].min(), 5.0))
df_pb_test['true_Activation Energy'] = scaler.fit_transform(
    df_pb_test['true_Activation Energy'].values.reshape(-1,1))
scaler = MinMaxScaler(feature_range=(df_pb_test['pred_Activation Energy'].min(), 5.0))
df_pb_test['pred_Activation Energy'] = scaler.fit_transform(
    df_pb_test['pred_Activation Energy'].values.reshape(-1,1))

grid = sns.jointplot(data=df_pb_test, x='true_Activation Energy',
              y='pred_Activation Energy',
              kind="reg",
              color='darkcyan',
              scatter_kws = {'alpha': 0.5, 'color': 'olive',
                             'edgecolors':'black', 'linewidth':0.5},
              line_kws = {'alpha': 0.5, 'color': 'k'},
              )
ax = grid.ax_joint
modify_axes_labels(ax)
modify_grid_axes_width(grid)
plt.savefig("results/pb_test.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
# Pb Training
# ============
df_pb_training = pd.read_csv(os.path.join(pb_fpath, "training_Activation Energy_0.csv"),
                             index_col="index")
scaler = MinMaxScaler(feature_range=(df_pb_training['true_Activation Energy'].min(), 5.0))
df_pb_training['true_Activation Energy'] = scaler.fit_transform(
    df_pb_training['true_Activation Energy'].values.reshape(-1,1))
scaler = MinMaxScaler(feature_range=(df_pb_training['pred_Activation Energy'].min(), 5.0))
df_pb_training['pred_Activation Energy'] = scaler.fit_transform(
    df_pb_training['pred_Activation Energy'].values.reshape(-1,1))

grid = sns.jointplot(data=df_pb_training, x='true_Activation Energy',
              y='pred_Activation Energy',
              kind="reg",
              color='tab:brown',
              scatter_kws = {'alpha': 0.5, 'color': 'crimson',
                             'edgecolors':'black', 'linewidth':0.5},
              line_kws = {'alpha': 0.5, 'color': 'k'},
              )
ax = grid.ax_joint
modify_axes_labels(ax)
modify_grid_axes_width(grid)
plt.savefig("results/pb_training.png", dpi=600, bbox_inches="tight")
plt.show()

#
# # %%
# # Pb Training and Test
# # =====================
# df_pb_test['hue'] = "Training"
# df_pb_training['hue'] = "Test"
# df = pd.concat([df_pb_test, df_pb_training], axis=0)
#
# g = sns.jointplot(data=df,
#                   x="true_Activation Energy",
#                   y="pred_Activation Energy",
#                   #scatter_kws = {'alpha': 0.5, 'edgecolors': 'k'},
#                   edgecolors = 'k',
#                   hue='hue',
#                   palette='husl')
# ax = g.ax_joint
# modify_axes_labels(ax)
# modify_legend(ax)
# modify_grid_axes_width(g)
# plt.show()



