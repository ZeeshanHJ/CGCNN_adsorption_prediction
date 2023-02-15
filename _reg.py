
import os
import site
site.addsitedir("D:\\mytools\\easy_mpl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from easy_mpl import regplot, ridge
from utils import plot_ridge_reg


f1 = np.array([243, 198, 193])/255
f2 = np.array([179, 214, 234])/255

l1 = np.array([218, 158, 146])/255
l2 = np.array([79, 176, 223])/255

# # %%
# # Cd Test
# # ========
cd_fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_cd_test = pd.read_csv(os.path.join(cd_fpath, "test_Activation Energy_0.csv"))
t_test_cd = df_cd_test['true_Activation Energy'].values
p_test_cd = df_cd_test['pred_Activation Energy'].values
r_test_cd = t_test_cd - p_test_cd
plot_ridge_reg(t_test_cd, p_test_cd, title="Cd (Test) ",
               ylim=(0.0, 2.0), save=False, add_ridge=False)
#
# # %%
# # Cd Training
# # ============
# cd_fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
# df_cd_training = pd.read_csv(os.path.join(cd_fpath, "training_Activation Energy_0.csv"))
# t_train_cd = df_cd_training['true_Activation Energy'].values
# p_train_cd = df_cd_training['pred_Activation Energy'].values
# r_train_cd = t_train_cd - p_train_cd
# plot_ridge_reg(t_train_cd, p_train_cd, title="Cd (Training) ",
#                ridge_color=l1, marker_color="seagreen",
#                ylim=(0.0, 2.0))

# %%
# Pb Test
# ========

# fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
# df = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
# t_test_pb = df['true_Activation Energy'].values
# p_test_pb = df['pred_Activation Energy'].values
#
# r_test_pb = t_test_pb - p_test_pb
# plot_ridge_reg(t_test_pb, p_test_pb, title="Pb (Test) ")

# # %%
# # Pb Training

# fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
# df = pd.read_csv(os.path.join(fpath, "training_Activation Energy_0.csv"))
# t_train_pb = df['true_Activation Energy'].values
# p_train_pb = df['pred_Activation Energy'].values
# plot_ridge_reg(t_train_pb, p_train_pb, title="Pb (Training) ",
#                ridge_color=l1, marker_color="seagreen",
#                cut=0.07, save=False)
#
# # %%
# # Hg Test
# # =======
#
# fpath = os.path.join(os.getcwd(), "results", "20221016_092247_Hg", "Activation Energy")
# df_test_hg = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
# t_test_hg = df_test_hg['true_Activation Energy'].values
# p_test_hg = df_test_hg['pred_Activation Energy'].values
#
# plot_ridge_reg(t_test_hg, p_test_hg, title="Hg (Test) ", cut=0.12)
#
# # %%
# # Hg Training
#
# fpath = os.path.join(os.getcwd(), "results", "20221016_092247_Hg", "Activation Energy")
# df_train_hg = pd.read_csv(os.path.join(fpath, "training_Activation Energy_0.csv"))
# t_train_hg = df_train_hg['true_Activation Energy'].values
# p_train_hg = df_train_hg['pred_Activation Energy'].values
#
# plot_ridge_reg(t_train_hg, p_train_hg, title="Hg (Training) ",
#                ridge_color=l1, marker_color="seagreen",
#                cut=0.11)



# # %%
# # Cr
# # ====
#
# fpath = os.path.join(os.getcwd(), "results", "20221016_132423_Cr", "Activation Energy")
# df_test_cr = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
# t_test_cr = df_test_cr['true_Activation Energy'].values
# p_test_cr = df_test_cr['pred_Activation Energy'].values
#
# plt.close('all')
# ax = ridge(pd.DataFrame(p_test_cr, columns=['']),
#            color=["white"],
#       line_color=["purple"],
#       line_width=3.0,
#       title="Cr (Test) ",
#            show=False,
#       )
#
# ax[0].set_ylim(-0., 2.0)
# ax2 = ax[0].twinx()
# ax2 = regplot(t_test_cr, p_test_cr,
#         scatter_kws={"alpha": 0.5, 'linewidths': 0.5, 'edgecolors': 'black',
#                      "cmap": "YlGn"
#                      },
#         marker_color="skyblue",
#         line_color="black", line_style='--',
#               line_kws = {"linewidth": 3.0},
#         marker_size=70,
#         show=False,
#         ax = ax2
#         )
#
# ax2.set_xlim(-5, 7)
# fig = ax2.get_figure()
# fig.set_figwidth(10)
# fig.set_figheight(8)
# ax2.axis("off")
# # plt.savefig(os.path.join(os.getcwd(), "results", "cr_test"), dpi=600, bbox_inches="tight")
# plt.show()
#
# # %%
# # Cr Training
# fpath = os.path.join(os.getcwd(), "results", "20221016_132423_Cr", "Activation Energy")
# df = pd.read_csv(os.path.join(fpath, "training_Activation Energy_0.csv"))
# t_train_cr = df['true_Activation Energy'].values
# p_train_cr = df['pred_Activation Energy'].values
#
# plt.close('all')
# ax = ridge(pd.DataFrame(p_train_cr.copy(), columns=['']),
#       #cmap=[f1, f1],
#            color=["white"],
#       line_color=[l1, l1],
#       line_width=3.0,
#       title="Cr (Training) ",
#            show=False,
#       )
#
# ax[0].set_ylim(-0., 2.0)
# #ax[0].set_xlim(-3, 10)
# ax2 = ax[0].twinx()
# ax2 = regplot(t_train_cr, p_train_cr,
#         scatter_kws={"alpha": 0.5, 'linewidths': 0.5, 'edgecolors': 'black',
#                      "cmap": "YlGn"
#                      },
#         marker_color="seagreen",
#         line_color="black", line_style='--',
#               line_kws = {"linewidth": 3.0},
#         marker_size=70,
#         show=False,
#         ax = ax2
#         )
#
# #ax.set_ylim(-3, 12)
# ax2.set_xlim(-5, 7)
# fig = ax2.get_figure()
# fig.set_figwidth(10)
# fig.set_figheight(8)
# ax2.axis("off")
# # plt.savefig(os.path.join(os.getcwd(), "results", "cr_train"), dpi=600, bbox_inches="tight")
# plt.show()
