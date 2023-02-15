import os
import site
site.addsitedir("D:\\mytools\\easy_mpl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from easy_mpl._violin import violin_plot
from easy_mpl import boxplot


fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
df_test_pb = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_pb = df_test_pb['true_Activation Energy'].values
p_test_pb = df_test_pb['pred_Activation Energy'].values

# axes = violin_plot(p_test_pb,
#                    scatter_kws={"s": 2, 'alpha': 1., 'edgecolors': None},
#             show_boxplot=False,
#             show_datapoints=True,
#             show=False,
#             )
# axes.set_xlim(-1.5, 1.5)
# axes.set_title("Pb Prediction")
# plt.show()


fpath = os.path.join(os.getcwd(), "results", "20221016_092247_Hg", "Activation Energy")
df_test_hg = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_hg = df_test_hg['true_Activation Energy'].values
p_test_hg = df_test_hg['pred_Activation Energy'].values

# axes = violin_plot(p_test_hg,
#             scatter_kws={"s": 2, 'alpha': 0.5, 'edgecolors': None},
#             show_boxplot=False,
#             show_datapoints=True,
#             show=False,
#             # violin_kws = {"showextrema": True}
#             )
# axes.set_xlim(-1.5, 1.5)
# axes.set_ylim(-3, 13)
# axes.set_title("Hg Prediction")
# plt.show()


fpath = os.path.join(os.getcwd(), "results", "20221016_132423_Cr", "Activation Energy")
df_test_cr = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_cr = df_test_hg['true_Activation Energy'].values
p_test_cr = df_test_hg['pred_Activation Energy'].values

# axes = violin_plot(p_test_cr,
#             scatter_kws={"s": 2, 'alpha': 0.5, 'edgecolors': None},
#             show_boxplot=False,
#             show_datapoints=True,
#             show=False,
#                    #violin_kws = {"showextrema": True}
#             )
# axes.set_xlim(-1.5, 1.5)
# axes.set_ylim(-3, 13)
# axes.set_title("Cr Prediction")
# plt.show()

fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_test_cd = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_cd = df_test_cd['true_Activation Energy'].values
p_test_cd = df_test_cd['pred_Activation Energy'].values

# axes = violin_plot(p_test_cd,
#             scatter_kws={"s": 2, 'alpha': 0.5, 'edgecolors': None},
#             show_boxplot=False,
#             show_datapoints=True,
#             show=False,
#                    #violin_kws = {"showextrema": True}
#             )
# axes.set_xlim(-1.5, 1.5)
# axes.set_ylim(-3, 13)
# axes.set_title("Cd Prediction")
# plt.show()

_, ax = plt.subplots()
axes = violin_plot([p_test_pb, #p_test_hg,
                    p_test_cd],
            scatter_kws={"s": 12, 'alpha': 0.5, 'edgecolors': None, 'linewidths': 0.2},
            show_boxplot=False,
            show_datapoints=True,
            show=False,
            fill=False,
            fill_colors=[np.array([253, 160, 231]) / 255,
                         #np.array([102, 217, 191]) / 255,
                         np.array([251, 173, 167]) / 255],
            datapoints_colors = ['seagreen',
                                 #np.array([237, 187, 147]) / 255,
                                np.array([197, 194, 218])/255
                                ],
            # violin_kws = {"showextrema": True}
        label_violin= True,
                   ax=ax,
            )
axes.set_xlim(-0.5, 1.5)
axes.set_ylim(-4.5, 13)
axes.set_title("Prediction Comparison", fontsize=16, fontweight="bold")

yticks = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
axes.set_yticks([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
axes.set_yticklabels(yticks, fontsize=14, fontweight="bold")

axes.set_xticks(range(2))
axes.set_xticklabels(["Pb", #"Hg",
                      "Cd"], size=14, ha="center", ma="center", fontweight="bold")
axes.set_facecolor("#fbf9f4")
axes.set_ylabel("Binding Energy", fontsize=16, fontweight="bold")
plt.savefig(os.path.join(os.getcwd(), "results", "violin_1124.png"),
            bbox_inches="tight", dpi=600)
plt.show()

# axes, _ = boxplot(
#     [p_test_pb, p_test_hg, p_test_cd],
#     show=False
# )
# #axes.set_xlim(-0.5, 2.5)
# axes.set_ylim(-4.5, 13)
# axes.set_title("Prediction Comparison", fontsize=20)
# axes.set_xticks(range(4))
# axes.set_xticklabels(["", "Pb", "Hg", "Cd"], size=18, ha="center", ma="center")
# axes.set_facecolor("#fbf9f4")
# axes.set_ylabel("Binding Energy", fontsize=18)
# plt.show()
