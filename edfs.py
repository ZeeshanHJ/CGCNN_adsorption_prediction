import os
import site
site.addsitedir("D:\\mytools\\easy_mpl")
site.addsitedir("D:\\mytools\\AI4Water")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from ai4water.utils.visualizations import edf_plot

_, ax = plt.subplots()

fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
df_test_pb = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_pb = df_test_pb['true_Activation Energy'].values
p_test_pb = df_test_pb['pred_Activation Energy'].values
r_test_pb = np.abs(t_test_pb - p_test_pb)
edf_plot(r_test_pb,
        c= np.array([200, 49, 40])/255,
         show=False, label="Pb", ax=ax)

# fpath = os.path.join(os.getcwd(), "results", "20221016_092247_Hg", "Activation Energy")
# df_test_hg = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
# t_test_hg = df_test_hg['true_Activation Energy'].values
# p_test_hg = df_test_hg['pred_Activation Energy'].values
# r_test_hg = np.abs(t_test_hg - p_test_hg)
# ax = edf_plot(r_test_hg,
#               show=False, label="Hg", ax=ax)

# fpath = os.path.join(os.getcwd(), "results", "20221016_132423_Cr", "Activation Energy")
# df_test_cr = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
# t_test_cr = df_test_cr['true_Activation Energy'].values
# p_test_cr = df_test_cr['pred_Activation Energy'].values
# r_test_cr = np.abs(t_test_cr - p_test_cr)
# ax = edf_plot(r_test_cr, xlabel="Absolute Error",
#               show=False, label="Cr Test", ax=ax)

fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_test_cd = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_cd = df_test_cd['true_Activation Energy'].values
p_test_cd = df_test_cd['pred_Activation Energy'].values
r_test_cd = np.abs(t_test_cd - p_test_cd)
ax: plt.Axes = edf_plot(r_test_cd,
              show=False, label="Cd", ax=ax)

xlim = ax.get_xlim()
xticks = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_xticklabels(xticks, fontsize=14, fontweight="bold")

ax.set_yticklabels(xticks, fontsize=14, fontweight="bold")

ax.legend(prop={"weight": "bold", "size": 13})

ax.set_xlabel("Absolute Error (%)", fontsize=16, fontweight='bold')
ax.set_ylabel("Cumulative Probability Error", fontsize=16, fontweight='bold')
ax.set_title("Empirical Distribution Plot", fontsize=16, fontweight="bold")

ax.grid(True)
plt.savefig(os.path.join(os.getcwd(), "results", "edf.png"),
            bbox_inches="tight", dpi=600)
plt.show()