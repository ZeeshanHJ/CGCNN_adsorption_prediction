
import os
import site

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

import numpy as np

site.addsitedir("D:\\mytools\\easy_mpl")

import pandas as pd

from easy_mpl import ridge, hist

cd_fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_cd_test = pd.read_csv(os.path.join(cd_fpath, "test_Activation Energy_0.csv"))
t_test_cd = df_cd_test['true_Activation Energy'].values
cd_fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_cd_training = pd.read_csv(os.path.join(cd_fpath, "training_Activation Energy_0.csv"))
t_train_cd = df_cd_training['true_Activation Energy'].values
cd = np.concatenate([t_test_cd, t_train_cd])

pb_fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
df_pb_test = pd.read_csv(os.path.join(pb_fpath, "test_Activation Energy_0.csv"))
t_test_pb = df_pb_test['true_Activation Energy'].values
df_pb_training = pd.read_csv(os.path.join(pb_fpath, "training_Activation Energy_0.csv"))
t_train_pb = df_pb_training['true_Activation Energy'].values
pb = np.concatenate([t_test_pb, t_train_pb])

# hist([cd, pb], share_axes=False, labels=["Cd", "Pb"])
#
#
# ax = ridge([cd, pb], labels=["Cd", "Pb"], share_axes=True, show=False,
#            fill_kws={"alpha": 0.5})
# ax[0].legend(fontsize=20)
# plt.show()


from seaborn import histplot
a = np.where(cd<3.0, cd, -1.0)
b = pd.Series(a)
ax = histplot(x=b)
ax.set_ylabel("Counts", fontsize=20, weight="bold")
ax.set_xlabel("DFT Calculated $\Delta \it{E} $(eV)", fontsize=20,
              weight="bold")
ax.tick_params(axis='both', labelsize=16)
ax.set_xticklabels(ax.get_xticks(), weight='bold', size=16)
ax.set_yticklabels(ax.get_yticks(), weight='bold', size=16)
ax.set_title("Cd")
plt.tight_layout()
plt.show()

a = np.where(pb<3.0, pb, -1.0)
b = pd.Series(a)
ax = histplot(x=b)
ax.set_ylabel("Counts", fontsize=20, weight="bold")
ax.set_xlabel("DFT Calculated $\Delta \it{E} $(eV)", fontsize=20,
              weight="bold")
ax.tick_params(axis='both', labelsize=16)
ax.set_xticklabels(ax.get_xticks(), weight='bold', size=16)
ax.set_yticklabels(ax.get_yticks(), weight='bold', size=16)
ax.set_title("Pb")
plt.tight_layout()
plt.show()
