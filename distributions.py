import numpy as np
import pandas as pd

import site
site.addsitedir("D:\\mytools\\easy_mpl")
from easy_mpl import ridge, hist
import seaborn as sns
import matplotlib.pyplot as plt

from surface import CoordsReader

reader = CoordsReader(
    heavy_metal="35-Pb",
    target_dpath=r'D:\collaborations\zeeshan\dft\data\Cr_Hg_Pb_Cd_data',
)

lead = reader.read_target_data(cutoff=(-5, 5.0))["DEads"]

reader = CoordsReader(
    heavy_metal="30-Hg",
    target_dpath='D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_Cd_data',
)
mercury = reader.read_target_data(cutoff=(-5, 5.0))["DEads"]

reader = CoordsReader(
    heavy_metal="04-Cr",
    target_dpath='D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_Cd_data',
)
chromium = reader.read_target_data(cutoff=(-5, 5.0))["DEads"]

max_len = max([len(lead), len(mercury), len(chromium)])

hg_diff = max_len - len(mercury)
dummy_hg = pd.Series([mercury.mean() for _ in range(hg_diff)])
hg = pd.concat([mercury, dummy_hg], axis=0)

cr_diff = max_len - len(chromium)
dummy_cr = pd.Series([chromium.mean() for _ in range(cr_diff)])
cr = pd.concat([chromium, dummy_cr], axis=0)


data = pd.DataFrame(np.column_stack((lead, hg, cr)),
                     columns=["Pb", "Hg", "Cr"])

ridge(data)

ax = sns.violinplot(data, palette="Blues")
ax.set_xticklabels(["Pb", "Hg", "Cr"], fontsize=14)
ax.set_ylabel("Binding Energy", fontsize=16)
plt.show()

# ax = sns.swarmplot(data, palette="deep")
# ax.set_xticklabels(["Pb", "Hg", "Cr"], fontsize=14)
# ax.set_ylabel("Binding Energy", fontsize=16)
# plt.show()


ax = sns.boxplot(data, palette="Blues")
ax.set_xticklabels(["Pb", "Hg", "Cr"], fontsize=14)
ax.set_ylabel("Binding Energy", fontsize=16)
plt.show()

# _, ax = plt.subplots(figsize=(10, 6))
# sns.lineplot(data, ax=ax)
# plt.show().
