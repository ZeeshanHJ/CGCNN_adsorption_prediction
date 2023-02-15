"""
==================================
Expected cummulative function
==================================
"""
import os
import site
site.addsitedir("D:\\mytools\\easy_mpl")
site.addsitedir("D:\\mytools\\AI4Water")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

from easy_mpl import plot

cd_fpath = os.path.join(os.getcwd(), "results", "20221027_104031_Cd", "Activation Energy")
df_cd_test = pd.read_csv(os.path.join(cd_fpath, "test_Activation Energy_0.csv"))
t_test_cd = df_cd_test['true_Activation Energy'].values
p_test_cd = df_cd_test['pred_Activation Energy'].values

# from statsmodels.distributions.empirical_distribution import ECDF
# ecdf_t = ECDF(t_test_cd)
# ecdf_p = ECDF(p_test_cd)
# #plot(ecdf.x, ecdf.y)
# plot(ecdf_t.x, ecdf_p.x)


def plot_ecdf(x, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    if isinstance(x, pd.Series):
        _name = x.name
        x = x.values
    else:
        assert isinstance(x, np.ndarray)
        _name = "ecdf"

    x, y = ecdf(x)
    ax.plot(x, y, label=_name, **kwargs)
    ax.legend()

    return ax


def ecdf(x: np.ndarray):
    # https://stackoverflow.com/a/37660583/5982232
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))

    return xs, ys


# ax = plot_ecdf(t_test_cd)
# plot_ecdf(p_test_cd, ax=ax)
# plt.show()


from ai4water import Model
from ai4water.datasets import busan_beach

model = Model(model="XGBRegressor", seed=2987,
              val_fraction=0.0, #x_transformation="minmax",
              #y_transformation="log"
              )
model.fit(data=busan_beach())

t,p = model.predict_on_test_data(data=busan_beach(), return_true=True)
ecdf_t = ECDF(t.reshape(-1,))
ecdf_p = ECDF(p)
#plot(ecdf.x, ecdf.y)
plot(ecdf_t.y, ecdf_p.y)