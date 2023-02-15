import os
import site
site.addsitedir("D:\\mytools\\easy_mpl")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from easy_mpl import ridge
from easy_mpl._ridge import RIDGE_CMAPS

# %%
# Residuals
# ==========

f1 = np.array([243, 198, 193])/255
f2 = np.array([179, 214, 234])/255

l1 = np.array([218, 158, 146])/255
l2 = np.array([79, 176, 223])/255

fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
df_test_pb = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_pb = df_test_pb['true_Activation Energy'].values
p_test_pb = df_test_pb['pred_Activation Energy'].values
r_test_pb = t_test_pb - p_test_pb

fpath = os.path.join(os.getcwd(), "results", "20221016_092247_Hg", "Activation Energy")
df_test_hg = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_hg = df_test_hg['true_Activation Energy'].values
p_test_hg = df_test_hg['pred_Activation Energy'].values
r_test_hg = t_test_hg - p_test_hg

# reading Cr data
fpath = os.path.join(os.getcwd(), "results", "20221016_132423_Cr", "Activation Energy")
df_test_cr = pd.read_csv(os.path.join(fpath, "test_Activation Energy_0.csv"))
t_test_cr = df_test_cr['true_Activation Energy'].values
p_test_cr = df_test_cr['pred_Activation Energy'].values
r_test_cr = t_test_cr - p_test_cr

# reading Pb data
fpath = os.path.join(os.getcwd(), "results", "20221015_180126_pb", "Activation Energy")
df_train_pb = pd.read_csv(os.path.join(fpath, "training_Activation Energy_0.csv"))
t_train_pb = df_train_pb['true_Activation Energy'].values
p_train_pb = df_train_pb['pred_Activation Energy'].values
r_train_pb = t_train_pb - p_train_pb

# reading Hg data
fpath = os.path.join(os.getcwd(), "results", "20221016_092247_Hg", "Activation Energy")
df_train_hg = pd.read_csv(os.path.join(fpath, "training_Activation Energy_0.csv"))
t_train_hg = df_train_hg['true_Activation Energy'].values
p_train_hg = df_train_hg['pred_Activation Energy'].values
r_train_hg = t_train_hg - p_train_hg

# reading Cr data
fpath = os.path.join(os.getcwd(), "results", "20221016_132423_Cr", "Activation Energy")
df_train_cr = pd.read_csv(os.path.join(fpath, "training_Activation Energy_0.csv"))
t_train_cr = df_train_cr['true_Activation Energy'].values
p_train_cr = df_train_cr['pred_Activation Energy'].values
r_train_cr = t_train_cr - p_train_cr

plt.close('all')
ridge(r_train_pb,
    cmap=[f2, f2, f2],
    line_color = [l2, l2],
      line_width=2.0,
      title=f"Pb Residual (Training)")

plt.close('all')
ridge(r_test_pb,
      cmap=[f1, f1],
      line_color=[l1, l1],
      line_width=2.0,
      title="Pb Residual (Test) ")


plt.close('all')
ridge(r_train_hg,
    cmap=[f2, f2, f2],
    line_color = [l2, l2],
      line_width=2.0,
      title=f"Hg Residual (Training)")

plt.close('all')
ridge(r_test_hg,
      cmap=[f1, f1],
      line_color=[l1, l1],
      line_width=2.0,
      title="Hg Residual (Test) ")

plt.close('all')
ridge(r_train_cr,
      cmap=[f2, f2],
      line_color=[l2, l2],
      line_width=2.0,
      title="Cr Residual (Training) ")

plt.close('all')
ridge(r_test_cr,
      cmap=[f1, f1],
      line_color=[l1, l1],
      line_width=2.0,
      title="Cr Residual (Test) ")

# %%
# Predictions
# ============
plt.close('all')
ridge(
    t_train_pb,
    cmap=[f1],
    line_color = [l1],
    line_width = 2.0,
    title= "Pb (Training)",
)

plt.close('all')
ridge(
    t_test_pb,
    cmap=[f2],
    line_color = [l2],
    line_width = 2.0,
    title= "Pb (Test)",
)

plt.close('all')
ridge(
    pd.DataFrame(np.column_stack([t_train_pb, p_train_pb]), columns=["True", "Prediction"]),
    cmap=[f1, f2],
    line_color = [l1, l2],
    line_width = 2.0,
    title= "Pb",
    fill_kws = {"alpha": 0.5}
)


# %%
# Hg

plt.close('all')
ridge(
    t_train_hg,
    cmap=[f1],
    line_color = [l1],
    line_width = 2.0,
    title= "Hg (Training)",
)

plt.close('all')
ridge(
    t_test_hg,
    cmap=[f2],
    line_color = [l2],
    line_width = 2.0,
    title= "Hg (Test)",
)

plt.close('all')
ridge(
    pd.DataFrame(np.column_stack([t_train_hg, p_train_hg]), columns=["True", "Prediction"]),
    cmap=[f1, f2],
    line_color = [l1, l2],
    line_width = 2.0,
    title= "Hg",
    fill_kws = {"alpha": 0.5}
)

# %%
# Cr

plt.close('all')
ridge(
    t_train_cr,
    cmap=[f1],
    line_color = [l1],
    line_width = 2.0,
    title= "Cr (Training)",
)

plt.close('all')
ridge(
    t_test_cr,
    cmap=[f2],
    line_color = [l2],
    line_width = 2.0,
    title= "Cr (Test)",
)

plt.close('all')
ridge(
    pd.DataFrame(np.column_stack([t_train_cr, p_train_cr]), columns=["True", "Prediction"]),
    cmap=[f1, f2],
    line_color = [l1, l2],
    line_width = 2.0,
    title= "Cr",
    fill_kws = {"alpha": 0.5}
)