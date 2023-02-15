
import os

import site
site.addsitedir("D:\\mytools\\AI4Water")
site.addsitedir("D:\\mytools\\easy_mpl")

import numpy as np
import matplotlib.pyplot as plt
import torch

from easy_mpl import plot

from ai4water.postprocessing import ProcessPredictions
from ai4water.utils.utils import dateandtime_now

from cgcnn.main import main

class Args:
    data_options = [
        os.path.join(os.getcwd(), "data", 'Cr_Hg_Pb_Cd_data'),
        "20-Cd",
    ]
    task = 'regression'
    disable_cuda = True
    workers = 0
    epochs = 100
    start_epoch = 0
    batch_size = 16
    learning_rate = 0.001
    lr = 0.01
    lr_milestones = [100]
    momentum = 0.9
    weight_decay = 0
    print_freq = 10
    resume = ''
    train_ratio = None
    train_size = 1000
    val_ratio = 0.1
    val_size = 300
    test_ratio = 0.1
    test_size = 680
    optim = 'SGD'
    atom_fea_len = 64
    h_fea_len = 128
    n_conv = 3
    n_h = 1
    cuda = torch.cuda.is_available()


args = Args()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

history, results = main(args, best_mae_error)


fig, ax = plt.subplots()
ax = plot(history['train_mae'], label="Training", ax=ax, show=False, title="MAE")
plot(history['val_mae'], label="Validation", ax=ax, xlabel="Epochs")

_, ax = plt.subplots()
ax = plot(history['train_loss'], label="Training", ax=ax, show=False, title="LOSS")
plot(history['val_loss'], label="Validation", ax=ax, xlabel="Epochs")

prefix = f"{dateandtime_now()}"

rpath = os.path.join(os.getcwd(), "results", prefix)

if not os.path.exists(rpath):
    os.makedirs(rpath)

pp = ProcessPredictions(
    mode='regression',
    output_features=['Activation Energy'],
    forecast_len=1,
    save=True,
    path=rpath,
    plots=['regression', 'prediction', 'residual', 'edf']
)

pp(np.array(results['train_true']),
   np.array(results['train_pred']),
   "all",
   prefix="training")

pp(np.array(results['test_true']),
   np.array(results['test_pred']),
   "all",
   prefix="test")

