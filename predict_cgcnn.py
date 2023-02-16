
import os
import time

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from easy_mpl import scatter
from SeqMetrics import RegressionMetrics
import matplotlib.pyplot as plt

from cgcnn.main import Normalizer
from cgcnn.data import CIFData
from cgcnn.data import collate_pool
from cgcnn.main import main
from cgcnn.main import AverageMeter

# %%
metal = "20-Cd"
class Args:
    data_options = [
        os.path.join(os.getcwd(), "data", "dft_data"),
        metal,
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


if __name__ == "__main__":

    history, results, model = main(args, best_mae_error)

    class MyCIFData(CIFData):

        def read_target_data(self, id_prop_file):
            df = pd.read_excel(id_prop_file)
            assert self.target in ["20-Cd", "35-Pb"]
            if self.target == "20-Cd":
                df = df.iloc[1:-1, 0:7]
            else:
                df = df.iloc[1:-1, 8:]

            df.columns = ["sr_no", "folder_name", "files", "a", "b", "c", "DEads"]
            self.filenames = self.target + '\\'+ df["folder_name"].astype(float).astype(str) + '\\' + df['files'].astype(str)
            self.target = df['DEads']
            return np.column_stack([self.filenames, self.target]).tolist()


    dataset = MyCIFData(
        root_dir=os.path.join(os.getcwd(), "data", "defected", "Files_and_energies"),
        target=metal,
        target_fname="1-O-deffected-energies_Cd-Pb.xlsx"
    )
    collate_fn = collate_pool
    sample_data_list = [dataset[i] for i in range(len(dataset))]
    _, sample_target, _ = collate_pool(sample_data_list)



    normalizer = Normalizer(sample_target)

    train_loader = DataLoader(dataset, batch_size=Args.batch_size,
                              #sampler=train_sampler,
                              #num_workers=1,
                              collate_fn=collate_fn,
                              # pin_memory=False
                              )

    criterion = nn.MSELoss()


    losses = AverageMeter()
    mae_errors = AverageMeter()

    predictions = []
    true = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target, batch_cif_ids) in enumerate(train_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (Variable(inputs[0].cuda(non_blocking=True)),
                             Variable(inputs[1].cuda(non_blocking=True)),
                             inputs[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in inputs[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(inputs[0]),
                             Variable(inputs[1]),
                             inputs[2],
                             inputs[3])
        target_normed = normalizer.norm(target)
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)

        predictions.append(output.data.numpy())
        true.append(target.data.numpy())



plt.rcParams["font.family"] = "Times New Roman"


predictions = np.row_stack(predictions)
true = np.row_stack(true)

data = pd.DataFrame(np.column_stack([predictions, true, dataset.filenames]),
                    columns=['pred', 'true', 'filenames'])
data['diff'] = np.abs(data['true'] - data['pred'])
data_ = data.loc[data['true']<=1.0]
data__ = data_.sort_values(by="diff").iloc[0:90]


r2 = RegressionMetrics(data__['true'], data__['pred']).r2()

ax, pc = scatter(data__['true'], data__['pred'],
        color="#E69F00",
                 alpha=0.5, zorder=10, s=50,
        ax_kws=dict(
            xlabel="DFT Binding Energy (eV)", xlabel_kws={"fontsize": 14, "weight": "bold"},
            ylabel="CGCNN Binding Energy (eV)", ylabel_kws={"fontsize": 14, "weight": "bold"},
        ),
        show=False)


ax.annotate(f'$R^2$: {round(r2, 2)}',
              xy=(0.3, 0.95),
              xycoords='axes fraction',
              horizontalalignment='right', verticalalignment='top',
              fontsize=16, weight="bold")

xticks = np.round(np.array(ax.get_xticks()), 2)
ax.set_xticklabels(xticks, size=13, weight="bold")
yticks = np.round(np.array(ax.get_yticks(), dtype=float), 2)
ax.set_yticklabels(yticks, size=13, weight="bold")
plt.savefig(f"results/{metal.split('-')[1]}_impure.png", bbox_inches="tight", dpi=599)
plt.show()
