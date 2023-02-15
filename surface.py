
import os

import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.interpolate import griddata
np.set_printoptions(suppress=True)
import site
site.addsitedir("D:\\mytools\\easy_mpl")
from easy_mpl import plot
from cgcnn.data_reader import CoordsReader


if __name__ == "__main__":

    reader = CoordsReader(
        heavy_metal = "35-Pb",
        target_dpath = 'D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_Cd_data',
        save_path = r'D:\collaborations\zeeshan\dft\results\20221015_180126_pb',
    )

    distances = [
        #"1.6", "1.7", "1.8", "1.9", "2.0",
        "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", '2.7', '2.8', '2.9', '3.0',
        "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4.0",
        "4.1", "4.2"
    ]

    for idx, dist in enumerate(distances):

        reader.generate_surface(
        #f'D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_data\\{heavy_metal}\\{dist}\\POSCAR-all.xyz',
            dist,
            cutoff=(-5.0, 0.0),
            save=False
        )