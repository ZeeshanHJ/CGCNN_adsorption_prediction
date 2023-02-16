


import os

import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.interpolate import griddata
np.set_printoptions(suppress=True)
import site
site.addsitedir("D:\\mytools\\easy_mpl")


class CoordsReader:

    def __init__(self, heavy_metal, target_dpath, save_path=""):

        self.save_path = save_path
        self.heavy_metal = heavy_metal
        self.target_dpath = target_dpath
        self.target_fpath = os.path.join(target_dpath, 'Energies(Cr_Hg_Pb_Cd).xlsx')


    def get_color(self, distance, cutoff=None):

        df = self.read_target_data()

        df = df[df['Folder_name']==distance]

        df = df.reset_index()
        index_removed = df[(df['DEads'] > cutoff[1]) | (df['DEads'] < cutoff[0])]

        df = df[(df['DEads']<=cutoff[1]) & (df['DEads']>=cutoff[0])]

        c = df['DEads'].values

        return c, index_removed.index.values

    def read_target_data(self, cutoff=None):
        df = pd.read_excel(self.target_fpath, header=1)

        if "Pb" in self.heavy_metal:
            cols = [col for col in df.columns if col.endswith(".2")]
            true_cols = [col[:-2] for col in cols]
        elif "Hg" in self.heavy_metal:
            cols = [col for col in df.columns if col.endswith(".1")]
            true_cols = [col[:-2] for col in cols]
        elif "Cr" in self.heavy_metal:
            cols = [col for col in df.columns if not col.endswith(".1") and not col.endswith(".2") and not col.endswith(".3")]
            true_cols = cols
        elif "Cd" in self.heavy_metal:
            cols = [col for col in df.columns if col.endswith(".3")]
            true_cols = [col[:-2] for col in cols]
        else:
            raise ValueError

        df = df[cols]
        df.columns = true_cols
        _filenames = df['cif_file_name']
        _filenames = np.array([fname.replace('_', '-') for fname in _filenames])
        df['Folder_name'] = df['Folder_name'].round(2).astype(str)
        df['files'] = f"{self.heavy_metal}\\" + df['Folder_name'] + "\\" + _filenames

        if cutoff:
            df = df[(df['DEads'] <= cutoff[1]) & (df['DEads'] >= cutoff[0])]

        return df[['Folder_name', 'files', 'DEads']]

    def generate_surface(self, distance:str, method="cubic",
                         cutoff=(-5, 10.0),
                         save = True
                         ):

        xyz_fpath = os.path.join(self.target_dpath, self.heavy_metal,  distance, 'POSCAR-all.xyz')

        mat_fname = distance.replace(".", "_")

        c, idx_rem = self.get_color(distance, cutoff=cutoff)

        xs, ys, zs = [], [], []

        metal_name = self.heavy_metal.split('-')[1]

        with open(xyz_fpath, 'r') as fp:
            for index, line in enumerate(fp.readlines()):
                if metal_name in line and index > 5:
                    line = line.split()[1:]
                    x, y, z = line
                    xs.append(float(x))
                    ys.append(float(y))
                    zs.append(float(z))

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        assert all([len(xs), len(ys), len(zs)]), f"{len(xs)} {len(ys)} {(len(zs))}"

        if len(idx_rem)>0:
            xs = np.delete(xs, idx_rem)
            ys = np.delete(ys, idx_rem)
            zs = np.delete(zs, idx_rem)

        xi = np.linspace(min(xs), max(xs), num=100)
        yi = np.linspace(min(ys), max(ys), num=100)

        X, Y = np.meshgrid(xi, yi)
        Z = griddata((xs, ys), zs, (X, Y), method=method)
        C = griddata((xs, ys), c, (X, Y), method=method)

        xname = f"x{mat_fname}"
        yname = f"y{mat_fname}"
        zname = f"z{mat_fname}"
        cname = f"c{mat_fname}"

        assert not mat_fname.endswith(".mat")

        if save:
            mat_fname =f'{mat_fname}_{method}.mat'
            savemat(os.path.join(self.save_path, mat_fname), {xname: X, yname: Y, zname: Z, cname: C})

        return