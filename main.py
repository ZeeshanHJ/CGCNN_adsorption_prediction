
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
from scipy.interpolate import griddata
import scipy
from scipy.io import savemat



def read_target_data(id_prop_file):
    df = pd.read_csv(id_prop_file)
    df = df.dropna()
    _filenames = df['cif_file_name']
    _filenames = np.array([fname.replace('_', '-') for fname in _filenames])
    files = "Pb_cif_files\\" + df['folder_name'].astype(str) + "\\" + _filenames
    _target = df['Eads']
    return files.values, _target.values


def xyz_from_cif(cif_file):
    with open(cif_file, 'r') as fp:
        for line in fp.readlines():
            if 'Pb' in line:
                vals = line.split()
                assert len(vals) == 6
                _x, _y, _z = vals[2:5]
                return float(_x), float(_y), float(_z)


target_path = os.path.join(os.getcwd(), 'data', 'Pb1')
target_file = os.path.join(target_path, 'id_prop.csv')
filenames, target = read_target_data(target_file)

st, en = 900, 1000
xs = np.full(en-st, np.nan)
ys = np.full(en-st, np.nan)
zs = np.full(en-st, np.nan)

for idx, f in enumerate(filenames[st:en]):
    x, y, z = xyz_from_cif(os.path.join(target_path, f))
    xs[idx] = x
    ys[idx] = y
    zs[idx] = z

c = np.array(target[st:en])


# norm=matplotlib.colors.SymLogNorm(1,vmin=c.min(),vmax=c.max())
# colors=plt.cm.coolwarm(norm(c))
#
# plt.close('all')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_trisurf(xs, ys, zs, #color='white',
#                 #edgecolors='grey',
#                        linewidth=0.0, #antialiased=False,
#                 alpha=0.5,
#                        facecolors = colors, #cm.jet(c/np.amax(c))
#                        )
# #fig.colorbar(surf, shrink=0.5, aspect=5)
# sc = ax.scatter(xs, ys, zs, c=colors, s=1.0)
#
# norm = cm.colors.Normalize(np.min(c), np.max(c))
# cb = cm.ScalarMappable(norm, cmap="jet")
# fig.colorbar(cb, shrink=0.5, aspect=5, pad=0.1)
# plt.savefig(f"pb_{st}_{en}", bbox_inches="tight", dpi=600)
# plt.show()



#%%

fig = plt.figure()
ax = fig.gca(projection='3d')

xi = np.linspace(min(xs), max(xs), num=100)
yi = np.linspace(min(ys), max(ys), num=100)


X, Y = np.meshgrid(xi, yi)
Z = griddata((xs, ys), zs, (X, Y), method="cubic")
C = griddata((xs, ys), c, (X, Y), method="cubic")

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       #cmap=cm.jet(c),
                       #facecolors=cm.jet(c/np.amax(c)),
                       linewidth=1, antialiased=True)

ax.set_zlim3d(np.nanmin(Z), np.nanmax(Z))
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

plt.show()

# savemat('2.4_cubic.mat', {'x': X, 'y': Y, 'z': Z, 'c': C})


fname = r'D:\collaborations\zeeshan\dft\data\Pb1\Pb_cif_files\3.2\POSCAR-all.xyz'
xs, ys, zs = [], [], []
with open(fname, 'r') as fp:
    for idx, line in enumerate(fp.readlines()):
        if 'Pb' in line and idx>5:
            line = line.split()[1:]
            x, y, z = line
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))


xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

xi = np.linspace(min(xs), max(xs), num=100)
yi = np.linspace(min(ys), max(ys), num=100)


X, Y = np.meshgrid(xi, yi)
Z = griddata((xs, ys), zs, (X, Y), method="cubic")
C = griddata((xs, ys), c, (X, Y), method="cubic")

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       #cmap=cm.jet(c),
                       #facecolors=cm.jet(c/np.amax(c)),
                       linewidth=1, antialiased=True)

ax.set_zlim3d(np.nanmin(Z), np.nanmax(Z))
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

plt.show()
# #savemat('4.2_cubic.mat', {'x3_2': X, 'y3_2': Y, 'z3_2': Z, 'c3_2': C})