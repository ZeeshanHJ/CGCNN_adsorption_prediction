
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from cgcnn.data_reader import CoordsReader
from umap import UMAP

reader = CoordsReader(
    heavy_metal="30-Hg",
    target_dpath='D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_Cd_data',
    save_path=r'D:\collaborations\zeeshan\dft\results\20221015_180126_pb',
)

distances = [
    "1.6", "1.7", "1.8", "1.9", "2.0",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", '2.7', '2.8', '2.9', '3.0',
    "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4.0",
    "4.1", "4.2"
]

energies = []
distances_ = []
points = 0

for idx, dist in enumerate(distances):

    xyz_fpath = os.path.join(reader.target_dpath, reader.heavy_metal, dist, 'POSCAR-all.xyz')

    c, idx_rem = reader.get_color(dist, cutoff=(-5.0, 0.0))

    print(idx, c.shape)
    points += len(c)

    if len(c)== 100:
        energies.append(c)
        distances_.append(float(dist))


# tsne = TSNE(n_components=2, random_state=313)
# ev_2D = tsne.fit_transform(np.column_stack(energies))
#
# s = plt.scatter(ev_2D[:, 0], ev_2D[:, 1], #c=y_test.reshape(-1,),
#                 cmap="Spectral",
#                 s=5)
# plt.gca().set_aspect('equal', 'datalim')
# plt.colorbar(s)
# plt.title('TSNE projection of shap values', fontsize=18)
# plt.show()
#
#
# ev_umap = UMAP(
#     n_components=2,
#     random_state=313).fit_transform(np.column_stack(energies))
# s = plt.scatter(ev_umap[:, 0], ev_umap[:, 1], #c=y_test.reshape(-1,),
#             s=5, cmap="Spectral")
# plt.gca().set_aspect('equal', 'datalim')
# cbar = plt.colorbar(s)
# #cbar.ax.set_ylabel('Predicted Adsorption Capacity', rotation=270)
# plt.title('UMAP projection of binding energies', fontsize=18)
# plt.show()