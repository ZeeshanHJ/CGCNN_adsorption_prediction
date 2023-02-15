
import pandas as pd
from easy_mpl import plot
import numpy as np
from cgcnn.data_reader import CoordsReader

reader = CoordsReader(
    heavy_metal = "35-Pb",
    target_dpath = 'D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_Cd_data',
    save_path = r'D:\collaborations\zeeshan\dft\results\20221015_180126_pb',
)

distances = [
    "1.6", "1.7",
    "1.8", "1.9", "2.0",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", '2.7', '2.8', '2.9', '3.0',
    "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4.0",
    "4.1", "4.2"
]

pb_mins = []
for idx, dist in enumerate(distances):

    c, idx_rem = reader.get_color(dist, cutoff=(-50.0, 5.0))

    print(dist, min(c))
    pb_mins.append(min(c))



reader = CoordsReader(
    heavy_metal = "20-Cd",
    target_dpath = 'D:\\collaborations\\zeeshan\\dft\\data\\Cr_Hg_Pb_Cd_data',
    save_path = r'D:\collaborations\zeeshan\dft\results\20221015_180126_pb',
)


cd_mins = []
for idx, dist in enumerate(distances):

    c, idx_rem = reader.get_color(dist, cutoff=(-50.0, 5.0))

    print(dist, min(c))
    cd_mins.append(min(c))

plot(np.array(distances).astype(float), cd_mins)

df = pd.DataFrame(np.column_stack([pb_mins, cd_mins]),
             columns=["Pb", "Cd"])

df.to_csv("surface_trends.csv")