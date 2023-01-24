import numpy as np
import tifffile
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from fnet.transforms_2 import normalize
# from fnet.transforms import normalize


imgs = []
csv_path = r'/home/huangqis/HuBMAP/train_path_SP29.csv' #load training images
df = pd.read_csv(csv_path)
for i in range(len(df)):
    im = tifffile.imread(df.iloc[i, :]['path']).astype(np.float32)
    imgs.append(normalize(im))
    print(i)

n = 29
scores = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        for k in range(len(imgs)):
            if k == 0:
                a = imgs[k][i].flatten()
                b = imgs[k][j].flatten()
            else:
                a = np.concatenate((a, imgs[k][i].flatten()))
                b = np.concatenate((b, imgs[k][j].flatten()))
        # scores[i, j] = pearsonr(a, b)[0]
        scores[i, j] = mean_absolute_error(a, b)

for j in range(n):
    # scores[j, j] = 1
    scores[j, j] = 0
    for i in range(j+1, n):
        scores[i, j] = scores[j, i]
 
np.save("/home/huangqis/HuBMAP/mae_mtx_valSP29", scores)  # save initial matrix for input selection