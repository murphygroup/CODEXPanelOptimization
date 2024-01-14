import numpy as np
import tifffile
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
# from fnet.transforms_2 import normalize as normalize2
from fnet.transforms import normalize, normalize2, normalize3
from tqdm import tqdm   
import argparse

def main(): 
    print("job start!")
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv_path', type=str, default='train_path.csv', help='path of csv file that store the path of the images')
    parser.add_argument('--num_biomarkers', type=int, default=29, help='the number of interested biomarkers/channels in the panel')
    parser.add_argument('--save_path', type=str, default='mae_mtx', help='path of saveing the output matrix')
    parser.add_argument('--seed', type=int, default = 29, help='random seed')
    parser.add_argument('--transform_signal', nargs='+', default=[], help='list of transforms on Dataset signal; normalze2 for lymph node and spleen, normalize for large/small intestine, normalize3 for basel breast cancer dataset')
    parser.add_argument('--transform_target', nargs='+', default=[], help='list of transforms on Dataset target')

    opts = parser.parse_args()
    print("load arguments successful!")

    # Set random seed
    if opts.seed is not None:
        seed = opts.seed
        np.random.seed(seed)
      
    imgs = []
    df = pd.read_csv(opts.csv_path)
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    for i in tqdm(range(len(df))):
        im = tifffile.imread(df.iloc[i, :]['path']).astype(np.float32)
        for t in transform_signal:
            im = t(im)
        imgs.append(im)
    print("Load images successful!")

    n = opts.num_biomarkers
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
     
    np.save(opts.save_path, scores)
    print("Job complete!") 


if __name__ == '__main__':
    main()