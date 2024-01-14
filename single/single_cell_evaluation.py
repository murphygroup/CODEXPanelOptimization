import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tifffile import imread
import pandas as pd
import os


###
# This script is used to generate single-cell profiles for each cell

def singel_cell_evaluation(test_path, mask_dir, synth_mp_dir, synth_single_dir, save_dir):
    """
    This function is used to generate single-cell profiles for each cell

    test_path: the path of the csv file containing the paths of the test images
    mask_dir: the directory of the masks
    synth_mp_dir: the directory of the synthetic images in multi-panel experiment
    synth_single_dir: the directory of the synthetic images in single-panel experiment
    save_dir: the directory to save the profiles
    """

    csv_path = test_path
    paths = pd.read_csv(csv_path)

    for q, p in enumerate(paths["path"]):
        cell_mask = imread(np.unique(os.path.join((mask_dir, "im{}_mask.tiff")).format(q + 1)))
        n_cell = len(cell_mask)


        im1 = imread(p).astype(np.float32)
        print(1)
        im3 = imread(os.path.join(synth_mp_dir, "{}_synth.tiff".format(q))).astype(np.float32)
        print(2)
        im2 = imread(os.path.join(synth_single_dir, "{}_synth.tiff".format(q))).astype(np.float32)
        print(3)
        print(n_cell)
        cell_profile_1 = np.zeros((n_cell - 1, im1.shape[0]))
        cell_profile_2 = np.zeros((n_cell - 1, im2.shape[0]))
        cell_profile_3 = np.zeros((n_cell - 1, im3.shape[0]))

        cov_profile_1 = []
        cov_profile_2 = []
        cov_profile_3 = []


        for id_ in range(1, n_cell):
            coord = np.argwhere(cell_mask == id_)
            if len(coord) == 0:
                continue
                
            # intensities = np.mean(expression[:, coord[:, 0], coord[:, 1]], axis=1)   # sum, cov
            cell_profile_1[id_ - 1] = np.mean(im1[:, coord[:, 0], coord[:, 1]], axis=1)  
            cell_profile_2[id_ - 1] = np.mean(im2[:, coord[:, 0], coord[:, 1]], axis=1)
            cell_profile_3[id_ - 1] = np.mean(im3[:, coord[:, 0], coord[:, 1]], axis=1)

            temp = []
            for i in range(im1.shape[0]):
                for j in range(im1.shape[0]):
                    
                    if i < j:
                        temp.append(stats.pearsonr(im1[i, coord[:, 0], coord[:, 1]], im1[j, coord[:, 0], coord[:, 1]])[0])
            cov_profile_1.append(temp)
            
            temp = []
            for i in range(im2.shape[0]):
                for j in range(im2.shape[0]):
                    
                    if i < j:
                        temp.append(stats.pearsonr(im2[i, coord[:, 0], coord[:, 1]], im2[j, coord[:, 0], coord[:, 1]])[0])
            cov_profile_2.append(temp)
            
            temp = []
            for i in range(im3.shape[0]):
                for j in range(im3.shape[0]):
                    
                    if i < j:
                        temp.append(stats.pearsonr(im3[i, coord[:, 0], coord[:, 1]], im3[j, coord[:, 0], coord[:, 1]])[0])
            cov_profile_3.append(temp)
        cov_profile_1 = np.array(cov_profile_1)
        cov_profile_2 = np.array(cov_profile_2)
        cov_profile_3 = np.array(cov_profile_3)

        # save
        np.save(os.path.join(save_dir, "{}_cell_profile_3.npy".format(q)), cell_profile_3)
        np.save(os.path.join(save_dir, "{}_cell_profile_2.npy".format(q)), cell_profile_2)
        np.save(os.path.join(save_dir, "{}_cell_profile_1.npy".format(q)), cell_profile_1)

        
        np.save(os.path.join(save_dir, "{}_cov_profile_3.npy".format(q)), cov_profile_3)
        np.save(os.path.join(save_dir, "{}_cov_profile_2.npy".format(q)), cov_profile_2)
        np.save(os.path.join(save_dir, "{}_cov_profile_1.npy".format(q)), cov_profile_1)


if __name__ == "__main__":
    # small intestine
    test_path = r"/home/huangqis/small_intestine_images/test_path.csv"
    mask_dir = r"..data/small_intestine/test_masks"
    synth_mp_dir = "../mp/small_intestine_output"
    synth_single_dir = "small_intestine_output/imgs"
    save_dir = "small_intestine_output"

    singel_cell_evaluation(test_path, mask_dir, synth_mp_dir, synth_single_dir, save_dir)
