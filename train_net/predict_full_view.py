from turtle import shape
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from UNext import R2U_Net
from PIL import Image
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from tifffile import imread, imsave, imwrite
from ResNet_2 import ResUnet 
from UNet2 import Unet
# from fnet.transforms_2 import normalize as normalize2    
from fnet.transforms import normalize, normalize2, normalize3                    
import argparse
from tqdm import tqdm   

def main():
    print("job start!")
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_run_dir', type=str, help='base directory for saved models')
    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing test Dataset')
    parser.add_argument('--input_index', type=int, nargs='+', default=[0, 1, 2], help='input channel index')
    parser.add_argument('--target_index', type=int, nargs='+', default=[3, 4, 5], help='output channel index')

    parser.add_argument('--seed', type=int, default = 29, help='random seed')
    parser.add_argument('--transform_signal', nargs='+', default=[], help='list of transforms on Dataset signal; normalze2 for lymph node and spleen, normalize for large/small intestine')
    parser.add_argument('--transform_target', nargs='+', default=[], help='list of transforms on Dataset target')

    parser.add_argument('--save_image', action="store_true", help="store synthetic images or not") 

    opts = parser.parse_args()
    print("load arguments successful!")

    # Set random seed
    if opts.seed is not None:
        seed = opts.seed
        np.random.seed(seed)

    path = opts.path_run_dir
    path_csv = opts.path_dataset_csv
    
    # iter_ = 3600
    # state_dict = torch.load(os.path.join(path, "checkpoints/model_{:06d}.p".format(iter_)))
    state_dict = torch.load(os.path.join(path, "model_best.p"))

    scores = []
    path2 = os.path.join(path, "imgs")
    if not os.path.exists(path2):
        os.mkdir(path2)

    paths = pd.read_csv(path_csv)

    patch_size = 192
    step_size = 128

    idx = opts.input_index
    idx2 = opts.target_index


    net = ResUnet(in_ch = len(idx), output_ch = len(idx2))
    # net = nn.DataParallel(net)

    net.load_state_dict(state_dict['nn_net_state']) 

    # load test images
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]
    images = []
    for p in tqdm(paths["path"]):
        imgs = imread(p).astype(np.float32)
        for t in transform_signal:
            imgs = t(imgs)
        images.append(imgs)
    print("Load images successful!")

           
    net.eval()
    net.cuda()

    scores = []
    scores2 = []
    scores3 = []

    q = 0

    predictions = []

    for ii in range(len(images)): # for each image
        stains = ["CD107a", "CD11c", "CD15", "CD163", "CD1c", "CD20", "CD21", "CD31", "CD34", "CD35", "CD3e", 
            "CD4", "CD44", "CD45", "CD45RO", "CD5", "CD68", "CD8", "CollagenIV", "DAPI-02", "ECAD", "FoxP3",  "HLA-DR",
            "Ki67", "LYVE-1", "PanCK", "Podoplanin", "SMActin", "Vimentin"]

        imgs = images[ii]

        input_ = imgs[idx]
        target = imgs[idx2]

        # d1 = target.shape[1] // patch_size
        # d2 = target.shape[2] // patch_size

        input_ = np.expand_dims(input_, 0)

        pred = np.zeros_like(target)
        mask = np.zeros_like(target)

        for i in range((target.shape[1] - patch_size)//step_size + 2):
            temp = []
            for j in range((target.shape[2] - patch_size)//step_size + 2):
                min1 = min(i*step_size + patch_size, target.shape[1])
                min2 = min(j*step_size + patch_size, target.shape[2])
                temp.append(input_[0, :, min1-patch_size:min1, min2-patch_size:min2])

                mask[:target.shape[0], min1-patch_size:min1, min2-patch_size:min2] += np.ones((target.shape[0], patch_size, patch_size))

            
            temp = np.array(temp)
            with torch.no_grad():
                in_ = torch.from_numpy(temp).type(torch.float32).cuda()
                out_ = net(in_)
                out_ = out_.detach().cpu().numpy()

            k = 0
            for j in range((target.shape[2] - patch_size)//step_size + 2):
                min1 = min(i*step_size + patch_size, target.shape[1])
                min2 = min(j*step_size + patch_size, target.shape[2])

                pred[:, min1-patch_size:min1, min2-patch_size:min2] += out_[k]  
                
                k += 1

        pred /= mask
        predictions.append(pred)

        # imwrite(os.path.join(path2, str(q) + "_norm.tiff"), imgs)
        # imwrite(os.path.join(path2, str(q) + "_pred.tiff"), pred)
        # imwrite(os.path.join(path2, str(q) + "_target.tiff"), target)
        # imwrite(os.path.join(path2, str(q) + "_input.tiff"), input_)
        

        # combine input and prd to synthetic images
        if opts.save_image:
            idxes= idx2[:]
            len_idxes = len(idxes)
            montage = pred[:]
            for i, j in enumerate(idx):
                for i2, j2 in enumerate(idxes):
                    if j < j2:
                        montage = np.insert(montage, i2, input_[:,i], axis = 0)
                        idxes.insert(i2, j)
                        break
                if len(idxes) == len_idxes:
                    montage = np.append(montage, input_[:,i], axis = 0)
                    idxes.append(j)
                len_idxes = len(idxes)
            # synthetic image
            imwrite(os.path.join(path2, str(q) + "_synth.tiff"), montage)
            print("synthetic images saved!")

        q += 1

    print("start to calculate statistic")
    for i in range(len(idx2)): # for each predict channel
        targets_c = []
        predictions_c = []
        for j in range(len(images)): # for each image
            c = idx2[i]
            tgt = images[j][c]
            pred = predictions[j][i]
            targets_c.append(tgt.flatten())
            predictions_c.append(pred.flatten())
        tgt = np.hstack(targets_c)
        pred = np.hstack(predictions_c)
        corr = pearsonr(pred, tgt)[0]
        mae = mean_absolute_error(pred, tgt)
        mse = mean_squared_error(pred, tgt)
        scores.append(corr)
        scores2.append(mae)
        scores3.append(mse)
        del targets_c 
        del predictions_c 
        del tgt 
        del pred 

    scores = np.array(scores)
    scores2 = np.array(scores2)
    scores3 = np.array(scores3)

    save_path = os.path.join(path, "checkpoints/correlation_"+"_test.npy")
    np.save(save_path, scores)
    save_path2 = os.path.join(path, "checkpoints/mae_"+"_test.npy")
    np.save(save_path2, scores2)
    save_path3 = os.path.join(path, "checkpoints/mse_"+"_test.npy")
    np.save(save_path3, scores3)
    print("pcc:",np.mean(scores))
    print("mae:",np.mean(scores2))
    print("mse:", np.mean(scores3))

    print("Job complete!") 

if __name__ == '__main__':
    main()