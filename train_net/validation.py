import os
import fnet
import argparse
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
# from fnet.transforms_2 import normalize as normalize2
from fnet.transforms import normalize, normalize2, normalize3
from copy import copy
from ResNet_2 import ResUnet 
import fnet.fnet_model_final as model_use
# from memory_profiler import profile
import time
import logging
import sys

# @profile
def load_validation_images(val_state, val_paths): 
    transform_signal = [eval(t) for t in opts.transform_signal]
    val_input = []
    val_output = []
    num_val = len(val_paths["path"])
    if val_state == 0:
        s = 0
        t = int(num_val/3)
    elif val_state == 1:
        s = int(num_val/3)
        t = int(num_val*2/3)
    elif val_state == -1: # load all the validation images 
        s = 0
        t = int(num_val)
    else:
        s = int(num_val*2/3)
        t = num_val
    for i in tqdm(range(s, t)): 
        img = imread(val_paths["path"][i])
        img = img.astype(np.float32)
        for t in transform_signal:
            img = t(img)
        val_input.append(img[opts.input_index])
        val_output.append(img[opts.target_index])
        del img
    return val_input, val_output

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='size of each batch')  # 128
parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
parser.add_argument('--nn_kwargs', type=json.loads, default={}, help='kwargs to be passed to nn ctor')
parser.add_argument('--nn_module', default='fnet_nn_3d', help='name of neural network module')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # 0.0002
    
parser.add_argument('--iter_checkpoint', nargs='+', type=int, default=100, help='iterations at which to save checkpoints of model') # 400 to 200
parser.add_argument('--n_iter', type=int, default=20000, help='number of training iterations')
parser.add_argument('--n_start', type=int, nargs='+', default=[0], help='number of training iterations that start to validate')

parser.add_argument('--patch_size', nargs='+', type=int, default=[192, 192], help='size of patches to sample from Dataset elements')

parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
parser.add_argument('--path_dataset_val_csv', type=str, help='path to csv for constructing validation Dataset (evaluated everytime the model is saved)')

parser.add_argument('--input_index', type=int, nargs='+', default=[2, 5, 6, 7, 8, 9, 22, 25, 27, 28], help='input channel index')
parser.add_argument('--target_index', type=int, nargs='+', default=[0, 1, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26], help='output channel index')

parser.add_argument('--in_dim', type=int, default = 10, help='random seed')
parser.add_argument('--out_dim', type=int, default = 19, help='random seed')

parser.add_argument('--path_run_dir', type=str, default='test_models', help='base directory for saved models')
parser.add_argument('--path_pretrain_dir', default='test_models', help='base directory for pretrained models')
parser.add_argument('--model_list', nargs='+', default=[], help='list of models should be loaded')
parser.add_argument('--path_output_dir', type=str, default='val_loss_test.npy', help='path of saving output array to update graph')

parser.add_argument('--seed', type=int, default = 29, help='random seed')
parser.add_argument('--transform_signal', nargs='+', default=[], help='list of transforms on Dataset signal')
parser.add_argument('--transform_target', nargs='+', default=[], help='list of transforms on Dataset target')
opts = parser.parse_args()

with open(os.path.join(opts.path_run_dir, 'validation_options.json'), 'w') as fo:
    json.dump(vars(opts), fo, indent=4, sort_keys=True)

 # Set random seed
    if opts.seed is not None:
        seed = opts.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   

# @profile
def main():
    # Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    time_start = time.time()
    # load validation images
    val_state = -1
    val_paths = pd.read_csv(opts.path_dataset_val_csv)
    val_input, val_output = load_validation_images(val_state, val_paths)
    # print("loading validation images complete!")
    logger.info("loading validation images complete!")

    # create a csv file if it is not exist
    assert os.path.exists(opts.path_run_dir)
    path_checkpoint_dir = os.path.join(opts.path_run_dir, 'checkpoints')
    assert os.path.exists(path_checkpoint_dir)
    if opts.path_dataset_val_csv is not None:
        path_losses_val_csv = os.path.join(opts.path_run_dir, 'losses_val.csv')
        if os.path.exists(path_losses_val_csv):
            fnetlogger_val = fnet.FnetLogger(path_losses_val_csv)
        else:
            fnetlogger_val = fnet.FnetLogger(columns=['num_iter', 'val_loss', 'mse_loss', 'val_pcc'])
    # print("creating a csv file complete!")
    logger.info("creating a csv file complete!")

    # calulate and save statistics over all the iterations
    path_model = os.path.join(opts.path_run_dir, 'model.p')
    assert os.path.exists(path_model)
    # print("starting validation")
    logger.info("starting validation")
    for i in range(opts.n_start[0]-1, opts.n_iter):
        if ((i + 1) % opts.iter_checkpoint == 0) or ((i + 1) == opts.n_iter):
            print("iteration "+str(i+1))
            # load model
            iter_ = i+1
            path_state_dict = os.path.join(opts.path_run_dir, "checkpoints/model_{:06d}.p".format(iter_))
            if not os.path.exists(path_state_dict):
                continue
            state_dict = torch.load(path_state_dict) 
            model = model_use.Model(
            nn_module=opts.nn_module,
            lr=opts.lr,
            gpu_ids=opts.gpu_ids,
            retrain = False,
            nn_kwargs=opts.nn_kwargs,
            in_dim = opts.in_dim,
            out_dim = opts.out_dim,
            )
            model.init_weights = False
            model.net.load_state_dict(state_dict['nn_net_state']) 
            # print("successfully load model @ iteration "+str(iter_))
            logger.info("successfully load model @ iteration "+str(iter_))

            model.net.eval()
            model.net.cuda()
            if opts.path_dataset_val_csv is not None:
                val_loss, val_mse, val_pcc = model.predict(val_input, val_output, opts.patch_size) 
                val_loss = np.array(val_loss)
                val_mse = np.array(val_mse)
                val_pcc = np.array(val_pcc)
                path_save_val_loss = os.path.join(path_checkpoint_dir, 'val_mae_{:06d}.npy'.format(i + 1))
                path_save_val_pcc = os.path.join(path_checkpoint_dir, 'val_pcc_{:06d}.npy'.format(i + 1))
                path_save_val_mse = os.path.join(path_checkpoint_dir, 'val_mse_{:06d}.npy'.format(i + 1))
                np.save(path_save_val_loss, val_loss)
                np.save(path_save_val_pcc, val_pcc)
                np.save(path_save_val_mse, val_mse)
                fnetlogger_val.add({'num_iter': i + 1, 'val_loss': np.mean(val_loss), 'mse_loss': np.mean(val_mse), 'val_pcc': np.mean(val_pcc)}) 
                fnetlogger_val.to_csv(path_losses_val_csv)
                # print('loss val log saved to: {:s}'.format(path_losses_val_csv))
                # print('elapsed time: {:.1f} s'.format(time.time() - time_start))
                logger.info('loss val log saved to: {:s}'.format(path_losses_val_csv))
                logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))

    path = opts.path_run_dir
    # load "losses_val.csv"
    losses_val = pd.read_csv(path_losses_val_csv)

    # pick top5 model by mse
    sort_idx = np.argsort(losses_val['mse_loss'])
    best_idx = sort_idx[0]
    top5_idx = sort_idx[:5]
    iter_best = losses_val['num_iter'][best_idx]
    iter_keep = losses_val['num_iter'][top5_idx]
    # pick top5 model by pcc
    sort_idx2 = np.argsort(losses_val['val_pcc'])[::-1]
    top5_idx2 = sort_idx2[:5]
    iter_keep2 = losses_val['num_iter'][top5_idx2]

    # delete rest of them
    idx_keep = np.union1d(top5_idx, top5_idx2)
    idx_delete = np.setdiff1d(losses_val.index.values, idx_keep)
    # rest_idx = sort_idx[3:]
    iter_delete = losses_val['num_iter'][idx_delete]
    for iter_ in iter_delete:
        model_path = os.path.join(path, "checkpoints/model_{:06d}.p".format(iter_))
        mae_path = os.path.join(path, "checkpoints/val_mae_{:06d}.npy".format(iter_))
        mse_path = os.path.join(path, "checkpoints/val_mse_{:06d}.npy".format(iter_))
        pcc_path = os.path.join(path, "checkpoints/val_pcc_{:06d}.npy".format(iter_))
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(mae_path):
            os.remove(mae_path)
        if os.path.exists(mse_path):
            os.remove(mse_path)
        if os.path.exists(pcc_path):
            os.remove(pcc_path)

    # for the best model with least mse, save the validation images (0-255,JEPG) into "./imgs_validation"
    imgs_val_path = os.path.join(path, 'imgs_validation')
    isExist = os.path.exists(imgs_val_path)
    if not isExist:
        os.makedirs(imgs_val_path)

    # load best model
    model_path = os.path.join(path, "checkpoints/model_{:06d}.p".format(iter_best))
    state_dict = torch.load(model_path) 
    model = model_use.Model(
    nn_module=opts.nn_module,
    lr=opts.lr,
    gpu_ids=opts.gpu_ids,
    retrain = False,
    nn_kwargs=opts.nn_kwargs,
    in_dim = opts.in_dim,
    out_dim = opts.out_dim,
    )
    model.init_weights = False
    model.net.load_state_dict(state_dict['nn_net_state']) 
    
    model.net.eval()
    model.net.cuda()

    path_best_model = os.path.join(path, "model_best.p")
    model.save_state(path_best_model)                           # save the best model

    if opts.path_dataset_val_csv is not None:
        val_mae_best, _, _ = model.predict(val_input, val_output, opts.patch_size, save_img = True, path_save=imgs_val_path) # save validation images for checking 
    
    val_mae_best = np.array(val_mae_best)
    # logger.info("the iteration of best model is", str(iter_best))
    path_save_best_val_loss = opts.path_output_dir                 
    np.save(path_save_best_val_loss, val_mae_best)              # save the loss for updating graph

    logger.info('complete!')
    logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))

if __name__ == "__main__":
    main()