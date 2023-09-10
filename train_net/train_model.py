import argparse
import fnet
import fnet.data
import fnet.fnet_model_final as model_use
import json
import logging
import numpy as np
import os
import pdb
import sys
import time
import torch
import warnings
from scipy.stats import pearsonr
from fnet.transforms_2 import normalize as normalize2  # lymph node, spleen
from fnet.transforms import normalize                  # large/small intestine
import pandas as pd
from tifffile import imread
from memory_profiler import profile
from tqdm import tqdm
from copy import copy

# @profile
def get_dataloader(remaining_iterations, opts, validation=False):
    transform_signal = [eval(t) for t in opts.transform_signal]
    transform_target = [eval(t) for t in opts.transform_target]

    ds_patch = fnet.data.BufferedPatchDataset3(
        path_csv = opts.path_dataset_csv if not validation else opts.path_dataset_val_csv,
        in_index = opts.input_index,
        out_index = opts.target_index,
        patch_size = opts.patch_size,
        buffer_size = opts.buffer_size if not validation else len(ds),
        buffer_switch_frequency = opts.buffer_switch_frequency if not validation else -1,
        npatches = remaining_iterations*opts.batch_size if not validation else 100*opts.batch_size,   # number of validation patches
        verbose = True,
        shuffle_images = opts.shuffle_images,
        transform_source = transform_signal,
        transform_target = transform_target,
        drop_patch_threshold = opts.drop_patch_threshold,
        **opts.bpds_kwargs,
    )
    dataloader = torch.utils.data.DataLoader(
        ds_patch,
        batch_size = opts.batch_size,
    )
    return dataloader

def freeze_layers(net):
    ''' freeze encoder and bridge '''
    for i, child in enumerate(net.children()):
        for param in child.parameters():
            param.requires_grad = False
            # param.grad = None
        if i > 4:
            break
    return net
    
def average_params(iters, paths, opts, logger):
    ''' everage all parameters of several models from paths'''
    dicts = []
    for i in range(len(paths)):
        path = paths[i]
        iter_ = int(iters[i])
        path_model_state = os.path.join(str(path), "checkpoints/model_{:06d}.p".format(iter_))
        model = fnet.functions.load_model_from_checkpoint(path_model_state, opts.in_dim, opts.out_dim, gpu_ids=opts.gpu_ids, id = 0)
        state_dict = model.net.state_dict()
        dicts.append(state_dict)
        logger.info('model loaded from: {:s}'.format(path_model_state))
    for key in dicts[0]:
        dicts[0][key] = sum(dicts[i][key] for i in range(len(dicts)))/len(dicts)
    return dicts[0]

def main():
    parser = argparse.ArgumentParser()
    # factor_yx = 0.37241  # 0.108 um/px -> 0.29 um/px
    # default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser.add_argument('--p', type=float, default=0.05, help='parameter of adjusted loss function')
    parser.add_argument('--batch_size', type=int, default=128, help='size of each batch')  # 128
    parser.add_argument('--bpds_kwargs', type=json.loads, default={}, help='kwargs to be passed to BufferedPatchDataset')
    parser.add_argument('--buffer_size', type=int, default=8, help='number of images to cache in memory')
    parser.add_argument('--buffer_switch_frequency', type=int, default=5000000000, help='BufferedPatchDataset buffer switch frequency')
    parser.add_argument('--class_dataset', default='CziDataset', help='Dataset class')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=0, help='GPU ID')
      
    parser.add_argument('--interval_save', type=int, default=100, help='iterations between saving log/model')  # 400 to 200
    parser.add_argument('--iter_checkpoint', type=int, default=100, help='iterations at which to save checkpoints of model') # 400 to 200
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # 0.0002
    parser.add_argument('--n_iter', type=int, default=20000, help='number of training iterations')
    parser.add_argument('--nn_kwargs', type=json.loads, default={}, help='kwargs to be passed to nn ctor')
    parser.add_argument('--nn_module', default='fnet_nn_3d', help='name of neural network module')

    # parser.add_argument('--loss_coef', default=10, help='constant in loss func')
    # parser.add_argument('--loss_coef', default=10, help='constant in loss func')
         
    # ER: z=64, mem, nuc_env, microtube: z=48
    parser.add_argument('--patch_size', nargs='+', type=int, default=[192, 192], help='size of patches to sample from Dataset elements')


    parser.add_argument('--path_dataset_csv', type=str, help='path to csv for constructing Dataset')
    parser.add_argument('--path_dataset_val_csv', type=str, help='path to csv for constructing validation Dataset (evaluated everytime the model is saved)')

    parser.add_argument('--input_index', type=int, nargs='+', default=[2, 5, 6, 7, 8, 9, 22, 25, 27, 28], help='input channel index')
    parser.add_argument('--target_index', type=int, nargs='+', default=[0, 1, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 26], help='output channel index')
    parser.add_argument('--out_drop_index', type=int, nargs='+', help='index of added input in the previous target list')
    parser.add_argument('--drop_patch_threshold', type=int, default=-100000000, help='threshold for dropping training patches')  

    parser.add_argument('--in_dim', type=int, default = 10, help='random seed')
    parser.add_argument('--out_dim', type=int, default = 19, help='random seed')
    
    
    parser.add_argument('--path_run_dir', default='test_models', help='base directory for saved models')
    parser.add_argument('--path_pretrain_dir', default='', help='base directory for pretrained models')
    parser.add_argument('--checkpoint_best_model', type=int, default=[], help='checkpoint of best model for transfer learning')
    parser.add_argument('--transfer_learning_0', action='store_true', help='set to transfer learning using the pretrained model on new dataset')
    parser.add_argument('--transfer_learning_1', action='store_true', help='set to transfer learning with freezing part of layers')
    parser.add_argument('--model_list', nargs='+', default=[], help='list of models should be loaded')
    
    parser.add_argument('--seed', type=int, default = 29, help='random seed')
    parser.add_argument('--shuffle_images', action='store_true', help='set to shuffle images in BufferedPatchDataset')
    parser.add_argument('--transform_signal', nargs='+', default=[], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=[], help='list of transforms on Dataset target')
    opts = parser.parse_args()
      
    opts.retrain = False

    time_start = time.time()
    if not os.path.exists(opts.path_run_dir):
        os.makedirs(opts.path_run_dir)
    if opts.iter_checkpoint is not None:
        path_checkpoint_dir = os.path.join(opts.path_run_dir, 'checkpoints')
        if not os.path.exists(path_checkpoint_dir):
            os.makedirs(path_checkpoint_dir)

    #Setup logging
    logger = logging.getLogger('model training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(opts.path_run_dir, 'run.log'), mode='a')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.addHandler(sh)

    # Set random seed
    if opts.seed is not None:
        seed = opts.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   
   
    
    #Instantiate Model
    path_model = os.path.join(opts.path_run_dir, 'model.p')
    path_pretrain = os.path.join(opts.path_pretrain_dir, 'model.p')

    if os.path.exists(path_pretrain):
        assert not os.path.exists(path_model)
        assert len(opts.out_drop_index) == 1
    
        # load pretrained model
        # iter_ = opts.checkpoint_best_model
        # path_model_state = os.path.join(opts.path_pretrain_dir, "checkpoints/model_{:06d}.p".format(iter_))
        path_model_state = os.path.join(opts.path_pretrain_dir, "model_best.p")
        state_dict = fnet.functions.load_pretrained_model(path_model_state, opts.in_dim - 1, opts.out_dim + 1, opts.out_drop_index[0], gpu_ids=opts.gpu_ids, id = 0) 
        logger.info('model loaded from: {:s}'.format(path_model_state))

        model = model_use.Model(
        nn_module=opts.nn_module,
        lr=opts.lr,
        gpu_ids=opts.gpu_ids,
        retrain = opts.retrain,
        nn_kwargs=opts.nn_kwargs,
        in_dim = opts.in_dim,
        out_dim = opts.out_dim,
        )
        model.init_weights = False
        model.retrain = True
        model.net.load_state_dict(state_dict['nn_net_state'])
        logger.info(model) 

    elif os.path.exists(path_model):
        logger.info('exist, continue training the model') 

        model = fnet.functions.load_model_from_dir_li(opts.path_run_dir, opts.in_dim, opts.out_dim, gpu_ids=opts.gpu_ids, id = 0)  
        logger.info('model loaded from: {:s}'.format(path_model))

        model.retrain = True

    else:
        logger.info('creating a new model')
        model = model_use.Model(
            nn_module=opts.nn_module,
            lr=opts.lr,
            gpu_ids=opts.gpu_ids,
            retrain = opts.retrain,
            nn_kwargs=opts.nn_kwargs,
            in_dim = opts.in_dim,
            out_dim = opts.out_dim,
            # patch_size = opts.patch_size[0]
        )
        logger.info('Model instianted from: {:s}'.format(opts.nn_module))
        
    logger.info(model) 

    # create new file to save history
    # fnetlogger = fnet.FnetLogger(columns=['num_iter', 'loss_batch'])
    path_losses_csv = os.path.join(opts.path_run_dir, 'losses.csv')
    if os.path.exists(path_losses_csv):
        fnetlogger = fnet.FnetLogger(path_losses_csv)
        logger.info('History loaded from: {:s}'.format(path_losses_csv))
    else: 
        fnetlogger = fnet.FnetLogger(columns=['num_iter', 'loss_batch'])

    n_remaining_iterations = max(0, (opts.n_iter - model.count_iter))

    
    dataloader_train = get_dataloader(n_remaining_iterations, opts)
    if opts.path_dataset_val_csv is not None:
        # dataloader_val = get_dataloader(n_remaining_iterations, opts, validation=True)
        path_losses_val_csv = os.path.join(opts.path_run_dir, 'losses_val.csv')
        if os.path.exists(path_losses_val_csv):
            fnetlogger_val = fnet.FnetLogger(path_losses_val_csv)
            logger.info('History loaded from: {:s}'.format(path_losses_val_csv))
        else:
            fnetlogger_val = fnet.FnetLogger(columns=['num_iter', 'val_loss', 'mse_loss', 'val_pcc'])

    if opts.path_dataset_csv is not None:
        path_losses_train_csv = os.path.join(opts.path_run_dir, 'corr_train.csv')
        if os.path.exists(path_losses_train_csv):
            fnetlogger_train = fnet.FnetLogger(path_losses_train_csv)
            logger.info('History loaded from: {:s}'.format(path_losses_train_csv))
        else:
            fnetlogger_train = fnet.FnetLogger(columns=['num_iter', 'corr_train', 'train_loss'])

    with open(os.path.join(opts.path_run_dir, 'train_options.json'), 'w') as fo:
        json.dump(vars(opts), fo, indent=4, sort_keys=True)

    for i, (signal, target) in enumerate(dataloader_train, model.count_iter):
            
        loss_batch = model.do_train_iter(signal, target, i)
        fnetlogger.add({'num_iter': i + 1, 'loss_batch': loss_batch[0]})
        print("{:6d},  {:.3f},  {:.3f},  {:.3f},  {:.3f},  {:.3f},  {:.3f}no".format(i + 1, loss_batch[0],  loss_batch[1], loss_batch[2], loss_batch[3], loss_batch[4], loss_batch[5]))
        dict_iter = dict(
            num_iter = i + 1,
            loss_batch = loss_batch,
        )

        if ((i + 1) % opts.interval_save == 0) or ((i + 1) == opts.n_iter):
            model.save_state(path_model)
            fnetlogger.to_csv(path_losses_csv)
            logger.info('BufferedPatchDataset buffer history: {}'.format(dataloader_train.dataset.get_buffer_history()))
            logger.info('loss log saved to: {:s}'.format(path_losses_csv))
            logger.info('model saved to: {:s}'.format(path_model))

           
            if opts.path_dataset_val_csv is not None:
                loss_val_sum = 0
                
                path_save_val_loss = os.path.join(path_checkpoint_dir, 'val_mae_{:06d}.npy'.format(i + 1))
                path_save_val_pcc = os.path.join(path_checkpoint_dir, 'val_pcc_{:06d}.npy'.format(i + 1))
                path_save_val_mse = os.path.join(path_checkpoint_dir, 'val_mse_{:06d}.npy'.format(i + 1))
                
            logger.info('elapsed time: {:.1f} s'.format(time.time() - time_start))
    
        
        if (i + 1) % opts.iter_checkpoint == 0:
            path_save_checkpoint = os.path.join(path_checkpoint_dir, 'model_{:06d}.p'.format(i + 1))
            model.save_state(path_save_checkpoint)
            logger.info('model checkpoint saved to: {:s}'.format(path_save_checkpoint))
        # if (i + 1) % (2 * opts.iter_checkpoint) == 0:
        #     pred(i + 1, r"5861_test_models", r"5861_test_paths.csv")


if __name__ == '__main__':
    
    main()
