from fnet.data.fnetdataset import FnetDataset
import numpy as np
import torch
from tifffile import imread
from tqdm import tqdm
import pandas as pd
import pdb

from tifffile import imwrite
import random

from memory_profiler import profile
from copy import copy

class BufferedPatchDataset3(FnetDataset):
    """Dataset that provides chunks/patchs from another dataset."""

    def __init__(self, 
                 path_csv: str = None, in_index = None, out_index = None,
                 patch_size = [192, 192], 
                 buffer_size = 1,
                 buffer_switch_frequency = 720, 
                 npatches = 100000,
                 verbose = False,
                 shuffle_images = True,
                 transform_source = None,
                 transform_target = None,
                 # drop_patch_threshold = -2000,
                 drop_patch_threshold = -np.inf,
                 dim_squeeze = None,
                 seed = None,                         
    ):

        self.df = pd.read_csv(path_csv).sample(frac=1)
        
        self.counter = 0
        
        self.buffer_switch_frequency = buffer_switch_frequency
        
        self.npatches = npatches
        
        self.buffer = list()
        
        self.verbose = verbose
        self.shuffle_images = shuffle_images
        self.dim_squeeze = dim_squeeze
        self.drop_patch_threshold = drop_patch_threshold
        
        shuffed_data_order = np.arange(0, len(self.df))

        if self.shuffle_images:
            np.random.shuffle(shuffed_data_order)
        
        if len(self.df) < buffer_size:
            buffer_size = len(self.df) 
        pbar = tqdm(range(0, buffer_size))
                       
        self.buffer_history = list()
        
        self.in_index = in_index
        self.out_index = out_index

            
        for i in pbar:
            #convert from a torch.Size object to a list
            if self.verbose: pbar.set_description("buffering images")

            datum_index = shuffed_data_order[i]
            element = self.df.iloc[datum_index, :]

            im_out = list()
    
            dic = element['path']
            # imgs = imread(dic).astype("f")
            imgs = imread(dic).astype(np.float32)
            for t in transform_source:
                imgs = t(imgs)

            im_out.append(copy(imgs[self.in_index]))
            im_out.append(copy(imgs[self.out_index]))
            im_out.append(copy(imgs[:1])) # Hochest(first channel)
            # print(imgs[:1].shape)
            del imgs
            
            # datum_size = datum[0].size()
            # datum_size2 = datum[1].size()
            datum_size = im_out[0].shape
            datum_size2 = im_out[1].shape
            
            self.buffer_history.append(datum_index)
            self.buffer.append(im_out)

            
        self.remaining_to_be_in_buffer = shuffed_data_order[i+1:]
            
        self.patch_size1 = [datum_size[0]] + patch_size
        self.patch_size2 = [datum_size2[0]] + patch_size

            
    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        self.counter +=1
        
        if (self.buffer_switch_frequency > 0) and (self.counter % self.buffer_switch_frequency == 0):
            if self.verbose: print("Inserting new item into buffer")
                
            self.insert_new_element_into_buffer()
        
        return self.get_random_patch()
                       
    def insert_new_element_into_buffer(self):
        #sample with replacement
                       
        self.buffer.pop(0)
        
        if self.shuffle_images:
            
            if len(self.remaining_to_be_in_buffer) == 0:
                self.remaining_to_be_in_buffer = np.arange(0, len(self.dataset))
                np.random.shuffle(self.remaining_to_be_in_buffer)
            
            new_datum_index = self.remaining_to_be_in_buffer[0]
            self.remaining_to_be_in_buffer = self.remaining_to_be_in_buffer[1:]
            
        else:
            new_datum_index = self.buffer_history[-1]+1
            if new_datum_index == len(self.dataset):
                new_datum_index = 0
                             
        self.buffer_history.append(new_datum_index)
        self.buffer.append(self.dataset[new_datum_index])
        
        if self.verbose: print("Added item {0}".format(new_datum_index))
  
    def get_random_patch(self):
        
        buffer_index = np.random.randint(len(self.buffer))
                                   
        datum = self.buffer[buffer_index] # [(len(in), w, h), (len(out), w, h)]

        patch = []
        generate_new_patch = False
        while not generate_new_patch:
            starts = np.array([np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0 for d, p in zip(datum[0].shape, self.patch_size1)])  #(0, x, y)

            # check sum(patch[DAPI/Hoechest]) 
            patch_size_Hochest = [1, 192, 192]
            ends_Hochest = starts + np.array(patch_size_Hochest)
            index3 = [slice(s, e) for s,e in zip(starts,ends_Hochest)]
            patch_Hochest = datum[2][tuple(index3)]
            sum_intensity = np.sum(patch_Hochest)
            # imwrite("/home/huangqis/large_intestine_images/test_patches/"+str(sum_intensity) + ".tiff", patch_Hochest)
        
            ends = starts + np.array(self.patch_size1) # (len(in), x+192, y+192)

            ends2 = starts + np.array(self.patch_size2) #(len(out), x+192, y+192)

            #thank you Rory for this weird trick
            index = [slice(s, e) for s,e in zip(starts,ends)]
            index2 = [slice(s, e) for s,e in zip(starts,ends2)]
            
            patch = [datum[0][tuple(index)], datum[1][tuple(index2)]]  # [torch.Size([len(in), 192, 192]), torch.Size([len(out), 192, 192])] 
            if self.dim_squeeze is not None:
                patch = [np.squeeze(d, self.dim_squeeze) for d in patch]

            if sum_intensity > self.drop_patch_threshold:
                return patch
            else:
                a = random.uniform(0, 1)
                if a > 0.8:
                    return patch

        return patch

    
    def get_buffer_history(self):
        return self.buffer_history


    
def _test():
    # dims_chunk = (2,3,4)
    dims_chunk = (4,5)
    ds_test = ChunkDatasetDummy(
        None,
        dims_chunk = dims_chunk,
    )
    print('Dataset len', len(ds_test))
    for i in range(3):
        print('***** {} *****'.format(i))
        element = ds_test[i]
        print(element[0])
        print(element[1])
    
if __name__ == '__main__':
    _test()
