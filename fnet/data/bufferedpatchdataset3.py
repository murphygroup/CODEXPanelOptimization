from fnet.data.fnetdataset import FnetDataset
import numpy as np
import torch

from tqdm import tqdm

import pdb

from tifffile import imwrite
import random


class BufferedPatchDataset3(FnetDataset):
    """Dataset that provides chunks/patchs from another dataset."""

    def __init__(self, 
                 dataset,
                 patch_size, 
                 buffer_size = 1,
                 buffer_switch_frequency = 720, 
                 npatches = 100000,
                 verbose = False,
                 transform = None,
                 shuffle_images = True,
                 dim_squeeze = None,
                 
    ):
        
        self.counter = 0
        
        self.dataset = dataset
        self.transform = transform
        
        self.buffer_switch_frequency = buffer_switch_frequency
        
        self.npatches = npatches
        
        self.buffer = list()
        
        self.verbose = verbose
        self.shuffle_images = shuffle_images
        self.dim_squeeze = dim_squeeze
        
        shuffed_data_order = np.arange(0, len(dataset))

        if self.shuffle_images:
            np.random.shuffle(shuffed_data_order)
        
        if len(dataset) < buffer_size:
            buffer_size = len(dataset) 
        pbar = tqdm(range(0, buffer_size))
                       
        self.buffer_history = list()
            
        for i in pbar:
            #convert from a torch.Size object to a list
            if self.verbose: pbar.set_description("buffering images")

            datum_index = shuffed_data_order[i]
            datum = dataset[datum_index]
            
            datum_size = datum[0].size()
            datum_size2 = datum[1].size()
            
            self.buffer_history.append(datum_index)
            self.buffer.append(datum)
            
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
            starts = np.array([np.random.randint(0, d - p + 1) if d - p + 1 >= 1 else 0 for d, p in zip(datum[0].size(), self.patch_size1)]) #(0, x, y)

            # check sum(patch[DAPI]) 
            patch_size_Hochest = [1, 192, 192]
            ends_Hochest = starts + np.array(patch_size_Hochest)
            index3 = [slice(s, e) for s,e in zip(starts,ends_Hochest)]
            patch_Hochest = datum[2][tuple(index3)]
            sum_intensity = np.sum(patch_Hochest.detach().numpy())
            # imwrite("/home/huangqis/large_intestine_images/test_patch/"+str(sum_intensity) + ".tiff", patch_Hochest.detach().numpy())
        
            ends = starts + np.array(self.patch_size1) # (len(in), x+192, y+192)

            ends2 = starts + np.array(self.patch_size2) #(len(out), x+192, y+192)

            #thank you Rory for this weird trick
            index = [slice(s, e) for s,e in zip(starts,ends)]
            index2 = [slice(s, e) for s,e in zip(starts,ends2)]
            
            patch = [datum[0][tuple(index)], datum[1][tuple(index2)]]  # [torch.Size([len(in), 192, 192]), torch.Size([len(out), 192, 192])] 
            if self.dim_squeeze is not None:
                patch = [torch.squeeze(d, self.dim_squeeze) for d in patch]

            
            if sum_intensity > -20000.0:
                generate_new_patch = True
            else:
                a = random.uniform(0, 1)
                # if a > 0.9:
                if a > 0.8:
                    generate_new_patch = True

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
