import torch.utils.data
from fnet.data.czireader import CziReader
from fnet.data.fnetdataset import FnetDataset
import pandas as pd
import numpy as np
from tifffile import imread
import pdb
import os

# import fnet.transforms_2 as transforms  # for LN and SP dataset
import fnet.transforms as transforms   # for intestine dataset

class CziDataset(FnetDataset):
    """Dataset for CZI files."""

    def __init__(self, dataframe: pd.DataFrame = None, path_csv: str = None, in_index = None, out_index = None,
                    transform_source = [transforms.normalize],
                    transform_target = None):
        
        
        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv).sample(frac=1)
            
        self.transform_source = transform_source
        self.transform_target = transform_target
        self.in_index = in_index
        self.out_index = out_index
        
        # assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

    def __getitem__(self, index):
        element = self.df.iloc[index, :]

        has_target = True # not np.isnan(element['channel_target'])
        # czi = CziReader(element['path_czi'])
        im_out = list()

        imgs = []
        dic = element['path']
        imgs = imread(dic).astype("f")
        for t in self.transform_source:
            imgs = t(imgs)
        im_out.append(imgs[self.in_index])
        im_out.append(imgs[self.out_index])
        im_out.append(imgs[:1]) # Hochest
                         
        im_out = [torch.from_numpy(im.astype(np.float32)).float() for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()
