import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import os
import time
import h5py
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import torch.utils.data as Data
from PIL import Image
import torch.utils.data 
from torch.utils.data import DataLoader
import torch.distributed as dist
import argparse
import random
from ..utils import read_dir



class DeepLesionTrain(torch.utils.data.Dataset):
    def __init__(self):
        super(DeepLesionTrain, self).__init__()

        #train
        txt_file = r'/path/to/train_640geo_dir.txt'
        self.a_dir = r'/path/to/train_640geo'
        self.k = 90
        with open(txt_file, 'r') as f:
            lines = [line[:-7] for line in f]

        self.CT = lines
        self.G_max = 0.5
        self.G_min = 0.0

    

    def normalize(self, data: np.ndarray) -> np.ndarray:
        G_min = self.G_min
        G_max = self.G_max

        if G_max <= G_min:
            raise ValueError("G_max <= G_min.")

        data_clipped = np.clip(data, G_min, G_max)

        data_norm_0_1 = (data_clipped - G_min) / (G_max - G_min)

        data_final = data_norm_0_1 * 2.0 - 1.0

        return data_final


    def denormalize(self, data_neg_one_to_one: np.ndarray) -> np.ndarray:
        G_min = self.G_min
        G_max = self.G_max
        
        if G_max <= G_min:
            raise ValueError("G_max <= G_min.")
            
        data_clipped = np.clip(data_neg_one_to_one, -1.0, 1.0)
        
        data_denorm_0_1 = (data_clipped + 1.0) / 2.0

        data_physical = data_denorm_0_1 * (G_max - G_min) + G_min

        return data_physical
     
    def __getitem__(self, index):
        index_1 = index // self.k
        index_2 = index % self.k    
        
        f = os.path.join(self.a_dir, self.CT[index_1], str(index_2))
        data_name = f
        f = f + '.h5'
        data = h5py.File(f,'r')
        data_A = data['ma_CT'][:]
        data_A = self.normalize(data_A)
        A = torch.FloatTensor(data_A)

        data_C = data['LI_CT'][:]
        data_C = self.normalize(data_C)
        C = torch.FloatTensor(data_C)

        num = random.randint(1, len(self.CT) - 1)
        gt = os.path.join(self.a_dir, self.CT[num], 'gt.h5')
        data = h5py.File(gt,'r')
        data = data['image'][:]
        data = self.normalize(data)
        D = torch.FloatTensor(data)
           
        return {"data_name": data_name, "A": A, 'C': C, 'D':D}

    def __len__(self):
        return len(self.CT) * self.k
    
class DeepLesionTest(torch.utils.data.Dataset):
    def __init__(self):
        super(DeepLesionTest, self).__init__()

        #test
        txt_file = r'/path/to/test_640geo_dir.txt'
        self.a_dir = r'path/to/test_640geo'
        self.k = 10

        with open(txt_file, 'r') as f:
            lines = [line[:-7] for line in f]

        self.CT = lines
        self.G_max = 0.5
        self.G_min = 0.0

    

    def normalize(self, data: np.ndarray) -> np.ndarray:
        G_min = self.G_min
        G_max = self.G_max

        if G_max <= G_min:
            raise ValueError("G_max <= G_min.")

        data_clipped = np.clip(data, G_min, G_max)

        data_norm_0_1 = (data_clipped - G_min) / (G_max - G_min)

        data_final = data_norm_0_1 * 2.0 - 1.0

        return data_final


    def denormalize(self, data_neg_one_to_one: np.ndarray) -> np.ndarray:
        G_min = self.G_min
        G_max = self.G_max
        
        if G_max <= G_min:
            raise ValueError("G_max <= G_min.")
            
        data_clipped = np.clip(data_neg_one_to_one, -1.0, 1.0)
        
        data_denorm_0_1 = (data_clipped + 1.0) / 2.0

        data_physical = data_denorm_0_1 * (G_max - G_min) + G_min

        return data_physical
     
    def __getitem__(self, index):
        index_1 = index // self.k
        index_2 = index % self.k    
        
        f = os.path.join(self.a_dir, self.CT[index_1], str(index_2))
        data_name = f
        f = f + '.h5'
        data = h5py.File(f,'r')
        data_A = data['ma_CT'][:]
        data_A = self.normalize(data_A)
        A = torch.FloatTensor(data_A)

        data_C = data['LI_CT'][:]
        data_C = self.normalize(data_C)
        C = torch.FloatTensor(data_C)

        gt = os.path.join(self.a_dir, self.CT[index_1], 'gt.h5')
        data = h5py.File(gt,'r')
        data = data['image'][:]
        data = self.normalize(data)
        B = torch.FloatTensor(data)

        num = random.randint(1, len(self.CT) - 1)
        gt = os.path.join(self.a_dir, self.CT[num], 'gt.h5')
        data = h5py.File(gt,'r')
        data = data['image'][:]
        data = self.normalize(data)
        D = torch.FloatTensor(data)
           
        return {"data_name": data_name, "A": A, 'C': C, 'D':D}

    def __len__(self):
        return len(self.CT) * self.k