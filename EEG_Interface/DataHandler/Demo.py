import os
import pandas as pd
from torch.utils.data import Dataset
import torch

class DemoDataset(Dataset):
    def __init__(self,inputPath):
        self.df = pd.read_csv(inputPath)
        self.labels = torch.Tensor(self.df['EyeDetection'].values)
        self.df = self.df.drop(columns=['EyeDetection'])
        self.data = torch.Tensor(self.df.values)
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]
    
    