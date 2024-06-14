import torch

from glob import glob
import os 
import numpy as np

# create a torch dataset for .pkl files
class BridgeAccDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 measurement='acc',
                 sns_loc=0,
                 transform=None,
                 split='train',
                 label_map=None):
        """
        Args:
            data_dir (string): Directory with all the data.
            measurement (string): measurement type. one of ['acc', 'disp']
            sns_loc (int): sensor location. e.g. 0, 1, 2, 3, 4, ... n
                           Note: this is not the element number. Check the data before using.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): split of the dataset. one of ['train', 'val', 'test']
        """

        self.data_dir = data_dir
        self.measurement = measurement
        self.sns_loc = sns_loc
        self.transform = transform

        self.data_list = sorted(glob(os.path.join(data_dir, split, '*.pkl')))

        if label_map:
            self.label_map = label_map
        else:
            self.label_map = {1: 0, 0.9: 1, 0.8: 2, 0.7: 3, 0.6: 4}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            output (dict)
        """
        data = np.load(self.data_list[idx], allow_pickle=True)
        output = {}
        output['measurement'] = data[self.measurement][self.sns_loc]

        # if the data has key 'loss_factor'
        if 'loss_factor' in data.keys():
            output['label'] = self.label_map[data['loss_factor']]
        else:
            output['label'] = 0

        if self.transform:
            output = self.transform(output)

        return output
    
    def get_metadata(self, idx):
        """
        Args:
            idx (int): Index
        
        Returns:
            metadata (dict)
        """
    
        return np.load(self.data_list[idx], allow_pickle=True)
