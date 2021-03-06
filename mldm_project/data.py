# -*- coding: utf-8 -*-
# @Time        : 2021/9/25 15:33:49
# @Author      : Li Xinze <sli_4@edu.hse.ru>, Katarina Kuchuk <kkuchuk@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Dataset and dataloader 

import os
from numpy.lib.function_base import select
import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from typing import Dict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
from mldm_project.data_processor import processor_v1
from mldm_project.data_processor import processor_v2


DATASET_PROCESSOR_MAP = {
    'v1': processor_v1,
    'v2': processor_v2
}

DATASET_SPLIT_MAP = {
    'train_valid': ['train', 'val'],
    'train': ['train_val'],
    'test': ['test']
}


class VentiatorDataset(Dataset):
    """VentiatorDataset Class
    """
    def __init__(self, data: pd.DataFrame):
        self.X = None
        self.y = None
        self.u_out = None
        self.data = data
        self.load()
        
    def __len__(self):
        """Return the length of dataset
        """
        return len(self.y)
    
    def __getitem__(self, idx):
        """Return a batch
        """
        return self.X[idx], self.y[idx], self.u_out[idx]

    def load(self):
        """prepare the dataset from processed data
        """
        df = self.data
        for column in ['id', 'breath_id']:
            df.drop(column, axis=1, inplace=True)
        
        
        RS = RobustScaler()
        # self.X = RS.fit_transform( df.drop('pressure', axis=1)).reshape(-1, 80, df.shape[-1] - 1)
        self.X = df.drop('pressure', axis=1).to_numpy().reshape(-1, 80, df.shape[-1] - 1)
        self.y = df[['pressure']].to_numpy().reshape(-1, 80, 1)
        self.u_out = df[['u_out']].to_numpy().reshape(-1, 80, 1)

        
        
class DataloaderGenerator():
    """DataloaderGenerator Class
    """
    def __init__(self, args: Dict, mode: str):
        self.args = args
        self.mode = mode

    def collate_fn(self, batch):
        X, y, u_out= map(list, zip(*batch))
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()
        u_out = torch.tensor(u_out)
        return X, y, u_out

    def load_dataloader(self, data_dict: Dict) -> Dict:
        dataloaders = {}
        config = {
            'batch_size': self.args['data_config']['batch_size'],
            'num_workers': self.args['data_config']['num_workers'],
            'collate_fn': self.collate_fn
        }
        for k, v in data_dict.items():
            if 'test' in k:
                dataloaders[k + '_loader'] = DataLoader(dataset=v, shuffle=False, **config)
            else:
                dataloaders[k + '_loader'] = DataLoader(dataset=v, shuffle=True, **config)
        return dataloaders

    def record_split(self, train_data, val_data):
        train_breath_id = train_data.breath_id.unique().tolist()
        val_breath_id = val_data.breath_id.unique().tolist()
        split_dict = {
            'train': str(train_breath_id),
            'val': str(val_breath_id)
        }
        split_df = pd.DataFrame([split_dict])
        save_path = os.path.join(self.args['general_config']['trial_path'], 'split.csv')
        split_df.to_csv(save_path, index=False)

    def generate_dataset(self, data: pd.DataFrame) -> Dict:
        """gengerate dataset dict acorrding to the mode

        Args:
            data (pd.DataFrame): processed data
        Returns:
            Dict: dataset dict
        """
        data_dict = {}
        if self.mode == 'train_valid':
            groups = data.breath_id
            gss = GroupShuffleSplit(n_splits=2, train_size=.8, random_state=42)
            train_idx , val_idx = next(gss.split(data, groups=groups))
            train_data = data.iloc[train_idx].copy()
            val_data = data.iloc[val_idx].copy()
            self.record_split(train_data, val_data)
            data_dict['train'] = VentiatorDataset(train_data)
            data_dict['val'] = VentiatorDataset(val_data)
            
        elif self.mode == 'train':
            data_dict['train'] = VentiatorDataset(data)
        elif self.mode == 'pred':
            data_dict['test'] = VentiatorDataset(data)

        return data_dict

    def get_input_dim(self, data_dict: Dict) -> int:
        X, y, u_out = list(data_dict.values())[0][0]
        return X.shape[-1]

    def generate_dataloader(self) -> Dict:
        """generate dataloader

        Args:
        Returns:
            Dict: dataloader dict
        """
        if 'train' in self.mode:
            data = pd.read_csv(self.args['data_config']['train_data_path'])
        elif self.mode == 'pred':
            data = pd.read_csv(self.args['data_config']['test_data_path'])
            data['pressure'] = 0

        data = DATASET_PROCESSOR_MAP[self.args['data_config']['processor']](data, self.mode)
        data_dict = self.generate_dataset(data)
        input_dim = self.get_input_dim(data_dict)
        return self.load_dataloader(data_dict), input_dim


