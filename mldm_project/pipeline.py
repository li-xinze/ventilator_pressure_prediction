# -*- coding: utf-8 -*-
# @Time        : 2021/9/25 10:24:40
# @Author      : Li Xinze <sli_4@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Training and inference process


import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from mldm_project.config import save_config
from mldm_project.models.lstm_v1 import LSTMv1
from mldm_project.models.transformer import TransformerModel
from mldm_project.data import  DataloaderGenerator
from pytorch_lightning.plugins import DDPPlugin
from mldm_project.utils import generate_checkpoint_callback
from mldm_project.utils import generate_early_stop_callback


MODEL_MAP = {
    'LSTMv1': LSTMv1,
    'Transformer': TransformerModel
}


class BasePipeline(ABC):
    """
    Pipeline's abstract base class，which defines basic methods of pipeline
    """
    def __init__(self, args: Dict, mode: str):
        self.args = args
        self.mode = mode
        self.dataloder = DataloaderGenerator(args, mode) 

    def record_config(self, input_dim):
        self.args['model_config']['input_dim'] = input_dim
        save_config(self.args, self.args['general_config']['trial_path'])

    @abstractmethod
    def run(self):
        pass

    def build_trainer(self):
        raise NotImplementedError('The build_trainer method is not realized!')

    def build_model(self):
        raise NotImplementedError('The build_model method is not realized!')


class TrainPipeline(BasePipeline):
    """TrainPipeline Class

    Attributes:
        args (Dict): config
        model (Subclass of BaseModel): model used
        mode (str): current mode in pipeline
        dataloder (Dict): dict of DataLoader objects
    """

    def build_model(self):
        return MODEL_MAP[self.args['model_config']['model']](self.args['model_config'])

    def build_trainer(self) -> pl.Trainer:
        """build_trainer

        Returns:
            pl.Trainer: a pl Triner
        """
        return pl.Trainer(default_root_dir=self.args['general_config']['trial_path'],
                            max_epochs=self.args['general_config']['num_epochs'],
                            gpus=self.args['general_config']['gpus'],
                            # logger=self.mlflow_logger,
                            plugins=DDPPlugin(find_unused_parameters=False),
                            **self.args['general_config']['trainer_config'])

    def train(self, dataloaders: Dict, model: Any): 
        """model training 

        Args:
            dataloaders (Dict): dataloader dict
        """
        self.trainer.fit(model, dataloaders['train_loader'])

    def save_model(self):
        """save model in trail_path
        """
        ckpt_path = os.path.join(
            self.args['general_config']['trial_path'],
            '{}_epoch={}.ckpt'.format(self.mode, self.args['general_config']['num_epochs'] - 1))
        self.trainer.save_checkpoint(ckpt_path)
        self.args['general_config']['load_ckpt_path'] = ckpt_path

    def run(self):
        dataloaders, input_dim = self.dataloder.generate_dataloader()
        self.record_config(input_dim)
        model = self.build_model()
        self.trainer = self.build_trainer()
        self.train(dataloaders, model)
        self.save_model()


class TrainValidPipeline(BasePipeline):
    """TrainValidPipeline Class

    Attributes:
        args (Dict): config
        mode (str): current mode in pipeline
        dataloder (Dict): dict of DataLoader objects
    """
    def __init__(self, args: Dict, mode: str):
        self.args = args
        self.mode = mode
        self.dataloder = DataloaderGenerator(args, mode) 

    def build_model(self):
        return MODEL_MAP[self.args['model_config']['model']](self.args['model_config'])

    def build_trainer(self) -> pl.Trainer:
        """build_trainer

        Returns:
            pl.Trainer: a pl Trainer
        """
        self.early_stop_callback = generate_early_stop_callback(self.args['general_config'])
        self.checkpoint_callback = generate_checkpoint_callback(self.args['general_config'])
        return pl.Trainer(default_root_dir=self.args['general_config']['trial_path'],
                          max_epochs=self.args['general_config']['num_epochs'],
                          gpus=self.args['general_config']['gpus'],
                        # logger=self.mlflow_logger,
                        # auto_lr_find=True,
                          plugins=DDPPlugin(find_unused_parameters=False),
                          callbacks=[self.early_stop_callback, self.checkpoint_callback],
                          **self.args['general_config']['trainer_config'])

    def train(self, dataloaders: Dict, model: Any):
        """model training 

        Args:
            dataloaders (Dict): dataloader dict
        """
        self.trainer.fit(model, dataloaders['train_loader'], dataloaders['val_loader'])
        ckpt_path = self.checkpoint_callback.best_model_path
        ckpt_name = os.path.basename(ckpt_path)
        self.args['general_config']['num_epochs'] = int(ckpt_name.split('-step')[0].split('epoch=')[-1]) + 1
        self.args['general_config']['load_ckpt_path'] = ckpt_path

    def run(self):
        dataloaders, input_dim = self.dataloder.generate_dataloader()
        self.record_config(input_dim)
        model = self.build_model()
        self.trainer = self.build_trainer()
        self.train(dataloaders, model)


class PredictPipeline(BasePipeline):
    """PredictPipeline Class

    Attributes:
        args (Dict): config
        mode (str): current mode in pipeline
        dataloder (Dict): dict of DataLoader objects
    """
    def __init__(self, args: Dict, mode: str):
        self.args = args
        self.mode = mode
        self.dataloder = DataloaderGenerator(args, mode) 
        
    def build_model(self):
        return MODEL_MAP[self.args['model_config']['model']].load_from_checkpoint(checkpoint_path=self.args['general_config']['load_ckpt_path'],
                                                                                   args=self.args['model_config'])

    def build_trainer(self) -> pl.Trainer:
        return pl.Trainer(gpus=self.args['general_config']['gpus'][:1])

    def predict(self, dataloaders: Dict, model: Any) -> np.ndarray:
        """inference 

        Args:
            model (Any): model
            dataloaders (Dict): dataloder dict

        Returns:
            np.ndarray: prediction result
        """
        results = self.trainer.predict(model=self.model, dataloaders=dataloaders['test_loader'])
        pressures = np.concatenate(results)
        return pressures.reshape(-1, 1)
      

    def run(self):
        """run pipeline
        """
        data = pd.read_csv(self.args['data_config']['test_data_path'])
        data = data[['id']]
        dataloaders, input_dim = self.dataloder.generate_dataloader()
        self.record_config(input_dim)
        self.trainer = self.build_trainer()
        model = self.build_model()
        data['pressure'] = self.predict(dataloaders, model)
        save_path = os.path.join(os.path.dirname(self.args['general_config']['load_ckpt_path']),
                                 os.path.basename(self.args['data_config']['test_data_path']).replace('.csv', '_PRED.csv'))
        data.to_csv(save_path, index=False)