# -*- coding: utf-8 -*-
# @Time        : 2021/10/02 03:36:32
# @Author      : Li Xinze <sli_4@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Stacked LSTM 



import torch
import torch.nn as nn
from typing import Dict
from mldm_project.models.base_model import BaseModel


class LSTMv1(BaseModel):
    def __init__(self, args: Dict):
        super(LSTMv1, self).__init__(args)

        self.lr = self.lr = args['lr']
        input_dim = args['input_dim']
        
        self.lstm_1 = nn.LSTM(input_dim, 300, bidirectional=True)
        self.lstm_2 = nn.LSTM(600, 250, bidirectional=True)
        self.lstm_3 = nn.LSTM(500, 150, bidirectional=True)
        self.lstm_4 = nn.LSTM(300, 100, bidirectional=True)
        self.out_layer = nn.Sequential(nn.Linear(200, 50),
                                       nn.ReLU(),
                                       nn.Linear(50, 1))

    def forward(self, inputs: torch.Tensor):
        # inputs size is [batch_size, sequence_len(80), embedding_dim]
        x = inputs
        x = x.permute(1, 0, 2)
        # permute since batch_first=False in lstm layers 
        for layer in [self.lstm_1, self.lstm_2, self.lstm_3, self.lstm_4]:
            x, _ = layer(x)
        x = x.permute(1, 0, 2)
        y = self.out_layer(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer