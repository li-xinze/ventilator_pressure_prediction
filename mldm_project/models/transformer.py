"""
The transformerModel is a modification of  
https://www.kaggle.com/takamichitoda/ventilator-train-transformer
"""

from typing import Sequence
import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mldm_project.models.base_model import BaseModel
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(BaseModel):

    def __init__(self, args):
        super().__init__(args)

        input_dim = args['input_dim']
        print('input_dim', input_dim)
        self.seq_emb = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.pos_encoder = PositionalEncoding(d_model=64, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=512, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        self.outlayer = nn.Linear(64, 1)
        self.lr = args['lr']

    def get_tgt_mask(self, size) -> torch.tensor:
            # Generates a squeare matrix where the each row allows one word more to be seen
            mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
            mask = mask.float()
            mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
            mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        h = self.seq_emb(X)
        # h = self.pos_encoder(h)
        h = h.permute(1, 0, 2)
        h = self.transformer_encoder(h)
        h = h.permute(1, 0, 2)
        regr = self.outlayer(h)
        return regr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=30, verbose=True),
                'monitor': 'mae',
            },
        }
        return optimizer