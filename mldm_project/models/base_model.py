
import torch
import pytorch_lightning as pl

from typing import Dict
from mldm_project.metric import VentilatorMAE
from mldm_project.utils import VentilatorLoss
from livelossplot import PlotLosses


class BaseModel(pl.LightningModule):
    """
    Models' abstract base class，which defines basic methods of Models
    """

    def __init__(self, args: Dict):
        super(BaseModel ,self).__init__()
        self.args = args
        self.valid_metric = VentilatorMAE()
        self.loss = VentilatorLoss()
        if args['on_jupyter']:
            self.losslog = {}
            self.liveloss = PlotLosses()

    def training_step(self, train_batch, batch_idx):
        """Override LightningModule's method, caculation loss on each training step
        """
        X, y, u_out= train_batch
        y_hat = self(X)
        loss = self.loss(y_hat, y, u_out)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """Override LightningModule's method, caculation loss on each validation step
        """
        X, y, u_out = val_batch
        y_hat = self(X)
        loss = self.loss(y_hat, y, u_out)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.valid_metric(y_hat, y, u_out)
        self.log('mae', self.valid_metric, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def training_epoch_end(self, outputs):
        """Override LightningModule's method, log train_loss to PlotLosses in case of using jupyter
        """
        if self.args['on_jupyter']:
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            self.losslog = {}
            self.losslog['loss'] = avg_loss

    def validation_epoch_end(self, outputs):
        """Override LightningModule's method, log val_loss to PlotLosses in case of using jupyter
        """
        if self.args['on_jupyter']:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            self.losslog['val_loss'] = avg_loss
            self.liveloss.update(self.losslog)
            self.liveloss.send()

    def predict(self, test_batch, batch_idx, dataloader_idx):
        """Override LightningModule's method, do inference
        """
        X, y, u_out = test_batch
        y_hat = self(X)
        return y_hat

    def get_progress_bar_dict(self):
        """Override LightningModule's method, don't show the version number
        """
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items