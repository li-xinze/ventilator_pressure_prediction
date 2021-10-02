# -*- coding: utf-8 -*-
# @Time        : 2021/9/26 11:39:31
# @Author      : Li Xinze <sli_4@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Modified pl MAE 


import torch
from pytorch_lightning.metrics.metric import Metric



class VentilatorMAE(Metric):
    """
    Class of caculating mae for pl (expiratory phase is not scored)
    """

    def __init__(
        self,
        compute_on_step = True,
        dist_sync_on_step = False,
        process_group = None,
        dist_sync_fn = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _mean_absolute_error_update(self, preds: torch.Tensor, target: torch.Tensor, u_out: torch.Tensor):
        sum_abs_error = torch.sum((1 - u_out) * torch.abs(preds - target))
        n_obs = torch.sum(1 - u_out)
        return sum_abs_error, n_obs

    def _mean_absolute_error_compute(self, sum_abs_error: torch.Tensor, n_obs: int) -> torch.Tensor:
        return sum_abs_error / n_obs

    def update(self, preds: torch.Tensor, target: torch.Tensor, u_out: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        sum_abs_error, n_obs = self._mean_absolute_error_update(preds, target, u_out)
        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self):
        """
        Computes mean absolute error over state.
        """
        return self._mean_absolute_error_compute(self.sum_abs_error, self.total)

