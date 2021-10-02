# -*- coding: utf-8 -*-
# @Time        : 2021/10/02 00:24:33
# @Author      : Li Xinze <sli_4@edu.hse.ru>, , Katarina Kuchuk <kkuchuk@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : collection of tools


import os
import time
import yaml
import torch
import errno
import functools
import numpy as np
import torch.nn as nn

from typing import Dict
from typing import Callable
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class VentilatorLoss(nn.Module):
    """Class for caculating loss (expiratory phase is not scored)
    """
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor, u_out: torch.Tensor):
        w = 1 - u_out
        mae = w * torch.abs(y - y_hat)
        mae = torch.sum(mae) / torch.sum(w)
        return mae


def read_yaml(path: str) -> Dict:
    """read yaml files

    Args:
        path (str): path to yaml file 

    Returns:
        Dict: data
    """
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.Loader)


def save_yaml(path: str, data: Dict):
    """save yaml (config file)

    Args:
        path (str): path to yaml file 
        data (Dict): data
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def mkdir_p(path):
    """Create a folder for the given path.
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise


def print_run_time(func: Callable) -> Callable:
    """print run time of a function

    Args:
        func (Callable): the function which need to record runtime
 
    Returns:
        Callable: a wrapper
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print('Function: {}, Run Time: {} min'.format(func.__name__, np.round(time_elapsed / 60, 2)))
        return result

    return wrapper


def generate_early_stop_callback(args: Dict) -> EarlyStopping:
    """generate early_stop_callback object for pl

    Args:
        args (Dict): config

    Returns:
        object: early_stop_callback object
    """
    early_stop_callback = EarlyStopping(monitor=args['monitor'], mode=args['mode'], **args['early_stop_callback'])
    return early_stop_callback


def generate_checkpoint_callback(args: Dict) -> ModelCheckpoint:
    """generate checkpoint_callback object for pl

    Args:
        args (Dict): config

    Returns:
        object: checkpoint_callback object
    """
    checkpoint_callback = ModelCheckpoint(dirpath=args['trial_path'],
                                          monitor=args['monitor'],
                                          mode=args['mode'],
                                          **args['checkpoint_callback'])
    return checkpoint_callback