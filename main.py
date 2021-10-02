# -*- coding: utf-8 -*-
# @Time        : 2021/10/02 02:09:29
# @Author      : Li Xinze <sli_4@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Entry


import torch
import argparse
import pytorch_lightning as pl

from typing import Dict
from mldm_project.utils import print_run_time
from mldm_project.config import load_config
from mldm_project.config import update_save_path
from mldm_project.pipeline import PredictPipeline
from mldm_project.pipeline import TrainPipeline
from mldm_project.pipeline import TrainValidPipeline


PIPELINE_MAP = {
    'train': TrainPipeline,
    'train_valid': TrainValidPipeline,
    'pred': PredictPipeline,
}


@print_run_time
def pipeline_run(args: Dict, mode: str) -> Dict:
    """run single pipeline

    Args:
        args (Dict): parameter dict from config file
        mode (str): name of pipeline

    Returns:
        Dict: updated parameter dict
    """
    # print(args)
    print('The pipeline {} has been started!'.format(mode))
    global_seed = args['general_config']['global_seed']
    pl.seed_everything(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    pipeline = PIPELINE_MAP[mode](args, mode)
    pipeline.run()
    print('The pipeline {} has been finished!'.format(mode))
    return args

def run(args):
    for p in args['general_config']['pipeline']:
        if p in PIPELINE_MAP:
            if p == 'test':
                args['general_config']['trainer_config']['accelerator'] = None
            args = pipeline_run(args, p)
        else:
            raise ValueError('Current pipeline type({}) is not supported!'.format(p))

def jupyter_run(config='config/config.yaml'):
    args = load_config(config)
    args = update_save_path(args)
    args['model_config']['on_jupyter'] = True
    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml', help='Running config path')
    args = parser.parse_args()
    params = load_config(args.config)
    params = update_save_path(params)
    params['model_config']['on_jupyter'] = False
    run(params)
