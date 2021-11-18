# -*- coding: utf-8 -*-
# @Time        : 2021/9/24 16:15:12
# @Author      : Li Xinze <sli_4@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : config tools


import os
from typing import Dict
from mldm_project.utils import mkdir_p
from mldm_project.utils import read_yaml
from mldm_project.utils import save_yaml



def load_config(config_path: str) -> Dict:
    """load config file

    Args:
        config_path (str): path to config file

    Returns:
        Dict: config dict
    """
    args = read_yaml(config_path)
    args = update_result_path(args, postfix=None)
    if len(args['general_config']['gpus']) == 1:
        args['general_config']['trainer_config']['accelerator'] = None
    print(f'Using config file: {config_path}')
    return args


def save_config(config: Dict, save_dir: str):
    """save config file

    Args:
        config (Dict): config dict
        save_dir (str): save path of config file
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_yaml(os.path.join(save_dir, 'config.yaml'), config)


def update_result_path(params: Dict, postfix: str = None) -> Dict:
    """update path for storing experiment result
    Args:
        config (Dict): basic config dict
        postfix (str): the postfix of the 

    Returns:
        Dict: 更新后的配置参数
    """
    if 'train' not in params['general_config']['pipeline'][0]:
        if isinstance(params['general_config']['load_ckpt_path'], list):
            params['general_config']['result_path'] = os.path.dirname(params['general_config']['load_ckpt_path'][0])
        else:
            params['general_config']['result_path'] = os.path.dirname(params['general_config']['load_ckpt_path'])
    else:
        if postfix is None:
            postfix = params['model_config']['model']
        params['general_config']['result_path'] = params['general_config']['result_path'].format(postfix)
        result_dir_id = 0
        path_exists = True
        while path_exists:
            result_dir_id += 1
            temp_path = os.path.join(params['general_config']['result_path'], str(result_dir_id))
            path_exists = os.path.exists(temp_path)
        params['general_config']['result_path'] = temp_path
        mkdir_p(params['general_config']['result_path'])
    return params


def update_save_path(params: Dict) -> Dict:
    """update save path of result

    Args:
        params (Dict): config

    Returns:
        Dict: updated config
    """
    if 'train' in params['general_config']['pipeline'][0]:
        path_to_results = os.path.join(params['general_config']['result_path'], params['data_config']['dataset'])
        mkdir_p(path_to_results)
        params['general_config']['trial_path'] = path_to_results
    return params