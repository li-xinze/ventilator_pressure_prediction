
import os
import yaml
import errno
import pandas as pd
import requests
import tensorflow as tf
from typing import Dict



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


def record_split_idx(args, train_idx, val_idx, fold):
    split_dict = {
        'train': str(train_idx.tolist()),
        'val': str(val_idx.tolist())
    }
    split_df = pd.DataFrame([split_dict])
    mkdir_p(os.path.join(args['general_config']['result_path'], 'splits'))
    save_path = os.path.join(args['general_config']['result_path'], 'splits', f'split_{fold}.csv')
    split_df.to_csv(save_path, index=False)


def set_gpus(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def send_bark_info(args: Dict, title: str, info: str):
    """发送消息到bark
    """
    url_prefix= args['bark_config']['url_prefix']
    if url_prefix:
        url = f'{url_prefix}/{title}/{info}?group={title}'
        requests.get(url)