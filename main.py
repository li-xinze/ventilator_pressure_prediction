import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']='0'
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
from project.pipeline import train_cv, train_valid, train_all
from project.config import load_config
from project.error import analysis_trial
from project.utils import send_bark_info


PIPELINE_MAP = {
    'train_cv': train_cv,
    'train_valid': train_valid,
    'train_all': train_all,
}


def seed_everything(args):
    SEED = args['general_config']['global_seed']
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

def run(args):
    for mode in args['general_config']['pipeline']:
        args = PIPELINE_MAP[mode](args)

def generate_report(args):
    df = analysis_trial(args)
    df.to_csv(os.path.join(args['general_config']['result_path'], 'report.csv'))
    postfix = args['general_config']['result_path'].strip('result/').replace('/', '_')
    info = f'MAE:[{np.mean(df.Mean)}]{postfix}'
    send_bark_info(args=args, title='Vent', info=info)

def jupyter_run(config='config/config.yaml'):
    args = load_config(config)
    seed_everything(args)
    run(args)
    generate_report(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml', help='Running config path')
    args = load_config(parser.parse_args().config)
    seed_everything(args)
    run(args)
    generate_report(args)