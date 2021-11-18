import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from mldm_project.pipeline import MODEL_MAP
from mldm_project.data import DataloaderGenerator
from mldm_project.data import DATASET_PROCESSOR_MAP
from mldm_project.utils import read_yaml
import matplotlib.pyplot as plt
import pytorch_lightning as pl


        
def load_model(model_path, args):
    args['on_jupyter'] = False
    model = MODEL_MAP[args['model']].load_from_checkpoint(checkpoint_path=model_path, args=args)
    return model

def cacul_loss(y_pred, y, u_out):
        w = 1 - u_out
        mae = w * np.abs(y - y_pred)
        mae = np.sum(mae) / np.sum(w)
        return mae

def predict(data, model, args):
    data = data.copy()
    dataloder = DataloaderGenerator(args, mode='pred')
    data = DATASET_PROCESSOR_MAP[args['data_config']['processor']](data, mode='pred')
    data_dict = dataloder.generate_dataset(data)
    dataloaders = dataloder.load_dataloader(data_dict)
    trainer = pl.Trainer(gpus=args['general_config']['gpus'][:1])
    results = trainer.predict(model=model, dataloaders=dataloaders['test_loader'])
    pressures = np.concatenate(results)
    return pressures

def get_mae(config_path, data):
    trail_path = config_path.strip('/config.yaml')
    model_path = os.path.join(trail_path, next(filter(lambda x: x.startswith('epoch'), os.listdir(trail_path))))
    args = read_yaml(config_path)
    model = load_model(model_path, args['model_config'])
    pressures = data.pressure.values.reshape(-1, 80, 1)
    u_outs = data.u_out.values.reshape(-1, 80, 1)
    args = read_yaml(config_path)
    args['data_config']['batch_size'] = 1024
    pressure_preds = predict(data, model, args)
    data['pressure_pred'] = pressure_preds.reshape(-1)
    res_dict = {}
    for breah_id, pressure, pressure_pred, u_out in zip(data.breath_id.unique(), pressures, pressure_preds, u_outs):
        res_dict[breah_id] = cacul_loss(pressure_pred, pressure, u_out)
    mae_df = pd.DataFrame({'mae': res_dict})
    return mae_df, data

def get_trial_data(config_path, label='val'):
    trail_path = config_path.strip('/config.yaml')
    split_path = os.path.join(trail_path, next(filter(lambda x: x.startswith('split'), os.listdir(trail_path))))
    data = pd.read_csv('data/train.csv')
    split_df = pd.read_csv(split_path)
    selelcted_breath_id = eval(split_df[label][0])
    selelcted_data = data[data.breath_id.isin(selelcted_breath_id)].copy()
    return selelcted_data

def vis_pred(data, mae_df, breath_id_list, columns=4):
    num = len(breath_id_list)
    rows = math.ceil(num / columns)
    print(rows)
    _, axes = plt.subplots(rows, columns, figsize=(8*columns, 6*rows))
    for idx, breath_id in enumerate(breath_id_list):
        select_data = data[data.breath_id == breath_id].copy()
        for y_label in ['u_in', 'u_out', 'pressure', 'pressure_pred']:
            ax = sns.lineplot(x=select_data.time_step, y=select_data[y_label], label=y_label, ax=axes[idx//columns][idx%columns])
        title = f'breath_id: {breath_id}, mae: {np.round(mae_df.loc[breath_id].mae, 4)}'
        ax.set_title(title)
    plt.show()