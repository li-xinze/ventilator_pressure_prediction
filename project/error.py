
import gc
gc.enable()
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from project.utils import set_gpus
from project.utils import read_yaml
from project.data import get_vent_data
from sklearn.metrics.pairwise import cosine_similarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


R = [5, 20, 50]
C = [10, 20, 50]

def cacul_loss(y_pred, y, u_out):
        w = 1 - u_out
        mae = w * np.abs(y - y_pred)
        mae = np.sum(mae) / np.sum(w)
        return mae


def get_mae(model_path, data, X):
    model = tf.keras.models.load_model(model_path)
    pressure_preds = model.predict(X, batch_size = 1024)
    data['pressure_pred'] = pressure_preds.reshape(-1)
    pressures = data.pressure.values.reshape(-1, 80, 1)
    u_outs = data.u_out.values.reshape(-1, 80, 1)
    res_dict = {}
    for breah_id, pressure, pressure_pred, u_out in zip(data.breath_id.unique(), pressures, pressure_preds, u_outs):
        res_dict[breah_id] = cacul_loss(pressure_pred, pressure, u_out)
    mae_df = pd.DataFrame({'mae': res_dict})
    del model
    gc.collect()
    return mae_df, data


def extract_data(args, idx, df):
    breath_id = df.breath_id.unique()[idx]
    df_tmp = df[df.breath_id.isin(breath_id)].copy()
    df_tmp = df_tmp.reset_index(drop=True)
    mode = args['data_config']['R_C']
    if mode != 'all':
        val_temp = (df_tmp.groupby('breath_id')['R_C'].first() == args['data_config']['R_C']).to_list()
        valid_idx = [i for i in range(len(val_temp)) if val_temp[i] == True]
        idx = list(map(idx.__getitem__, valid_idx))
        df_final = df_tmp[df_tmp.R_C == args['data_config']['R_C']]
    else:
        df_final = df_tmp
    return df_final, sorted(idx)


def split_train_val(args, df, X, split_path):
    split_df = pd.read_csv(split_path)
    train_idx = eval(split_df['train'][0])
    val_idx = eval(split_df['val'][0])
    train_df, train_idx = extract_data(args, train_idx, df)
    val_df, val_idx = extract_data(args, val_idx, df)
    result_dict = {}
    X_train, X_valid = X[train_idx], X[val_idx]
    result_dict['X'] = (X_train, X_valid)
    result_dict['data'] = (train_df, val_df)
    return result_dict


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


def error_analysis(args, fold_num=0, mode='val'):
    fold_num = str(fold_num)
    result_path = args['general_config']['result_path']
    model_file = next(filter(lambda x: fold_num in x, os.listdir(os.path.join(result_path, 'models'))))
    model_path = os.path.join(result_path, 'models', model_file)
    split_file = next(filter(lambda x: fold_num in x, os.listdir(os.path.join(result_path, 'splits'))))
    split_path = os.path.join(result_path, 'splits', split_file)
    set_gpus(args['general_config']['gpus'])
    df = pd.read_csv(args['data_config']['train_data_path'])
    R_C = args['data_config']['R_C']
    df['R_C'] = [f'{r}_{c}' for r, c in zip(df['R'], df['C'])]
    if R_C == 'all':
        pass
    else:
        df = df[df.R_C==R_C]
    X, y = get_vent_data(args)
    
    res = split_train_val(args, df, X, split_path)
    if mode=='all' or mode=='val':
        val_mae_df, val_data = get_mae(model_path, data=res['data'][1], X=res['X'][1])
    if mode=='all' or mode=='train':
        train_mae_df, train_data = get_mae(model_path, data=res['data'][0], X=res['X'][0])
    if mode == 'all':
        return pd.concat([train_mae_df, val_mae_df]), pd.concat([val_data, train_data])
    elif mode == 'val':
        return val_mae_df, val_data
    elif mode == 'train':
        return train_mae_df, train_data


def analysis_trial(args, mode='val'):
    set_gpus(args['general_config']['gpus'])
    result_path = args['general_config']['result_path']
    config_path = os.path.join(result_path, 'config.yaml')
    args = read_yaml(config_path)
    split_files = list(filter(lambda x: x.startswith('split'), os.listdir(os.path.join(result_path, 'splits'))))
    df = pd.read_csv(args['data_config']['train_data_path'])
    X, y = get_vent_data(args)
    df['R_C'] = [f'{r}_{c}' for r, c in zip(df['R'], df['C'])]
    R_C = args['data_config']['R_C']
    mae_dict = defaultdict(lambda: {}, {})
    if R_C == 'all':
        rc_list = [f'{r}_{c}' for r in R for c in C]
    else:
        df = df[df.R_C==R_C]
        rc_list = [R_C]
    for rc in rc_list:
        print(rc)
        for split_file in tqdm(split_files):
            split_path = os.path.join(result_path, 'splits', split_file)
            fold_num = next(filter(str.isdigit, split_file))
            args['data_config']['R_C'] = rc
            res = split_train_val(args, df, X, split_path)
            model_file = next(filter(lambda x: fold_num in x, os.listdir(os.path.join(result_path, 'models'))))
            model_path = os.path.join(result_path, 'models', model_file)
            if mode == 'all' or mode == 'train':
                train_mae_df, _ = get_mae(model_path, data=res['data'][0], X=res['X'][0])
                mae_dict[f'{rc}_train'][f'fold{fold_num}'] = np.round(train_mae_df.mae.mean(), 4)
            if mode == 'all' or mode == 'val': 
                val_mae_df, _ = get_mae(model_path, data=res['data'][1], X=res['X'][1])
                mae_dict[f'{rc}_val'][f'fold{fold_num}'] = np.round(val_mae_df.mae.mean(), 4)
            gc.collect()
    mae_df = pd.DataFrame(mae_dict).T
    mae_df['Mean'] = mae_df.mean(axis=1)
    gc.collect()
    return mae_df


def search_similar_id(target_id, u_in_table, th):
    if target_id in u_in_table['breath_id'].tolist():
        target_rc = u_in_table[u_in_table['breath_id']==target_id]['RC'].values[0]
        target_vec = u_in_table[u_in_table['breath_id']==target_id].to_numpy()[:,2:]
        refer_table = u_in_table[(u_in_table['RC']==target_rc) & (u_in_table['breath_id']!=target_id)].reset_index(drop=True)
        refer_vec = refer_table.to_numpy()[:,2:]
        breaths = refer_table['breath_id'].unique().tolist()
        breath_map = {i:b for i,b in enumerate(breaths)}
        cs = cosine_similarity(target_vec, refer_vec)[0]
        similar_idx = list(np.where(cs > th)[0])
        similar_id = [breath_map[i] for i in similar_idx]
    else:
        similar_id = []
    return similar_id

def viz_similar_id(target_id, data, u_in_table, threshold=0.999):
    rc = u_in_table[u_in_table['breath_id']==target_id]['RC'].values[0]
    similar_ids = search_similar_id(target_id, u_in_table, th=threshold)
    viz_id = [target_id] + similar_ids
    
    fig, axes = plt.subplots(figsize=(20, 12), nrows=2,sharex=True)
    for id_ in viz_id:
        tmp = data[data['breath_id']==id_].copy()
        axes[0].plot(tmp['time_step'], tmp['u_in'], label='breath_id : ' + str(id_))
        axes[1].plot(tmp['time_step'], tmp['pressure'], label='breath_id : ' + str(id_))
#       axes[1].plot(tmp['time_step'], tmp_pred['pressure_pred'], '--', label='breath_id : ' + str(id_) + 'pred')
        axes[0].legend(loc='upper right')
        axes[0].grid(color='g', linestyle=':', linewidth=0.3)
        axes[0].set_title('u_in')
        axes[1].legend(loc='upper right')
        axes[1].grid(color='g', linestyle=':', linewidth=0.3)
        axes[1].set_title('pressure')
        fig.suptitle(f'target_id : {target_id}  (RC={rc})')