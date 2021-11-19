import pandas as pd
import numpy as np




def add_rollling_feature(df, window_size):
    df[f'rolling_{window_size}_mean'] = df.groupby('breath_id')['u_in'] \
                                .rolling(window=window_size, min_periods=1).mean() \
                                .reset_index(drop=True)
    df[f'rolling_{window_size}_std'] = df.groupby('breath_id')['u_in'] \
                                .rolling(window=window_size, min_periods=1).std() \
                                .reset_index(drop=True)
    df[f'rolling_{window_size}_max'] = df.groupby('breath_id')['u_in'] \
                                .rolling(window=window_size, min_periods=1).max() \
                                .reset_index(drop=True)
    df = df.fillna(0)
    return df

def add_lag_feature(df):
    for lag in range(1, 5):
        df[f'breath_id_lag{lag}']=df['breath_id'].shift(lag).fillna(0)
        df[f'breath_id_lag{lag}same']=np.select([df[f'breath_id_lag{lag}']==df['breath_id']], [1], 0)
        # u_in 
        df[f'u_in_lag{lag}'] = df['u_in'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_in_lag_{lag}_back'] = df['u_in'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_in_time{lag}'] = df['u_in'] - df[f'u_in_lag{lag}']
        # df[f'u_in_time{lag}_back'] = df['u_in'] - df[f'u_in_lag_{lag}_back']
        df[f'u_out_lag{lag}'] = df['u_out'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_out_time{lag}'] = df['u_out'] - df[f'u_out_lag{lag}']
        # df[f'u_out_lag_{lag}_back'] = df['u_out'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']
    # breath_time
    df['time_step_lag'] = df['time_step'].shift(1).fillna(0) * df[f'breath_id_lag{lag}same']
    df['breath_time'] = df['time_step'] - df['time_step_lag']
    drop_columns = ['time_step_lag']
    drop_columns += [f'breath_id_lag{i}' for i in range(1, 5)]
    drop_columns += [f'breath_id_lag{i}same' for i in range(1, 5)]
    df = df.drop(drop_columns, axis=1)
    # fill na by zero
    df = df.fillna(0)
    return df

def add_feature(df):
    df['time_delta'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['delta'] = df['time_delta'] * df['u_in']
    df['area'] = df.groupby('breath_id')['delta'].cumsum()
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] / df['count']
    df = df.drop(['count','one'], axis=1)
    return df

def add_category_features(df):
    c_dic = {10: 0, 20: 1, 50:2}
    r_dic = {5: 0, 20: 1, 50:2}
    rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
    rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])} 
    df['C_cate'] = df['C'].map(c_dic)
    df['R_cate'] = df['R'].map(r_dic)
    df['RC_sum'] = (df['R'] + df['C']).map(rc_sum_dic)
    df['RC_dot'] = (df['R'] * df['C']).map(rc_dot_dic)
    return df

