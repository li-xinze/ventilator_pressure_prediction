# -*- coding: utf-8 -*-
# @Time        : 2021/10/02 00:34:41
# @Author      : Li Xinze <sli_4@edu.hse.ru>, Katarina Kuchuk <kkuchuk@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Feature Engineering 


import pandas as pd
from sklearn.preprocessing import RobustScaler
'''
### Principle of writing a data_processor:
- Make sure that [breath_id], [pressure] columns are in return pd.DataFrame
- No categorical featrues should be contained in return pd.DataFrame
'''



def sampler_v1(df: pd.DataFrame, mode):
    """Sample strategy
    """
    if 'train' in mode:
        df['R_C'] = [f'{r}_{c}' for r, c in zip(df['R'], df['C'])]
        df.drop(['R', 'C'], axis=1, inplace=True)
        df = df[df['R_C'] == '5_50']
        df = df.drop('R_C', axis=1)
    elif mode == 'pred':
        df.drop(columns=['R', 'C'], inplace=True)
    return df

def sampler_v0(df: pd.DataFrame, mode):
    """Sample strategy
    """
    pd.get_dummies(df, columns=['R', 'C'])
    return df

def processor_v1(df: pd.DataFrame, mode: str):
    """Add features

    Args:
        df (pd.DataFrame): input raw dataframe

    Returns:
        pd.DataFrame: processed dataframe
    """
    # pressure_map = {}
    # for idx, pressure in enumerate(sorted(df.pressure.unique())):
    #     pressure_map[pressure] = idx
    # df.pressure = df.pressure.apply(lambda x: pressure_map[x])
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag'] = df['u_in'].shift(2).fillna(0)
    return sampler_v1(df, mode)


def processor_v2(df: pd.DataFrame, mode):
    df = sampler_v1(df, mode)
    # df['time_step_s1'] = df.groupby('breath_id')['time_step'].shift(1)
    # df = df.fillna(0)
    # df['det_t'] = df['time_step'] - df['time_step_s1']
    # df['area'] = df['det_t'] * df['u_in']
    # df.drop(['time_step_s1', 'det_t'], axis=1, inplace=True)


    # df['first_pre'] = df.groupby('breath_id')['pressure'].transform('first')
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)
    
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    
    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    # df['R'] = df['R'].astype(str)
    # df['C'] = df['C'].astype(str)
    # df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    # df = pd.get_dummies(df)
    
    return df