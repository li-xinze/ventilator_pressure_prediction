# -*- coding: utf-8 -*-
# @Time        : 2021/10/02 00:34:41
# @Author      : Li Xinze <sli_4@edu.hse.ru>, Katarina Kuchuk <kkuchuk@edu.hse.ru>
# @Project     : ventilator_pressure_prediction
# @Description : Feature Engineering 


import pandas as pd

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
        df = df[df['R_C'] == '20_10']
        df = df.drop('R_C', axis=1)
        return df[:40000]
    elif mode == 'pred':
        df.drop(['R', 'C'], axis=1, inplace=True)
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
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['u_in_lag'] = df['u_in'].shift(2).fillna(0)
    return sampler_v1(df, mode)


def processor_v2(df: pd.DataFrame):
    """new ideas!
    """
    
    return df