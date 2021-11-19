import pandas as pd
from project.data_processor import DATASET_PROCESSOR_MAP
from sklearn.preprocessing import RobustScaler

R = [5, 20, 50]
C = [10, 20, 50]

def get_vent_data(args):
    print('Loading data...........')
    rc = args['data_config']['R_C']
    train = pd.read_csv(args['data_config']['train_data_path'])
    if rc =='all':
        pass
    else:
        train['R_C'] = [f'{r}_{c}' for r, c in zip(train['R'], train['C'])]
        train.drop(['R', 'C'], axis=1, inplace=True)
        assert rc in [f'{r}_{c}' for r in R for c in C]
        train = train[train['R_C'] == rc]
        train = train.drop('R_C', axis=1)

    train = DATASET_PROCESSOR_MAP[args['data_config']['processor']](train, rc)
    y = train[['pressure']].to_numpy().reshape(-1, 80, 1)
    train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
    RS = RobustScaler()
    X = RS.fit_transform(train)
    X = X.reshape(-1, 80, train.shape[-1])
    print('Finish loading data!')
    return X, y