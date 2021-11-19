
from project.feature import add_feature
from project.feature import add_lag_feature
from project.feature import add_category_features


def processor_v1(df, mode):
    print('Adding features..........')
    df = add_feature(df)
    df = add_lag_feature(df)
    # df['first_pre'] = df.groupby('breath_id')['pressure'].transform('first')
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    if mode == 'all':
        df = add_category_features(df)
    print('Finish adding features')
    return df



def processor_v2(df, mode):
    print('Adding features..........')
    df = add_feature(df)
    df = add_lag_feature(df)
    # df['first_pre'] = df.groupby('breath_id')['pressure'].transform('first')
    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')
    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    #df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    if mode == 'all':
        df = add_category_features(df)
    print('Finish adding features')
    return df



def processor_v3(df, mode):
    print('Adding features..........')
    df = add_feature(df)
    df = add_lag_feature(df)
    # df['first_pre'] = df.groupby('breath_id')['pressure'].transform('first')
    df['breath_id__u_in__mean'] = df.groupby(['breath_id'])['u_in'].transform('mean')
    df['breath_id__u_out__mean'] = df.groupby(['breath_id'])['u_out'].transform('mean')
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']
    if mode == 'all':
        df = add_category_features(df)
    print('Finish adding features')
    return df


DATASET_PROCESSOR_MAP = {
    'v1': processor_v1,
    'v2': processor_v2,
    'v3': processor_v3
}