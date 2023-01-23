import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer


def importation(str1):
    df_train = pd.read_csv(str1)

    train = df_train[['TARGET', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'DEF_30_CNT_SOCIAL_CIRCLE']]

    train['TRAINING'] = True
    y = train.TARGET

    train.drop(columns = 'TARGET', inplace=True)
    x = pd.concat([train], axis = 0)

    return x,y


def preprocess(df):
    cols_to_scale = df.select_dtypes(exclude = ['object', 'bool']).columns.values.tolist()

    # Scaling
    robust_scaler = RobustScaler().fit(df[cols_to_scale])
    df[cols_to_scale] = robust_scaler.transform(df[cols_to_scale])
    # get dummies
    df = pd.get_dummies(df)
    # Imputer 
    imputer = IterativeImputer()
    df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)
    return df


