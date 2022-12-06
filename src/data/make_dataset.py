from fct.fct_etl import *
import pandas as pd

df_test = create_df('application_test.csv.zip')
df_train = create_df('application_train.csv.zip')

df_train = df_train.dropna()
df_train = df_train.select_dtypes(exclude=['object'])

df_test = df_test.dropna()
df_test = df_test.select_dtypes(exclude=['object'])

export_df(df_train, 'processed_train.csv')
export_df(df_test, 'processed_test.csv')