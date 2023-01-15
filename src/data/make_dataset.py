from fct.fct_etl import *


#Loading the datasets
df_test = create_df(raw_folder, 'application_test.csv.zip')
df_train = create_df(raw_folder, 'application_train.csv.zip')

#Cleaning the data
corr_rank = df_train.corr()['TARGET'].sort_values()

#Exporting the processed datasets
export_df(df_train, processed_folder, 'processed_train.csv')
export_df(df_test, processed_folder, 'processed_test.csv')