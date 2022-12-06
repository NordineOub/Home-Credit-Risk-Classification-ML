from fct.fct_etl import * 

df_train = create_df(processed_folder, 'processed_train.csv')
df_test = create_df(processed_folder, 'processed_test.csv')

X_train, y_train = df_train.iloc[:,:-1],df_train.iloc[:,-1]
X_test, y_test = df_test.iloc[:,:-1],df_test.iloc[:,-1]

