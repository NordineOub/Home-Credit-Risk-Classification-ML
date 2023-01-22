from fct.fct_etl import *

#Loading the data
df_train = pd.read_csv(processed_folder + 'processed_train.csv') 
df_test = pd.read_csv(processed_folder + 'processed_test.csv')

#Tagging the different columns
train = df_train[['TARGET', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'DEF_30_CNT_SOCIAL_CIRCLE']]
test = df_test[['EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'DEF_30_CNT_SOCIAL_CIRCLE']]


train['TRAINING'] = True
test['TRAINING'] = False

y = train.TARGET

train.drop(columns = 'TARGET', inplace=True)

#Joining the two datasets for a better training
x = pd.concat([train, test], axis = 0)

cols_to_scale = x.select_dtypes(exclude = ['object', 'bool']).columns.values.tolist()


robust_scaler = RobustScaler().fit(x[cols_to_scale])
standard_scaler = StandardScaler().fit(x[cols_to_scale])
minmax_scaler = MinMaxScaler().fit(x[cols_to_scale])

x[cols_to_scale] = robust_scaler.transform(x[cols_to_scale])

x = pd.get_dummies(x)


imputer = IterativeImputer()

x = pd.DataFrame(imputer.fit_transform(x), columns = x.columns)

#Resetting the datasets
train = x[x['TRAINING'] == True]
test = x[x['TRAINING'] == False]
train.drop(columns = 'TRAINING', inplace=True)
test.drop(columns = 'TRAINING', inplace=True)

#Splitting the datasets
xtrain, xtest, ytrain, ytest = train_test_split(train, y, test_size = .3, random_state = 7)

#Training
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(xtrain, ytrain)

#Exporting the model to the right folder
joblib_file = model_path + "Gradient_boosting_model.pkl"
joblib.dump(clf,joblib_file)