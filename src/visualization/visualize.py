import shap
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt



joblib_file = "Gradient_boosting_model.pkl"

clf = joblib.load(joblib_file)

clf = joblib.load(joblib_file)

def create_df(filepath, filename, compression = None):
    return pd.read_csv(filepath + filename, compression)

processed_folder = '../data/processed/'

#Loading the datasets
df_test = create_df(processed_folder, 'processed_test.csv')
df_train = create_df(processed_folder, 'processed_train.csv')

train = df_train[['TARGET', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'DEF_30_CNT_SOCIAL_CIRCLE']]
test = df_test[['EXT_SOURCE_3', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'DAYS_EMPLOYED', 'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'DEF_30_CNT_SOCIAL_CIRCLE']]

train['TRAINING'] = True
test['TRAINING'] = False
y = train.TARGET

train.drop(columns = 'TARGET', inplace=True)

x = pd.concat([train, test], axis = 0)
x = pd.get_dummies(x)

imputer = IterativeImputer()

x = pd.DataFrame(imputer.fit_transform(x), columns = x.columns)

train = x[x['TRAINING'] == True]
test = x[x['TRAINING'] == False]
train.drop(columns = 'TRAINING', inplace=True)
test.drop(columns = 'TRAINING', inplace=True)

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size = .3, random_state = 7)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(clf)

shap.initjs()

shap_values = explainer.shap_values(x_train)

shap.dependence_plot('DAYS_EMPLOYED', shap_values, x_train)

shap_values = explainer.shap_values(x_train[:2])

shap.force_plot(explainer.expected_value, shap_values[0, :], x_test.iloc[0, :])

shap_values_gbd = explainer(x_train)

shap.summary_plot(shap_values_gbd, show=False)

point_index = 0
shap_values = explainer.shap_values(x_train[:2])

# plot the summary of the point feature_names=feature_names
shap.summary_plot(shap_values, color = 'red',alpha = 0.5)

shap.plots.bar(shap_values_gbd)

shap.plots.bar(shap_values_gbd.abs.max(0))

shap.plots.beeswarm(shap_values_gbd)

shap.plots.heatmap(shap_values_gbd[:1000])

shap.plots.bar(shap_values_gbd.abs.sum(0))