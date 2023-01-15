import joblib
import pandas as pd

from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

raw_folder = '../../data/raw/'
processed_folder = '../../data/processed/'
model_path = '../../models/'
plot_path = '../../reports/figures/'

def create_df(filepath, filename, compression = None):
    return pd.read_csv(filepath + filename, compression)

def export_df(dataframe, exportpath, exportname):
    dataframe.to_csv(exportpath + exportname)