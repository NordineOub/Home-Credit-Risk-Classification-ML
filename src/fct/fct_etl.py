import pandas as pd

raw_folder = '../../data/raw/'
processed_folder = '../../data/processed/'
model_path = '../../models/'
plot_path = '../../reports/figures/'

def create_df(filepath, filename, compression = None):
    return pd.read_csv(filepath + filename, compression)

def export_df(dataframe, exportpath, exportname):
    dataframe.to_csv(exportpath + exportname)

