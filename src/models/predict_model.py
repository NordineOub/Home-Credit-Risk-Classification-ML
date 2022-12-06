from fct.fct_etl import * 
from sklearn.metrics import mean_squared_error
import xgboost as xgb

xg_reg = xgb.Booster()
xg_reg.load_model(model_path + "xgboost.json")

preds = xg_reg.predict(X_test)
