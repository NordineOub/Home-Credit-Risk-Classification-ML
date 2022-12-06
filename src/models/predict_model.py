from fct.fct_etl import * 
from features.build_features import X_test
from sklearn.metrics import mean_squared_error
import xgboost as xgb

xg_reg = xgb.Booster()
xg_reg.load_model(model_path + "xgboost.json")

preds = xg_reg.predict(X_test)
