from sklearn import ensemble
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb

xgb_params = {
    'cols_sample_by_tree': 0.75,
    'colsample_bylevel':0.75, 
    'eval_metric': 'rmse',
    'nthread': -1,
    'subsample': 0.75,
    'eta': 0.03,
    'max_depth': 9,
    'min_child_weight': 2**7,
    'objective': 'reg:squarederror'
}

lgb_params = {
               'feature_fraction': 0.75,
               'metric': 'rmse',
               'objective': 'rmse',
               'lambda_l2': 0.1,
               'nthread':-1, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.03, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0 
              }

#'rf' : ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
MODELS = {
    'lgb': lgb_params,  
    'xgb': xgb_params,
    'lr' : LinearRegression(n_jobs=-1, normalize=False, fit_intercept=True, copy_X=True)
}