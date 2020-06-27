import dispatcher
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import utils
import datetime as DT


class M5AccuracyTriner():
    def __init__(self, features_path, models_path):
        super().__init__()
        self.__features_path = features_path
        self.__models_path = models_path
        self.__m5utils = utils.M5AccuracyUtils('temp')
        self.__today = str(DT.date.today())

    def __load_feature_sets(self):
        X_train_final = pd.read_csv(os.path.join(self.__features_path,'X_train.csv'))
        y_train_final = pd.read_csv(os.path.join(self.__features_path,'y_train.csv'))
        return X_train_final, y_train_final


    def train_models(self):
        self.__m5utils.init_event()    
        self.__m5utils.log_event(0,'starting to train...')
        X_train_final, y_train_final = self.__load_feature_sets()
        self.__m5utils.log_event(1,'loaded the features to train...')
        for model in dispatcher.MODELS:
            if (model=='lgb'):    
                self.__m5utils.log_event(2,'starting to train LGB...')
                model = lgb.train(dispatcher.MODELS.get(model), lgb.Dataset(X_train_final, label=y_train_final), 100)
                self.__m5utils.log_event(3,'finished LGB training...')
                model_name = str('lgb') + str(self.__today) + '.pkl'
                joblib.dump(model,os.path.join(self.__models_path,model_name))
                self.__m5utils.log_event(4,'saved the LGB model...')
   
            

if __name__ =='__main__':
    features_path = '..//features//'
    models_path = '..//models//'
    trainer = M5AccuracyTriner(features_path, models_path)
    trainer.train_models()