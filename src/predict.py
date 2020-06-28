import dispatcher
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import joblib
import utils
import datetime as DT

class M5AccuracyPredictor():
    def __init__(self, features_path, models_path, submission_path, data_path):
        super().__init__()
        self.__features_path = features_path
        self.__models_path = models_path
        self.__data_folder = data_path
        self.__submission_folder = submission_path
        self.__today = str(DT.date.today())
        self.__m5utils = utils.M5AccuracyUtils('temp')
        

    def __load_feature_sets(self):
        X_valid = pd.read_csv(os.path.join(self.__features_path,'X_valid.csv'))
        y_test = pd.read_csv(os.path.join(self.__features_path,'X_test.csv'))
        return X_valid, y_test

    def __load_models(self):
        model = joblib.load(os.path.join(self.__models_path,'lgb2020-06-27.pkl'))
        return model

   

    def predict(self):
        self.__m5utils.init_event()    
        self.__m5utils.log_event(0,'starting to predict...')
        # loading the feature sets to predict on
        X_valid, X_test = self.__load_feature_sets()
        self.__m5utils.log_event(1,'loaded feature sets...')

        # loading the model
        model = self.__load_models()
        self.__m5utils.log_event(2,'loaded the models...')
        
        # predictiona
        preds_valid = model.predict(X_valid.drop(['id'], axis=1))
        preds_eval =  model.predict(X_test.drop(['id'], axis=1))
        self.__m5utils.log_event(3,'predictions done....')


        df_submission_valid= X_valid[['id','d']]
        df_submission_valid['target'] = preds_valid.clip(0,preds_valid.max())


        df_submission_eval = X_test[['id','d']]
        df_submission_eval['target'] = preds_eval.clip(0,preds_eval.max())

        #generating the prediction columns
        col_append = ['F'+str(i) for i in range(1,29)]


        #1. Converting 

        #valid set
        submission_pivot_valid = df_submission_valid.pivot_table('target', ['id'], 'd')
        submission_pivot_valid.columns = col_append
        submission_pivot_valid = submission_pivot_valid.reset_index()

        #eval set
        submission_pivot_eval = df_submission_eval.pivot_table('target', ['id'], 'd')
        submission_pivot_eval.columns = col_append
        submission_pivot_eval = submission_pivot_eval.reset_index()

        #Concatenating
        df_submission = pd.concat([submission_pivot_valid, submission_pivot_eval], axis=0)

        

        # loading the sample submission file from m5
        df_sample_sub = pd.read_csv(os.path.join(self.__data_folder,'sample_submission.csv'))

        # merging with the sample submission
        df_submission = df_sample_sub[['id']].merge(df_submission, how='inner', on='id')
        self.__m5utils.log_event(4,'predictions prepared....')

        # saving the submission
        model_name = 'lgb'
        file_name =  'sub_' + model_name
        save_path = self.__submission_folder +  file_name + '_' + str(self.__today) + '.csv'
        df_submission.to_csv(save_path, index=False)
        self.__m5utils.log_event(5,'predictions saved....')




if __name__ =='__main__':
    features_path = '..//features//'
    models_path = '..//models//'
    submission_path = '..//submissions//'
    data_path = '..//data//'
    
    predictor = M5AccuracyPredictor(features_path,models_path,submission_path,data_path)
    predictor.predict()




