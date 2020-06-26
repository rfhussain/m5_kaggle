import pandas as pd
import numpy as np
import os
import gc
from . import utils

class M5AccuracyCook():
    def __init__(self, submission_path, days_to_train):
        super().__init__()
        self.submission_path = submission_path
        self.__data_folder = '..//data//'
        self.__features_folder = '..//features//'
        self.__m5util = utils.M5AccuracyUtils(self.submission_path)
        self.__days_to_train = days_to_train
        
    
    def cook_data(self):
        '''
        This is a data preparation class method
        This method will cook the data for us and create multipe feature sets, to train the data
        '''
        #######################################################
        # Data Preparation Pipeline.
        #
        #
        # Remember, that scikit-learn also provides a library
        # for pipeline, where we can add even the training of 
        # model in the pipeline
        #######################################################


        # 1. read the evaluation data set
        df_eval = pd.read_csv(os.path.join(self.__data_folder,'sales_train_evaluation.csv'))

        # 2. melting the data frame
        df_eval = self.__m5util.melt_dataframe(df_eval)

        # 3. joining the calendar and prices
        df_eval = self.__m5util.merge_sales_calendar_prices(df_eval)

        # 4. adding the group by columns
        group_by_cols = ['item_id','dept_id']
        df_eval = self.__m5util.add_group_by_cols(group_by_cols,df_eval)

        # 5. adding the label encodes
        df_eval = self.__m5util.add_label_encodes(df_eval)

        # 6. adding the Means
        mean_enc_cols = ['store_id_code','cat_id_code','state_id_code','item_id_code','dept_id_code']
        index_cols = ['item_id_code','store_id_code']
        df_eval = self.__m5util.get_mean_attributes(mean_enc_cols,index_cols,df_eval)

        #########################################################
        # IMPORTANT: for time being we are just getting 180 days
        # but will have to get at least 1 year of data
        #########################################################

        # 7. getting last 180 days of data
        df_eval =df_eval[(df_eval.d > (df_eval.d.max() - self.__days_to_train))]


        # 8. adding the lags
        mean_enc_cols = [col for col in df_eval.columns if 'mean' in str(col)]
        exception_cols = mean_enc_cols + ['cat_id', 'date', 'day', 'id','sell_price', 'snap_CA', 'snap_TX', 'snap_WI',
                                  'state_id', 'wday', 'wm_yr_wk','year','month','dom']

        index_cols = ['store_id_code','item_id_code','d']
        shift_range = [x for x in range(1,10)] 

        df_eval = self.__m5util.add_lags(df_eval,shift_range,index_cols,exception_cols)


        #########################################################
        # 9. Train Test Splits
        # The split will be performed and results will be saved
        # The training will be performed in the train.py
        # 
        # 
        # Note: currently we are creating only one feature set
        # but future we will be creating multiple feature set
        #########################################################

        drop_columns= ['id','target','target_dept','target_item','month','year','dom']

        # ########## TRAINING DATA (This data will be used to predict for X_valid)
        X_train = df_eval[df_eval.d <= df_eval.d.max()-28].drop(drop_columns, axis=1) 
        y_train = df_eval[df_eval.d <= df_eval.d.max()-28]['target']

        # ########## TRAINING DATA (This data will be used to predict for X_Test) 
        # so without any filtering, we take complete evaluation set, 
        # because the prediction is in the future and data isn't available        
        X_train_final = df_eval.drop(drop_columns, axis=1) 
        y_train_final = df_eval['target']

        # ########## VALIDATION DATA (for prediction from 1913 ~ 1941) 
        X_valid = df_eval[df_eval.d > df_eval.d.max()-28].drop(list(set(drop_columns) - set(['id'])), axis=1)
        y_valid = df_eval[df_eval.d > df_eval.d.max()-28]['target']
        # convert the id from _evauation to _validation
        X_valid['id'] = X_valid['id'].apply(lambda x: x.replace('_evaluation','_validation'))

        # ########## EVALUATION DATA  (for prediction from 1942 ~ 1969) 
        X_test  =  df_eval[df_eval.d > df_eval.d.max()-28].drop(list(set(drop_columns) - set(['id'])), axis=1)
        #adding 28 for all the days
        X_test.d = X_test.d + 28 


        # 10. Saving the Records
        X_train.to_csv(os.path.join(self.__features_folder,'X_train.csv'))
        y_train.to_csv(os.path.join(self.__features_folder,'y_train.csv'))

        X_valid.to_csv(os.path.join(self.__features_folder,'X_valid.csv'))
        y_valid.to_csv(os.path.join(self.__features_folder,'y_valid.csv'))

        X_test.to_csv(os.path.join(self.__features_folder,'X_test.csv'))

        X_train_final.to_csv(os.path.join(self.__features_folder,'X_train_final.csv'))
        y_train_final.to_csv(os.path.join(self.__features_folder,'y_train_final.csv'))
