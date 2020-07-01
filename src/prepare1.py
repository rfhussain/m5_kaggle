import pandas as pd
import numpy as np
import os
import gc
import utils
from datetime import datetime 
import joblib


class M5AccuracyCook1():
    def __init__(self, features_folder, data_folder , days_to_train):
        super().__init__()
        self.__data_folder = data_folder
        self.__features_folder = features_folder
        self.__m5util = utils.M5AccuracyUtils(self.__data_folder)
        self.__days_to_train = days_to_train
        self.__cook_id =1
    
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
        self.__m5util.init_event()
        self.__m5util.log_event(0,f'cook {self.__cook_id} preparing to cook...')

        # 1. read the evaluation data set
        df_eval = pd.read_csv(os.path.join(self.__data_folder,'sales_train_evaluation.csv'))

        self.__m5util.log_event(1,'evaluation dataframe loaded...')

        # 2. melting the data frame
        df_eval = self.__m5util.melt_dataframe(df_eval)

        self.__m5util.log_event(2,'performed the melt of evaluation data frame... ')

        # 3. joining the calendar and prices
        df_eval = self.__m5util.merge_sales_calendar_prices(df_eval,0)        


        self.__m5util.log_event(3,'joined the calendar and prices with evaluation data... ')

        # 4. adding the group by columns
        group_by_cols = ['item_id','dept_id']
        df_eval = self.__m5util.add_group_by_cols(group_by_cols,df_eval)

        self.__m5util.log_event(4,'added the group by cols for... ' + str(group_by_cols))

        # 5. adding the label encodes
        df_eval = self.__m5util.add_label_encodes(df_eval)

        self.__m5util.log_event(5,'performed the label encodes on categorical columns... ')

        # 6. adding the Means
        mean_enc_cols = ['store_id_code','cat_id_code','state_id_code','item_id_code','dept_id_code']
        index_cols = ['item_id_code','store_id_code']
        df_eval = self.__m5util.get_mean_attributes(mean_enc_cols,index_cols,df_eval)

        self.__m5util.log_event(6,'added the mean attributes to...(please refer to prepare.py pipeline #6...)')

        #########################################################
        # IMPORTANT: for time being we are just getting 180 days
        # but will have to get at least 1 year of data
        #########################################################

        # 7. getting last 180 days of data
        df_eval =df_eval[(df_eval.d > (df_eval.d.max() - self.__days_to_train))]

        
        self.__m5util.log_event(7,'restricted the data by ' + str(self.__days_to_train))

        
        #####################EXTRA FUNCTION#####################
        #
        # Deleting additional columns which otherwise should
        # not be deleted when working on a full feature set 
        # however we are currently deleting them
        # ['wm_yr_wk','wday','sell_price']
        #####################EXTRA FUNCTION#####################
        df_eval = self.__m5util.del_additional_cols(df_eval)


        # 8. adding the lags
        mean_enc_cols = [col for col in df_eval.columns if 'mean' in str(col)]
        exception_cols = mean_enc_cols + ['cat_id', 'date', 'day', 'id','sell_price', 'snap_CA', 'snap_TX', 'snap_WI',
                                  'state_id', 'wday', 'wm_yr_wk','year','month','dom']

        index_cols = ['store_id_code','item_id_code','d']
        #shift_range = [x for x in range(1,29)] 
        shift_range = [7,21,28] 
        df_eval = self.__m5util.add_lags(df_eval,shift_range,index_cols,exception_cols)

        self.__m5util.log_event(8,'added lags...')

        # 8.1 adding the rolling means on the lags
        wins=[7,21,28]
        lag_cols = [col for col in df_eval.columns if 'lag' in str(col)]
        shift_range = [7,21,28]
        self.__m5util.add_rolling_means(df_eval,wins,shift_range,lag_cols)
        self.__m5util.log_event(81,'added rolling means on lags...')

        
        #########################################################
        # 9. Train Test Splits
        # The split will be performed and results will be saved
        # The training will be performed in the train.py
        # 
        # 
        # Note: currently we are creating only one feature set
        # but future we will be creating multiple feature set
        #########################################################

        

        # ########## TRAINING DATA (This data will be used to predict for X_valid)
        X_train = df_eval[df_eval.d <= df_eval.d.max()-28]
        y_train = df_eval[df_eval.d <= df_eval.d.max()-28]['target']

        # ########## TRAINING DATA (This data will be used to predict for X_Test) 
        # so without any filtering, we take complete evaluation set, 
        # because the prediction is in the future and data isn't available        
        #X_train_final = df_eval
        #y_train_final = df_eval['target']

        # ########## VALIDATION DATA (for prediction from 1913 ~ 1941) 
        X_valid = df_eval[df_eval.d > df_eval.d.max()-28] # .drop(list(set(drop_columns) - set(['id'])), axis=1)
        y_valid = df_eval[df_eval.d > df_eval.d.max()-28]['target']
        # convert the id from _evauation to _validation
        X_valid['id'] = X_valid['id'].apply(lambda x: x.replace('_evaluation','_validation'))

        # ########## EVALUATION DATA  (for prediction from 1942 ~ 1969) 
        X_test  =  df_eval[df_eval.d > df_eval.d.max()-28] # .drop(list(set(drop_columns) - set(['id'])), axis=1)
        #adding 28 for all the days
        X_test.d = X_test.d + 28 

        # dropping the columns
        drop_columns = ['id','d','target','month','year','dom','target_item_id','target_dept_id']
        X_train.drop(drop_columns, axis=1, inplace=True)
        #X_train_final.drop(drop_columns, axis=1, inplace=True)
        X_valid.drop(list(set(drop_columns) - set(['id','d'])), axis=1, inplace=True)
        X_test.drop(list(set(drop_columns) - set(['id','d'])), axis=1, inplace=True)


        self.__m5util.log_event(9,'data split performed on X_train, X_valid, X_test & X_test_final...')


        # 10. Saving the Records

        # ########## Train Set  (1 ~ 1942-28)  means 1~1913
        X_train.to_csv(os.path.join(self.__features_folder,f'X_train_{self.__cook_id}.csv'), index=False)
        y_train.to_csv(os.path.join(self.__features_folder,f'y_train_{self.__cook_id}.csv'), index=False)

        # ########## Valid Set  (1913 ~ 1942) for prediction
        X_valid.to_csv(os.path.join(self.__features_folder,f'X_valid_{self.__cook_id}.csv'), index=False)
        y_valid.to_csv(os.path.join(self.__features_folder,f'y_valid_{self.__cook_id}.csv'), index=False)

        # ########## Test set (1942 ~ 1969) for prediction
        X_test.to_csv(os.path.join(self.__features_folder,f'X_test_{self.__cook_id}.csv'), index=False)

        # ########## Final Training Set (1 ~ 1942) for training data to predict on test set(1942 ~ 1969)
        #X_train_final.to_csv(os.path.join(self.__features_folder,f'X_train_final_{self.__cook_id}.csv'), index=False)
        #y_train_final.to_csv(os.path.join(self.__features_folder,f'y_train_final_{self.__cook_id}.csv'), index=False)


        self.__m5util.log_event(10,'saved the feature set at ' + str(self.__features_folder)  + ' folder')
