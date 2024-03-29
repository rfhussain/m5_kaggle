import pandas as pd
import numpy as np
import gc
import os
from tqdm import tqdm_notebook
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")




class M5AccuracyUtils():

    def __init__(self,data_folder):
        super().__init__()
        self.__data_folder = data_folder
        #self.init_event()

    def add_rolling_means(self,df,wins,shift_range,lag_cols):
        '''
        this function will add the rolling mean with the specified window
        parameters:
        df : the data frame to add the mean to 
        win: the window to add the rolling mean
        shift_range: the lag range
        '''
        for win in tqdm(wins):
            for lag,lag_col in zip(shift_range, lag_cols):
                df[f"rmean_{lag}_{win}"] = df[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())
        
        #dropping the nulls
        df.dropna(inplace=True)

        #type casting
        for mean_col in [col for col in df.columns if 'rmean' in str(col)]:
            df[mean_col] = df[mean_col].astype('int16')  

        


    def get_mean_attributes(self, mean_cols,index_cols, df):
        '''
        This function will add the required mean attributes
        Parameters:
        - mean_cols : list like ['cols_1','col_2',.....]
        - index_cols: list .. for this dataframe ['item_id_code','item_id_code']
        - df : mean data frame to add mean to        
        '''
        for col in mean_cols:
            mean_attrib = col + '_target_mean'
            mean_values = df.groupby(col).target.mean()
            df[mean_attrib] = df[col].map(mean_values)
            # deleting the columns after adding the mean
            if col not in index_cols:
                df.drop([col], axis=1, inplace=True)
            #type casting
            df[mean_attrib] = df[mean_attrib].astype('float16')
        
        gc.collect()
        # returning the resulting df
        return df

    def del_additional_cols(self,df):
        '''
        This function is temporary
        Remember, that for basic data for training, we are deleting some wanted colummns, which are not to be deleted
        '''
        df = df.drop(['wm_yr_wk','wday','sell_price'], axis=1)
        return df

    def __get_d_of_m(self,col): 
        return col.day

    def __get_calendar_df(self):
        '''
        - This function will return the calendar data frame 
        - It will also remove d_ from the day column
        - and type cast it to int
        '''
        d_parser = lambda x: pd.datetime.strptime(x,'%Y-%m-%d')
        df = pd.read_csv(os.path.join(self.__data_folder,'calendar.csv'), parse_dates=["date"], date_parser=d_parser)
        df['d'] = df['d'].apply(lambda x: x.replace('d_',''))
        df['d'] = df['d'].astype('int16')
        gc.collect()

        #get day of month
        df['dom'] = df['date'].apply(self.__get_d_of_m)

        #dropping the date column
        df.drop(['date'], axis=1, inplace=True)

        #returning the data frame        
        return df


    def __get_prices_df(self):
        df = pd.read_csv(os.path.join(self.__data_folder,'sell_prices.csv'))
        return df

    def merge_sales_calendar_prices(self,df,cat_col):
        df_cal = self.__get_calendar_df()
        df_prices = self.__get_prices_df()

        #merging sales with calendar
        if int(cat_col)==0:
            calendar_merge_cols = ['wm_yr_wk','wday','d','month','year','dom']
        elif int(cat_col)==1:
            calendar_merge_cols = ['wm_yr_wk','wday','d','month','year','dom','snap_CA','snap_TX','snap_WI']

        df = df.merge(df_cal[calendar_merge_cols], on=['d'], how='left')

        
        #merging sales with prices
        if int(cat_col)==0:
            df = df.merge(df_prices, how='left', on=['store_id','item_id','wm_yr_wk'])
        else:
            df = df.merge(df_prices, how='left', on=['store_id','item_id','wm_yr_wk'])

        #type casting the resulting columns
        df.wm_yr_wk     = df.wm_yr_wk.astype('int16')
        df.wday         = df.wday.astype('int8')
        df.month        = df.month.astype('int8')
        df.dom          = df.dom.astype('int8')
        df.year         = df.year.astype('int16')
        df.sell_price   = df.sell_price.astype('float16')
        
        if int(cat_col)==1:
            df.snap_CA   = df.snap_CA.astype('int8')
            df.snap_TX   = df.snap_TX.astype('int8')
            df.snap_WI   = df.snap_WI.astype('int8')
            


        #deleting the dataframes not required
        del df_cal, df_prices
        gc.collect()

        #returning the dataframe
        return df

    def melt_dataframe(self,df):
        
        # we will only work on the evaluation dataset
        df_rows = df.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name ='d',
        value_name ='target'
        )

        #remove the d_
        df_rows['d']   = df_rows['d'].apply(lambda x: x.replace('d_',''))

        #down casting
        df_rows.d        = df_rows.d.astype('int16')
        df_rows.target   = df_rows.target.astype('int16')

        gc.collect()

        return df_rows

    def add_group_by_cols(self,cols,df):
        '''
        This function will add the group by columns to the df and return the resulting df
        Parameters:
        - cols : a list of columns ['col-1','col-2',...]
        - df   : the main data frame where to add the group by        
        '''
        for col in cols:
            gb = df.groupby([col,'d'], as_index=False).target.sum()
            gb.rename(columns = {'target':'target_' + str(col)}, inplace=True)
            df = df.merge(gb, how='left', on=[col,'d']).fillna(0)
            del gb
            gc.collect()
        #returning the data frame
        return df

    def add_label_encodes(self,df):
        '''
        This function will add the label encoders on the following columns
        dept_id, cat_id, store_id, state_id, item_id

        It will delete the above columns since the label encode has been added
        '''
        #adding the label encoding
        le = LabelEncoder()
        df['dept_id_code'] = le.fit_transform(df.dept_id)
        df['cat_id_code'] = le.fit_transform(df.cat_id)
        df['store_id_code'] = le.fit_transform(df.store_id)
        df['state_id_code'] = le.fit_transform(df.state_id)
        df['item_id_code'] = le.fit_transform(df.item_id)

        #deleting the columns
        df.drop(['dept_id','cat_id','store_id','state_id','item_id'], axis=1, inplace=True)

        #type casting to save memory
        df.dept_id_code = df.dept_id_code.astype('int8')
        df.cat_id_code  = df.cat_id_code.astype('int8')
        df.store_id_code= df.store_id_code.astype('int8')
        df.state_id_code= df.state_id_code.astype('int8')
        df.item_id_code = df.item_id_code.astype('int16')

        #returning the df
        return df

    def add_lags(self, df, shift_range, index_cols, exception_cols):
        '''
        This function will add the lags to the data frame

        Parameters: 
        - df : the data frame to add lags to
        - shift_range : list of how many lags like [1,2,3,...]
        - index_cols : list of index cols in our case ['item_id_code','store_id_code']
        - exception_cols : list of those columns which are not supposed to be lags ['col1','col2',....]
        '''

        #getting the lag columns
        cols_to_rename = list(df.columns.difference(index_cols + exception_cols)) 

        for day_shift in tqdm(shift_range):

            train_shift = df[index_cols + cols_to_rename].copy()
            #print('copied to train_shift...')

            train_shift['d'] = train_shift['d'] + day_shift
            #print(f'added lag by adding {day_shift} to [d] column...')
            
            foo = lambda x: '{}_lag_{}'.format(x, day_shift) if x in cols_to_rename else x
            train_shift = train_shift.rename(columns=foo)
            #print('renamed the columns as col_lag_x...')

            df = pd.merge(df, train_shift, on=index_cols, how='left').fillna(0)
            #print(f'performed the merge ---------{day_shift}--------------..\n')
            #print('*'*day_shift, day_shift)

        # type casting the lag columns (in my opinion lags are int values)
        for lag_col in [col for col in df.columns if 'lag' in str(col)]:
            df[lag_col] = df[lag_col].astype('int16')  
        #print('type casted all lag columns to int16 (please note: i did int because sample submission file has int values for results)...')

        # returning the resulting data frame
        return df
    
    def log_event(self,sn,msg,df):
        if df is not None:
            err_msg = msg + '\r columns for df :' + str(df.columns)
            self.__process_log(err_msg)

        print('{:<5d}{:<77}{:<11}'.format(sn,msg,datetime.now().strftime("%H:%M:%S")))        

    def init_event(self):
        print('\n')
        print('{:5}{:77}{:11}'.format('SN','Event Performed','Time Elapsed'))
        print('-----------------------------------------------------------------------------------------------')

    def __process_log(self,errtxt):
        f = open('logs.txt','a+')
        f.writelines('' + errtxt + '\r\r\r')
        f.close()











    

    






