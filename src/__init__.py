import prepare
import prepare1
import prepare2
import prepare3
import train
import predict

class M5Accuracy():
    def __init__(self,data_path,features_path,models_path, submission_path,days_to_train):
        super().__init__()
        self.__submission_folder = submission_path
        self.__features_folder = features_path
        self.__data_folder = data_path
        self.__models_folder = models_path
        self.__days_to_train = days_to_train
        self.__m5_cook = prepare.M5AccuracyCook(self.__features_folder,self.__data_folder,self.__days_to_train)        
        self.__m5_cook1 = prepare1.M5AccuracyCook1(self.__features_folder,self.__data_folder,self.__days_to_train)        
        self.__m5_cook2 = prepare2.M5AccuracyCook2(self.__features_folder,self.__data_folder,self.__days_to_train)        
        self.__m5_cook3 = prepare3.M5AccuracyCook3(self.__features_folder,self.__data_folder,self.__days_to_train)        
        self.__m5_trainer = train.M5AccuracyTriner(self.__features_folder,self.__models_folder)
        self.__m5_predictor = predict.M5AccuracyPredictor(self.__features_folder,self.__models_folder,self.__submission_folder,self.__data_folder)
        

    def execute_m5(self,bypass_cooking, bypass_training):
        # data preparation
        if bypass_cooking==False: 
            #self.__m5_cook.cook_data()
            #self.__m5_cook1.cook_data()
            #print('\n\n\n')
            #self.__m5_cook2.cook_data()
            #print('\n\n\n')
            self.__m5_cook3.cook_data()

        # training the models
        #if bypass_training==False: self.__m5_trainer.train_models()
        # doing the predictions and saving them
        #self.__m5_predictor.predict()
        # perform stacking

        # perform boosting

        # perform averaging


if __name__ =='__main__':

    submission_path = '..//submissions//'
    data_path = '..//data//'
    features_path = '..//features//'
    models_path = '..//models//'
    days_to_train=1100
    bypass_cooking = False # False means data preparation will be done again
    bypass_training = True # False means the models will be trained again

    #instantiation of m5 main object
    m5 = M5Accuracy(data_path,features_path,models_path,submission_path,days_to_train)
    m5.execute_m5(bypass_cooking,bypass_training)
    