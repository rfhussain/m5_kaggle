import prepare
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
        self.__m5_trainer = train.M5AccuracyTriner(self.__features_folder,self.__models_folder)
        self.__m5_predictor = predict.M5AccuracyPredictor(self.__features_folder,self.__models_folder,self.__submission_folder,self.__data_folder)
        

    def execute_m5(self,bypass_cooking, bypass_training):
        # data preparation
        if bypass_cooking==False: self.__m5_cook.cook_data()
        # training the models
        if bypass_training==False: self.__m5_trainer.train_models()
        # doing the predictions and saving them
        self.__m5_predictor.predict()






if __name__ =='__main__':

    submission_path = '..//submissions//'
    data_path = '..//data//'
    features_path = '..//features//'
    models_path = '..//models//'
    days_to_train=365
    bypass_cooking = True # False means data preparation will be done again
    bypass_training = True # False means the models will be trained again

    #instantiation of m5 main object
    m5 = M5Accuracy(data_path,features_path,models_path,submission_path,days_to_train)
    m5.execute_m5(bypass_cooking,bypass_training)
    