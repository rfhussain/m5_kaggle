import prepare
import train
import predict

class M5Accuracy():
    def __init__(self,data_path,features_path, submission_path,days_to_train):
        super().__init__()
        self.__submission_folder = submission_path
        self.__features_folder = features_path
        self.__data_folder = data_path
        self.__days_to_train = days_to_train
        self.__m5_cook = prepare.M5AccuracyCook(self.__features_folder,self.__data_folder,self.__days_to_train)        
        #self.__m5_trainer = train.M5Trainer(self.)
        

    def execute_m5(self):
        # data preparation
        self.__m5_cook.cook_data()
        # data training


if __name__ =='__main__':

    submission_path = '..//submissions//'
    data_path = '..//data//'
    features_path = '..//features//'
    days_to_train=180

    m5 = M5Accuracy(data_path,features_path,submission_path,days_to_train)
    m5.execute_m5()
    