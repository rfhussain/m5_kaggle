from . import prepare
from . import train
from . import predict

class M5Accuracy():
    def __init__(self, submission_path):
        super().__init__()
        self.__submission_path = submission_path
        self.__days_to_train = 180
        self.__m5_cook = prepare.M5AccuracyCook(self.__submission_path,self.__days_to_train)
        self.__m5_trainer = train.M5Trainer(self.)
        

    def execute_m5(self):
        # data preparation
        self.__m5_cook.cook_data()
        # data training




if __name__ =='__main__':
    submission_path = '..//submissions//'
    m5 = M5Accuracy(submission_path)
    m5.execute_m5()