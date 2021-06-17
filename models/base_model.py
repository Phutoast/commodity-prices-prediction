
class BaseModel(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def train(self):
        """
        Training the data given the training parameters.
        """
        pass
    
    def predict(self, x_pred, y_pred):
        """
        Predict the data given data given the current model
        """
        pass
