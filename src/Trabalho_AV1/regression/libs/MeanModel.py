import numpy as np

class MeanModel:
    def __init__(self, y_train):
        self.y_train = y_train
        
    def fit(self):
        self.beta_0 = np.mean(self.y_train)
        
    def predict(self, X_teste):
        N = X_teste.shape[0]
        if len(X_teste.shape)>2:
            return np.ones((N,N))*self.beta_0
        return np.ones((N,1))*self.beta_0