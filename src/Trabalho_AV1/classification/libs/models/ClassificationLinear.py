import numpy as np

class ClassificationLinear:
    def __init__(self, X_train, y_train, fit_intercept = True):
        self.X_train = X_train
        self.y_train = y_train
        self.fit_intercept = fit_intercept
        self.N , self.p = X_train.shape
        if fit_intercept: 
            self.X_train = np.hstack((np.ones((self.N,1)), X_train))
        self.weigth_hat = None
        
    def fit(self):
        self.weigth_hat = np.linalg.inv(self.X_train.T @ self.X_train) @ self.X_train.T @ self.y_train

    def predict(self, X_test):
        if self.fit_intercept:
            N = X_test.shape[0]
            vet_one = np.ones((N,1))
            X_test = np.hstack((vet_one, X_test))
            
        return X_test @ self.weigth_hat
    
    def straight(self, class_index, x1_space):
        return -self.weigth_hat[0, class_index]/ self.weigth_hat[2, class_index] - self.weigth_hat[1, class_index] / self.weigth_hat[2, class_index] * x1_space