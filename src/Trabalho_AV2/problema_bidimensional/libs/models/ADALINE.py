import numpy as np

class ADALINE:
    def __init__(self,X_train, y_train, learning_rate):
        self.d = y_train
        self.lr = learning_rate
        self.p,self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)),
            X_train
        ))
        self.w = np.random.random_sample((self.p+1,1))-.5

    def activation_function(self, u):
        return 1 if u>=0 else -1

    def fit(self, epsilon=0.001):
        error = True
        epochs = 0
        EQM1 = float('inf')

        while error:
            erro_i = 0
            error = False
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                d_k = self.d[0,k]
                u_k = (self.w.T@x_k)[0,0]
                e_k = d_k - u_k
                erro_i +=  (1/2) * (e_k**2)
                self.w = self.w + self.lr * e_k * x_k
            
            EQM2 = erro_i / self.N
            if(abs(EQM2 - EQM1) >= epsilon):
                EQM1 = erro_i / self.N
                error = True
            epochs+=1

    def predict(self, X_test):
        X_test = np.vstack((
            -np.ones((1,self.N)),
            X_test
        ))
        u = self.w.T@X_test
        return self.activation_function(u)