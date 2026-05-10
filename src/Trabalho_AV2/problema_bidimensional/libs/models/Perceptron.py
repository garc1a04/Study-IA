import numpy as np

class Perceptron:
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
        return np.where(u >= 0, 1, -1)

    def fit(self, total_epochs=100):
        error = True
        epochs = 0
        historico_erros = []

        while error and epochs <= total_epochs:
            error = False
            erros_na_epoca = 0
            
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                d_k = self.d[0,k]
                u_k = (self.w.T @ x_k)[0,0]
                y_k = self.activation_function(u_k)
                e_k = d_k - y_k
                
                if e_k != 0:
                    error = True
                    erros_na_epoca += 1 
                    self.w = self.w + self.lr * e_k * x_k
                    
            historico_erros.append(erros_na_epoca)
            epochs += 1
        return historico_erros

    def predict(self, X_test):
        N_test = X_test.shape[1]
        
        X_test_bias = np.vstack((
            -np.ones((1, N_test)),
            X_test
        ))
        u = self.w.T @ X_test_bias
        return self.activation_function(u)