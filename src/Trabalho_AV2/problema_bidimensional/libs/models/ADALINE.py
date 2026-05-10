import numpy as np

class ADALINE:
    def __init__(self, X_train, y_train, learning_rate):
        self.d = y_train.reshape(1, -1)
        self.lr = learning_rate
        self.p, self.N = X_train.shape
        
        self.X_train = np.vstack((
            -np.ones((1, self.N)),
            X_train
        ))

        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5

    def activation_function(self, u):
        return np.where(u >= 0, 1, -1)

    def fit(self, epsilon=1e-6, total_epochs=100):
        error = True
        epochs = 0
        prev_mse = float('inf')
        historico_mse = []

        while error and epochs < total_epochs:
            sum_sq_error = 0
            
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(-1, 1)
                d_k = self.d[0, k]
                u_k = (self.w.T @ x_k)[0, 0]
                e_k = d_k - u_k
                sum_sq_error += e_k**2
                self.w = self.w + self.lr * e_k * x_k
            
            curr_mse = sum_sq_error / self.N
            historico_mse.append(curr_mse)
            
            if abs(curr_mse - prev_mse) < epsilon:
                error = False
            
            prev_mse = curr_mse
            epochs += 1
            
        return historico_mse

    def predict(self, X_test):
        n_samples = X_test.shape[1]
        X_test_bias = np.vstack((
            -np.ones((1, n_samples)),
            X_test
        ))
        
        u = self.w.T @ X_test_bias
        return self.activation_function(u)