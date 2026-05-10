import numpy as np

class MultilayerPerceptron:
    def __init__(self, topology, X_train, Y_train, learning_rate, max_epochs, precision):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.pr = precision
        self.topology = topology
        self.D = Y_train
        m = Y_train.shape[0]
        self.p, self.N = X_train.shape
        self.topology.append(m)
        
        self.X_train = np.vstack((
            -np.ones((1, self.N)), X_train
        ))
        
        self.W = []
        for i, q in enumerate(self.topology):
            if i == 0:
                W = np.random.random_sample((q, self.p + 1)) - 0.5
            else:
                W = np.random.random_sample((q, self.topology[i-1] + 1)) - 0.5
            self.W.append(W)
            
        self.u = [None] * len(self.W)
        self.y = [None] * len(self.W)
        self.delta = [None] * len(self.W)

    def g(self, u):
        s = np.exp(-u)
        return (1 - s) / (1 + s)
    
    def g_d(self, u):
        s = self.g(u)
        return 0.5 * (1 - s**2)

    def eqm(self):
        eqm = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            self.forward(x_k)
            d_k = self.D[:, k].reshape(self.topology[-1], 1)
            eqm += np.sum((d_k - self.y[-1])**2)
        return eqm / (2 * self.N)

    def forward(self, x):
        for j, W in enumerate(self.W):
            if j == 0:
                self.u[j] = W @ x
            else:
                yb = np.vstack((
                    -np.ones((1, 1)), self.y[j-1]
                ))
                self.u[j] = W @ yb
                
            self.y[j] = self.g(self.u[j])

    def backward(self, x, d):
        for j in range(len(self.W)-1, -1, -1):
            if j == len(self.W)-1:
                self.delta[j] = self.g_d(self.u[j]) * (d - self.y[-1])
                yb = np.vstack((
                    -np.ones((1, 1)), self.y[j-1]
                ))
                self.W[j] = self.W[j] + self.lr * self.delta[j] @ yb.T
            elif j == 0:
                Wnb = ((self.W[j+1])[:, 1:]).T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb @ self.delta[j+1])
                self.W[j] = self.W[j] + self.lr * self.delta[j] @ x.T
            else:
                Wnb = ((self.W[j+1])[:, 1:]).T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb @ self.delta[j+1])
                yb = np.vstack((
                    -np.ones((1, 1)), self.y[j-1]
                ))
                self.W[j] = self.W[j] + self.lr * self.delta[j] @ yb.T

    def fit(self):
        self.historico_erros = []
        epochs = 0
        EQM = self.eqm()
        self.historico_erros.append(EQM) 
        
        while epochs < self.max_epochs and EQM > self.pr:
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                d_k = self.D[:, k].reshape(self.topology[-1], 1)
                self.forward(x_k)
                self.backward(x_k, d_k)
                
            epochs += 1
            EQM = self.eqm()
            self.historico_erros.append(EQM)
            
        return self.historico_erros

    def predict(self, X_test):
        N_test = X_test.shape[1]
        X_ext = np.vstack((
            -np.ones((1, N_test)), X_test
        ))
        
        predictions = np.zeros((self.topology[-1], N_test))
        
        for k in range(N_test):
            x_k = X_ext[:, k].reshape(self.p + 1, 1)
            self.forward(x_k)
            predictions[:, k] = self.y[-1].ravel()
            
        return predictions