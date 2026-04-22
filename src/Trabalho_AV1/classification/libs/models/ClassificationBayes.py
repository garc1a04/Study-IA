import numpy as np

class ClassificadorGaussianoTradicional:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.mu = {}
        self.sigma = {}
        self.priors = {}

    def fit(self):
        y_train = self.y_train.ravel() # Blindagem contra matriz (N,1)
        N, p = self.X_train.shape
        for c in self.classes:
            X_c = self.X_train[y_train == c]
            self.mu[c] = np.mean(X_c, axis=0)
            self.sigma[c] = np.cov(X_c, rowvar=False) + np.eye(p) * 1e-6
            self.priors[c] = X_c.shape[0] / N

    def predict(self, X_test):
        n_amostras = X_test.shape[0]
        posteriors = np.zeros((n_amostras, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            inv_sigma = np.linalg.inv(self.sigma[c])
            det_sigma = np.linalg.det(self.sigma[c])
            diff = X_test - self.mu[c]
            term1 = -0.5 * np.log(det_sigma)
            term2 = -0.5 * np.sum((diff @ inv_sigma) * diff, axis=1)
            term3 = np.log(self.priors[c])
            posteriors[:, i] = term1 + term2 + term3
            
        predicoes = self.classes[np.argmax(posteriors, axis=1)]
        return predicoes.reshape(-1, 1)

class ClassificadorGaussianoCovarianciasIguais:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.mu = {}
        self.sigma = None
        
    def fit(self):
        y_train = self.y_train.ravel()
        self.sigma = np.cov(self.X_train, rowvar=False) + np.eye(self.X_train.shape[1]) * 1e-6
        for c in self.classes:
            X_c = self.X_train[y_train == c]
            self.mu[c] = np.mean(X_c, axis=0)

    def predict(self, X_test):
        inv_sigma = np.linalg.inv(self.sigma)
        distancias = np.zeros((X_test.shape[0], len(self.classes)))
        
        for i, c in enumerate(self.classes):
            diff = X_test - self.mu[c]
            distancias[:, i] = np.sum((diff @ inv_sigma) * diff, axis=1)
            
        predicoes = self.classes[np.argmin(distancias, axis=1)]
        return predicoes.reshape(-1, 1)

class ClassificadorGaussianoMatrizAgregada:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.mu = {}
        self.sigma_agregada = None 
        
    def fit(self):
        y_train = self.y_train.ravel()
        N, p = self.X_train.shape
        self.sigma_agregada = np.zeros((p, p))
        for c in self.classes:
            X_c = self.X_train[y_train == c]
            self.mu[c] = np.mean(X_c, axis=0)
            self.sigma_agregada += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)
        self.sigma_agregada /= (N - len(self.classes))
        self.sigma_agregada += np.eye(p) * 1e-6

    def predict(self, X_test):
        inv_sigma = np.linalg.inv(self.sigma_agregada)
        distancias = np.zeros((X_test.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            diff = X_test - self.mu[c]
            distancias[:, i] = np.sum((diff @ inv_sigma) * diff, axis=1)
            
        predicoes = self.classes[np.argmin(distancias, axis=1)]
        return predicoes.reshape(-1, 1)
    
class ClassificadorGaussianoRegularizado:
    def __init__(self, X_train, y_train, alpha=0.5):
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha 
        self.classes = np.unique(y_train)
        self.mu = {}
        self.sigma = {}
        self.priors = {}
        
    def fit(self):
        y_train = self.y_train.ravel()
        N, p = self.X_train.shape
        sigma_agregada = np.zeros((p, p))
        cov_classes = {}
        
        for c in self.classes:
            X_c = self.X_train[y_train == c]
            self.mu[c] = np.mean(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / N
            cov_c = np.cov(X_c, rowvar=False)
            cov_classes[c] = cov_c
            sigma_agregada += cov_c * (X_c.shape[0] - 1)
            
        sigma_agregada /= (N - len(self.classes))
        
        for c in self.classes:
            self.sigma[c] = (1 - self.alpha) * cov_classes[c] + self.alpha * sigma_agregada
            self.sigma[c] += np.eye(p) * 1e-6

    def predict(self, X_test):
        posteriors = np.zeros((X_test.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            inv_sigma = np.linalg.inv(self.sigma[c])
            det_sigma = np.linalg.det(self.sigma[c])
            diff = X_test - self.mu[c]
            term1 = -0.5 * np.log(det_sigma)
            term2 = -0.5 * np.sum((diff @ inv_sigma) * diff, axis=1)
            term3 = np.log(self.priors[c])
            posteriors[:, i] = term1 + term2 + term3
            
        predicoes = self.classes[np.argmax(posteriors, axis=1)]
        return predicoes.reshape(-1, 1)

class ClassificadorBayesIngenuo:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.mu = {}
        self.sigma = {}
        self.priors = {}
        
    def fit(self):
        y_train = self.y_train.ravel()
        N = self.X_train.shape[0]
        for c in self.classes:
            X_c = self.X_train[y_train == c]
            self.mu[c] = np.mean(X_c, axis=0)
            variancias = np.var(X_c, axis=0, ddof=1) + 1e-6
            self.sigma[c] = np.diag(variancias)
            self.priors[c] = X_c.shape[0] / N

    def predict(self, X_test):
        posteriors = np.zeros((X_test.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            inv_sigma = np.linalg.inv(self.sigma[c])
            det_sigma = np.linalg.det(self.sigma[c])
            diff = X_test - self.mu[c]
            term1 = -0.5 * np.log(det_sigma)
            term2 = -0.5 * np.sum((diff @ inv_sigma) * diff, axis=1)
            term3 = np.log(self.priors[c])
            posteriors[:, i] = term1 + term2 + term3
            
        predicoes = self.classes[np.argmax(posteriors, axis=1)]
        return predicoes.reshape(-1, 1)