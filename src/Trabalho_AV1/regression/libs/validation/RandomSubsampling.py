import numpy as np
from libs.metrics import metrics as mtcs

class RandomSubsamplingValidation:
    def __init__(self, X, y, models, R=500, train_size=0.8):
        self.X = X
        self.y = y
        self.models = models
        self.R = R
        self.train_size = train_size
        self.results = {}
        
    def run(self):
        N = self.X.shape[0]

        for name in self.models:
            self.results[name] = {"mse": [],"r2": []}
            
        for _ in range(self.R):
            
            idx = np.random.permutation(N)
            X_embaralhado = np.copy(self.X)[idx, :]
            y_embaralhado = np.copy(self.y)[idx, :]

            X_treino = X_embaralhado[:int(N*.8), :]
            y_treino = y_embaralhado[:int(N*.8), :]
            
            X_teste = X_embaralhado[int(N*.8):, :]
            y_teste = y_embaralhado[int(N*.8):, :]

            for name, model_class in self.models.items():
                model = model_class(X_treino, y_treino)
                model.fit()
                y_pred = model.predict(X_teste)

                self.results[name]["mse"].append(mtcs.mse(y_teste, y_pred))
                self.results[name]["r2"].append(mtcs.r2(y_teste, y_pred))

        return self.results
