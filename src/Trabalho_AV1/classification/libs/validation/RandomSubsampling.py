import numpy as np
from libs.metrics import metrics as mtcs

class RandomSubsamplingValidation:
    def __init__(self, X, y, models, R=500, train_size=0.8, alpha=0):
        self.X = X
        self.y = y
        self.models = models
        self.R = R
        self.train_size = train_size
        self.results = {}
        self.alpha = alpha
        
    def run(self):
        N = self.X.shape[0]
        split_idx = int(N * self.train_size)

        for name in self.models:
            self.results[name] = {"acuracia": []} 
            
        for _ in range(self.R):
            # Embaralhar os índices
            idx = np.random.permutation(N)
            X_embaralhado = self.X[idx]
            y_embaralhado = self.y[idx]

            # Separação 80% Treino / 20% Teste
            X_treino = X_embaralhado[:split_idx]
            y_treino = y_embaralhado[:split_idx]
            
            X_teste = X_embaralhado[split_idx:]
            y_teste = y_embaralhado[split_idx:]

            for name, model_class in self.models.items():                
                if name == "Gaussiano Regularizado (RDA)":
                    model = model_class(X_treino, y_treino, alpha=self.alpha)
                else:
                    model = model_class(X_treino, y_treino)
                
                model.fit()
                y_pred = model.predict(X_teste)
                if y_teste.ndim > 1 and y_teste.shape[1] > 1:
                    y_teste_classes = np.argmax(y_teste, axis=1)
                    y_pred_classes = np.argmax(y_pred, axis=1)
                    acc = np.mean(y_teste_classes == y_pred_classes)
                else:
                    acc = np.mean(y_teste.ravel() == y_pred.ravel())
                self.results[name]["acuracia"].append(acc)
                
        return self.results