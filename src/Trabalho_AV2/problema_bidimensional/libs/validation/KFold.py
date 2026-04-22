import numpy as np
from libs.models.ClassificationBayes import ClassificadorGaussianoRegularizado

class KFoldCrossValidationGaussiano:
    def __init__(self, X, y, lambdas, k=5):
        self.X = X
        self.y = y
        self.lambdas = lambdas
        self.k = k
        self.results = {}
        
    def run(self):
        N = self.X.shape[0]
        
        idx = np.random.permutation(N)
        X_embaralhado = np.copy(self.X)[idx, :]
        y_embaralhado = np.copy(self.y)[idx, :]

        X_folds = np.array_split(X_embaralhado, self.k)
        y_folds = np.array_split(y_embaralhado, self.k)

        melhor_lambda = None
        melhor_acuracia = -1

        for lbd in self.lambdas:
            acuracias_do_fold = []

            for i in range(self.k):
                X_teste = X_folds[i]
                y_teste = y_folds[i]

                X_treino = np.vstack([X_folds[j] for j in range(self.k) if j != i])
                y_treino = np.vstack([y_folds[j] for j in range(self.k) if j != i])

                modelo = ClassificadorGaussianoRegularizado(X_treino, y_treino, alpha=lbd)
                modelo.fit()
                y_pred = modelo.predict(X_teste)
                acuracia = np.mean(y_teste == y_pred) 
                acuracias_do_fold.append(acuracia)


            acuracia_media = np.mean(acuracias_do_fold)
            self.results[lbd] = acuracia_media

            if acuracia_media > melhor_acuracia:
                melhor_acuracia = acuracia_media
                melhor_lambda = lbd

        return melhor_lambda, melhor_acuracia, self.results