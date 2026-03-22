import os
import numpy as np
import matplotlib.pyplot as plt
from libs.RegressionLinear import LinearRegression

caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "aerogerador.dat")
data = np.loadtxt(caminho_arquivo, delimiter='\t')

X_data = data[:,0].reshape(len(data), 1)
Y_data = data[:,1].reshape(len(data), 1)

bp = 1

N = len(data)
rodadas = 500
k = 5

for r in range(rodadas):
    
    X_embaralhado = np.copy(X_data)[N%k:]
    y_embaralhado = np.copy(Y_data)[N%k:]

    X_treino = X_embaralhado[:int(N*.8), :]
    y_treino = y_embaralhado[:int(N*.8), :]
    
    X_teste = X_embaralhado[int(N*.8):, :]
    y_teste = y_embaralhado[int(N*.8):, :]
    
    model = LinearRegression(X_treino, y_treino)
    model.fit()
    y_pred = model.predict(X_teste)
    
