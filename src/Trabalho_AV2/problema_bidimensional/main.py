import os
import numpy as np
import matplotlib.pyplot as plt
    
caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spiral_d.csv")
data = np.loadtxt(caminho_arquivo, delimiter=',')
X_data = data[:,:-1]
Y_data = data[:,-1:]
p = len(X_data[0])

print(f"X_data: {X_data}")
print(f"Y_data: {Y_data}")
print(f"P: {p}")


plt.title("Gráfico de espalhamento do dataset")
plt.scatter(X_data[:,0], X_data[:,1], edgecolors='k')
plt.show()

