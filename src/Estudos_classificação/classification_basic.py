import os
import numpy as np
import matplotlib.pyplot as plt

caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data", "EMG.csv")
data = np.loadtxt(caminho_arquivo, delimiter=',')
data = data.T

classes = [
    2,4,5
]
nomes_classes = [
    # 'Neutro',
    'Sorriso',
    # 'Sobrancelha levantada',
    'Surpreso',
    'Rabugento',
 ]

cores = [
    'b',
    '#FFBB00',
    # 'pink',
    '#5CFFFF',
    # 'r',
]

C = len(classes)
X = np.empty((0, 2))
Y = np.empty((0, C))

for i,classe in enumerate(classes):
    X_classe = data[data[:,-1]==classe, :-1]    
    plt.scatter(X_classe[:,0], X_classe[:,1], c=cores[i], label = nomes_classes[i],
                edgecolors='k')
    X = np.vstack((
        X, X_classe
    ))
    
    y = -1*np.ones((1,C))
    y[0,i] = 1
    Y_classe = np.tile(y, (X_classe.shape[0],1))
    
    Y = np.vstack((
        Y, Y_classe
    ))
    
X = np.hstack((
    np.ones((X.shape[0],1)),X
))

W_hat = np.linalg.inv(X.T@X)@X.T@Y

x1 = np.linspace(-200,8000,1000)

x2 = -W_hat[0,0]/W_hat[2,0] - W_hat[1,0]/W_hat[2,0]*x1

plt.plot(x1,x2,c='k')

x2 = -W_hat[0,1]/W_hat[2,1] - W_hat[1,1]/W_hat[2,1]*x1
plt.plot(x1,x2,c='r')

x2 = -W_hat[0,2]/W_hat[2,2] - W_hat[1,2]/W_hat[2,2]*x1
plt.plot(x1,x2,c='b')


x_novo = np.array([1, 1544, 1425]).reshape(1,3)

# x1 = np.linspace(-200, 5000, 1000)
# X1, X2 = np.meshgrid(x1, x1)

# X_plot = np.hstack((
#     np.ones((X1.size,1)),
#     X1.flatten().reshape(X1.size,1),
#     X2.flatten().reshape(X1.size,1)
# ))

y_pred  = x_novo @ W_hat
Y_discriminante = np.argmax(y_pred,axis=1)
print(nomes_classes[Y_discriminante[0]])

# y_plot = Y_discriminante.reshape(X1.shape)

# plt.contourf(X1, X2, y_plot, alpha=0.2, cmap='Set3')

plt.scatter(x_novo[0,1], x_novo[0,2], marker='*')
bp = 1
plt.xlim(-50,3800)
plt.ylim(-50,4100)
plt.legend()
plt.show()