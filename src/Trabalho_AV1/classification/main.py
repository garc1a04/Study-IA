import os
import numpy as np
import matplotlib.pyplot as plt
from libs.models.ClassificationLinear import ClassificationLinear
from libs.validation.RandomSubsampling import RandomSubsamplingValidation

"""
Organiza os dados do arquivo em matrizes NumPy (X e Y).

Dimensões conforme o modelo:
- MQO: X (N, p) e Y (N, C)
"""

caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "EMGsDataset.csv")
data = np.loadtxt(caminho_arquivo, delimiter=',')

data = data.T

classes = [
    1,2,3,4,5
]
nomes_classes = [
    'Neutro',
    'Sorriso',
    'Sobrancelha levantada',
    'Surpreso',
    'Rabugento',
 ]

cores = [
    'b',
    "#FF00BF",
    "#00FF40",
    '#5CFFFF',
    'r',
]

C = len(classes)
X = np.empty((0, 2))
Y = np.empty((0, C))

for i,classe in enumerate(classes):
    X_data = data[data[:,-1]==classe, :-1]    
    plt.scatter(X_data[:,0], X_data[:,1], c=cores[i], label = nomes_classes[i],
                edgecolors='k')
    X = np.vstack((
        X, X_data
    ))
    
    y = -1*np.ones((1,C))
    y[0,i] = 1
    Y_data = np.tile(y, (X_data.shape[0],1))
    
    Y = np.vstack((
        Y, Y_data
    ))

"""
N = 50000
C = 5
P = 2

X = (50000, 2)
Y = (50000, 5)
"""

print(X.shape)
print(Y.shape)

"""
análise exploratória visual dos dados.

Gera um gráfico de espalhamento colorido por categoria para:
1. Avaliar a separabilidade das classes (linear vs. não-linear).
2. Identificar sobreposições
3. Formular hipóteses para a escolha do classificador.

R1: Claramente elas não são linearmente separaveis devido a forma como os dado se entrelação.

- Neutro com o surpreso, existem diversos dados quase juntos.

- Rabugento e surpreso também tem vários dados juntos.

- Fazendo esse problema sendo melhorado de forma não linear

R2: Como dito anteriormente, Neutro com supreso, Rabugento e Neutro além de diversas classes possuirem diversos
valores no ponto 0,0.

3. Formular hipóteses para a escolha do classificador.

R: MQO irá sofrer com relação a esses dados, pois existem diversos valores no 0,0;Tendo assim um problema na decisão do bias, além da classe Sobrancelha levantada e Sorriso, possuem valores muitos pertos no dos eixos.
"""
plt.grid(True)
plt.xlim(-50,3800)
plt.ylim(-50,4100)
plt.legend()
plt.show()

"""
Implementação de modelos de classificação:
- MQO: Abordagem via Mínimos Quadrados. (feito).
- Gaussianos: Versões Tradicional, Covariâncias Iguais e Matriz Agregada.
- Regularização: Modelo de Friedman para matrizes de covariância.
- Probabilístico: Naive Bayes (Independência condicional).
"""

classification_traditional = ClassificationLinear(X,Y)
classification_traditional.fit()
x1 = np.linspace(-200, 5000, 1000)

plt.figure()
for i,classe in enumerate(classes):
    X_data = data[data[:,-1]==classe, :-1]    
    plt.scatter(X_data[:,0], X_data[:,1], c=cores[i], label = nomes_classes[i],
                edgecolors='k')
    
for class_idx in range(classification_traditional.weigth_hat.shape[1]):
    x2 = classification_traditional.straight(class_idx, x1)
    plt.plot(x1, x2, label=f'Class {class_idx}')

x1_plot, x2_plot = np.meshgrid(x1, x1)

X_plot = np.hstack((
    x1_plot.flatten().reshape(x1_plot.size,1),
    x2_plot.flatten().reshape(x2_plot.size,1)
))

y_pred  = classification_traditional.predict(X_plot)
Y_discriminante = np.argmax(y_pred,axis=1)
y_plot = Y_discriminante.reshape(x1_plot.shape)

plt.contourf(x1_plot, x2_plot, y_plot, alpha=0.4, cmap='Set3')
plt.grid(True)
plt.xlim(-100,3800)
plt.ylim(-100,4100)
plt.legend()
plt.show()

"""
Executa a validação dos modelos via Monte Carlo (R=500).

Para cada uma das 500 rodadas:
1. Realiza o split aleatório dos dados (80/20).
2. Treina e testa os 5 modelos implementados.
3. Armazena a acurácia obtida para posterior análise estatística.
"""

models = {
    "MQO": ClassificationLinear,
}

validator = RandomSubsamplingValidation(X, Y, models, R=500)
results = validator.run()

print("\n===== RESULTADOS (MSE) =====\n")
print(f"{'Modelo':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-"*60)

for model_name in results:
    mse_values = results[model_name]["mse"]
    mean = np.mean(mse_values)
    std = np.std(mse_values)
    min_val = np.min(mse_values)
    max_val = np.max(mse_values)
    print(f"{model_name:<10} {mean:>12.2f} {std:>12.2f} {min_val:>12.2f} {max_val:>12.2f}")


print("\n===== RESULTADOS (R²) =====\n")
print(f"{'Modelo':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-"*60)

for model_name in results:
    r2_values = results[model_name]["r2"]
    mean = np.mean(r2_values)
    std = np.std(r2_values)
    min_val = np.min(r2_values)
    max_val = np.max(r2_values)
    
    print(f"{model_name:<10} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")

# W_hat = np.linalg.inv(X.T@X)@X.T@Y

# x1 = np.linspace(-200,8000,1000)

# x2 = -W_hat[0,0]/W_hat[2,0] - W_hat[1,0]/W_hat[2,0]*x1

# plt.plot(x1,x2,c='k')

# x2 = -W_hat[0,1]/W_hat[2,1] - W_hat[1,1]/W_hat[2,1]*x1
# plt.plot(x1,x2,c='r')

# x2 = -W_hat[0,2]/W_hat[2,2] - W_hat[1,2]/W_hat[2,2]*x1
# plt.plot(x1,x2,c='b')


# x_novo = np.array([1, 1544, 1425]).reshape(1,3)

# x1 = np.linspace(-200, 5000, 1000)
# X1, X2 = np.meshgrid(x1, x1)

# X_plot = np.hstack((
#     np.ones((X1.size,1)),
#     X1.flatten().reshape(X1.size,1),
#     X2.flatten().reshape(X1.size,1)
# ))

# y_pred  = x_novo @ W_hat
# Y_discriminante = np.argmax(y_pred,axis=1)
# print(nomes_classes[Y_discriminante[0]])

# y_plot = Y_discriminante.reshape(X1.shape)

# plt.contourf(X1, X2, y_plot, alpha=0.2, cmap='Set3')

# plt.scatter(x_novo[0,1], x_novo[0,2], marker='*')
# bp = 1
# plt.xlim(-50,3800)
# plt.ylim(-50,4100)
# plt.legend()
# plt.show()