import os
import numpy as np
import matplotlib.pyplot as plt
from libs.models.ClassificationLinear import ClassificationLinear
from libs.validation.RandomSubsampling import RandomSubsamplingValidation
from libs.validation.KFold import KFoldCrossValidationGaussiano
from libs.models.ClassificationBayes import ClassificadorBayesIngenuo, ClassificadorGaussianoCovarianciasIguais, ClassificadorGaussianoRegularizado, ClassificadorGaussianoTradicional, ClassificadorGaussianoMatrizAgregada

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

X_mqo = X
Y_mqo = Y

X_bayes = data[:, :-1]
Y_bayes = data[:, -1]

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
Para o classicador gaussiano regularizado, ha a depend^encia da denic~ao de seu hiperpar^ametro .
Solicita-se ent~ao que aplique a validac~ao cruzada chamada k􀀀fold cross validation, para identicar qual
e o valor de  ideal dentre uma sequ^encia de diferentes . Neste caso, os valores a serem testados s~ao
dados pela seguinte lista:
 = f0; 0:001; 0:01; 0:1; 0:2; 0:3; 0:4; 0:5; 0:6; 0:7; 0:8; 0:9; 1g
Para compor o resultado do valor de  ideal analise na validac~ao cruzada a acuracia do modelo regularizado
"""

y_ajustado = Y_bayes.reshape(-1, 1)
valores_lambda = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
buscador = KFoldCrossValidationGaussiano(X=X_bayes, y=y_ajustado, lambdas=valores_lambda, k=5)
melhor_lbd, acuracia_max, historico = buscador.run()

print("-" * 50)
print(f"RESULTADO FINAL:")
print(f"O MELHOR valor de lambda (alpha) é: {melhor_lbd}")
print(f"Acurácia média obtida com ele: {acuracia_max:.4f}")
print("-" * 50)

"""
Executa a validação dos modelos via Monte Carlo (R=500).

Para cada uma das 500 rodadas:
1. Realiza o split aleatório dos dados (80/20).
2. Treina e testa os 5 modelos implementados.
3. Armazena a acurácia obtida para posterior análise estatística.
"""

models = {
    "MQO": ClassificationLinear,
    "Gaussiano Tradicional (QDA)": ClassificadorGaussianoTradicional,
    "Covariâncias Iguais (LDA)": ClassificadorGaussianoCovarianciasIguais,
    "Matriz Agregada": ClassificadorGaussianoMatrizAgregada,
    "Bayes Ingênuo (Naive Bayes)": ClassificadorBayesIngenuo,
    "Gaussiano Regularizado (RDA)": ClassificadorGaussianoRegularizado
}

results = {}

for nome_modelo, model in models.items():
    if nome_modelo == "MQO":
        validator = RandomSubsamplingValidation(X=X_mqo, y=Y_mqo, models={nome_modelo: model})
    else:
        validator = RandomSubsamplingValidation(X=X_bayes, y=y_ajustado, models={nome_modelo: model}, alpha=melhor_lbd)
    results[nome_modelo] = validator.run()

print("\n===== RESULTADOS (ACURÁCIA) =====\n")
print(f"{'Modelo':<35} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-" * 85)

for model_name in results:
    acc_values = results[model_name][model_name]["acuracia"] 
    mean = np.mean(acc_values)
    std = np.std(acc_values)
    min_val = np.min(acc_values)
    max_val = np.max(acc_values)
    print(f"{model_name:<35} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")


x1 = np.linspace(-100, 3800, 300)
x2 = np.linspace(-100, 4100, 300)
x1_plot, x2_plot = np.meshgrid(x1, x2)
X_plot = np.c_[x1_plot.ravel(), x2_plot.ravel()]

modelos = {
    "Gaussiano Tradicional (QDA)": ClassificadorGaussianoTradicional(X_bayes, Y_bayes),
    "Covariâncias Iguais (LDA)": ClassificadorGaussianoCovarianciasIguais(X_bayes, Y_bayes),
    "Matriz Agregada": ClassificadorGaussianoMatrizAgregada(X_bayes, Y_bayes),
    "Gaussiano Regularizado (RDA)": ClassificadorGaussianoRegularizado(X_bayes, Y_bayes, alpha=melhor_lbd),
    "Bayes Ingênuo (Naive Bayes)": ClassificadorBayesIngenuo(X_bayes, Y_bayes),
    "MQO": ClassificationLinear(X_mqo, Y_mqo) 
}

classes = np.unique(Y_bayes)
cores_plot = ['b', "#FF00BF", "#00FF40", '#5CFFFF', 'r'] 
nomes_plot = ['Neutro', 'Sorriso', 'Sobrancelha', 'Surpreso', 'Rabugento']

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, (nome_modelo, modelo) in enumerate(modelos.items()):
    modelo.fit()
    y_pred = modelo.predict(X_plot)
    if nome_modelo == "MQO":
        y_pred = np.argmax(y_pred, axis=1) + 1
        
    y_plot = y_pred.reshape(x1_plot.shape)
    ax = axes[i]
    ax.contourf(x1_plot, x2_plot, y_plot, alpha=0.4, cmap='Set3')
    ax.contour(x1_plot, x2_plot, y_plot, colors='k', linewidths=0.5, alpha=0.5)

    for j, classe in enumerate(classes):
        X_classe = X_bayes[Y_bayes == classe]
        ax.scatter(X_classe[:, 0], X_classe[:, 1], c=cores_plot[j], 
                   label=nomes_plot[j], edgecolors='k', s=40)
        
    ax.set_title(f"{nome_modelo}", fontsize=12, fontweight='bold')
    ax.set_xlim(-100, 3800)
    ax.set_ylim(-100, 4100)
    ax.grid(True, linestyle='--', alpha=0.6)
    if i == 0:
        ax.legend(loc='best', fontsize='small')

fig.suptitle('Fronteiras de Decisão dos Modelos Classificadores', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

labels = []
data_boxplot = []

for model_name in results:
    acc_values = results[model_name][model_name]["acuracia"]
    labels.append(model_name)
    data_boxplot.append(acc_values)

plt.figure(figsize=(12, 6))

box = plt.boxplot(
    data_boxplot,
    labels=labels,
    patch_artist=True,
    showmeans=True,
    meanline=True
)

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel("Acurácia")
plt.title("Distribuição das Acurácias dos Modelos (Monte Carlo - 500 Rodadas)")
plt.ylim(0.70, 1.00)
plt.xticks(rotation=20)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

bp = 1