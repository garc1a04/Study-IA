import os
import numpy as np
import matplotlib.pyplot as plt
from libs.LinearRegression import LinearRegression
from libs.MeanModel import MeanModel

"""
1. Faca uma visualização inicial dos dados através do grafico de espalhamento. Nessa etapa, faca discussões sobre quais serão as caractersticas de um modelo que consegue entender o padr~ao entre variaveis regressoras e variaveis observadas.
"""

# Carregando o arquivo
caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "aerogerador.dat")
data = np.loadtxt(caminho_arquivo, delimiter='\t')
print(data.shape)

# plotando os dados

x = data[:,0]
y = data[:,1]

plt.scatter(x, y)
plt.title("Gráfico de dispersão")
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.show()

"""
2. Em seguida, organize os dados de modo que as variáveis regressoras sejam armazenadas em uma matriz (X) de dimensão R^(Nxp). Faça o mesmo para o vetor da variável dependente (y), organizando-o em um vetor de dimensão R^(Nx1).

3. Os modelos a serem implementados nessa etapa serão: 

- MQO tradicional: feito

- MQO regularizado (Tikhonov): cilindro não passou :)

- média da variável dependente: feito

Observação: lembre-se que todos os modelos estimam o valor do intercepto.


(pulado)
4. Para o modelo regularizado, necessita-se da definição de seu hiperparâmetro lambda. Assim, sua equipe deve testar o presente modelo para os seguintes valores de lambda:
lambda = {0, 0.25, 0.5, 0.75, 1}
Assim, ao todo, existirão 6 estimativas diferentes do vetor beta pertencente a R^((p+1) x 1).
"""

X_data = x.reshape(-1,1)
Y_data = y.reshape(-1,1)

"""
5. Para validar os modelos utilizados na tarefa de regressão, sua equipe deve projetar a validação chamada: 

- Random Subsampling Validation.

Essa necessita da definição da quantidade de rodadas da simulação (R = 500). Em cada rodada, deve-se realizar o particionamento em 80% dos dados para treinamento e 20% para teste. As medidas de desempenho de cada um dos 5 modelos diferentes devem ser: 

- MSE
- R²

compondo assim duas listas de 500 valores para cada modelo avaliado.
"""

MQO_vet_mse = []
Mean_vet_mse = []

MQO_vet_r2 = []
Mean_vet_r2 = []

N = X_data.shape[0]
rodadas = 500
for r in range(rodadas):

    idx = np.random.permutation(N)
    X_embaralhado = np.copy(X_data)[idx, :]
    y_embaralhado = np.copy(Y_data)[idx, :]

    #Partição (80/20)
    X_treino = X_embaralhado[:int(N*.8), :]
    y_treino = y_embaralhado[:int(N*.8), :]
    
    X_teste = X_embaralhado[int(N*.8):, :]
    y_teste = y_embaralhado[int(N*.8):, :]
    
    model = LinearRegression(X_treino, y_treino)
    model.fit()
    y_pred = model.predict(X_teste)
    
    # plt.scatter(X_teste, y_teste)   
    # plt.plot(X_teste,y_pred)
    # plt.title("Gráfico de dispersão de test (MQO tradicional)")
    # plt.xlabel("Velocidade do vento")
    # plt.ylabel("Potência gerada")
    # plt.show()
    
    model_mean = MeanModel(y_treino)
    model_mean.fit()
    y_pred_mean = model_mean.predict(X_teste)
    
    MSE_MQO = np.mean((y_pred - y_teste)**2)
    MSE_MEAN = np.mean((y_pred_mean - y_teste)**2)
    MQO_vet_mse.append(MSE_MQO)
    Mean_vet_mse.append(MSE_MEAN)
    
    SS_res_MQO = np.sum((y_teste - y_pred)**2)
    SS_res_MEAN = np.sum((y_teste - y_pred_mean)**2)
    SS_tot = np.sum((y_teste - np.mean(y_teste))**2)
    R2_MQO = 1 - (SS_res_MQO / SS_tot)
    R2_MEAN = 1 - (SS_res_MEAN / SS_tot)
    MQO_vet_r2.append(R2_MQO)
    Mean_vet_r2.append(R2_MEAN)


"""
6. Ao final das R rodadas, calcule para cada modelo utilizado, bem como para cada métrica de desempenho, a média aritmética, o desvio-padrão, o maior valor e o menor valor. Coloque os resultados obtidos em uma tabela e discuta os resultados obtidos. 
"""

mean_MSE_MQO = np.mean(MQO_vet_mse)
mean_MSE_Mean = np.mean(Mean_vet_mse)

std_MSE_MQO = np.std(MQO_vet_mse)
std_MSE_Mean = np.std(Mean_vet_mse)

max_MSE_MQO = np.max(MQO_vet_mse)
max_MSE_Mean = np.max(Mean_vet_mse)

min_MSE_MQO = np.min(MQO_vet_mse)
min_MSE_Mean = np.min(Mean_vet_mse)

print("\n===== RESULTADOS (MSE) =====\n")

print(f"{'Modelo':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-"*60)

print(f"{'MQO':<10} {mean_MSE_MQO:>12.2f} {std_MSE_MQO:>12.2f} {min_MSE_MQO:>12.2f} {max_MSE_MQO:>12.2f}")
print(f"{'Mean':<10} {mean_MSE_Mean:>12.2f} {std_MSE_Mean:>12.2f} {min_MSE_Mean:>12.2f} {max_MSE_Mean:>12.2f}")
    
mean_R2_MQO = np.mean(MQO_vet_r2)
mean_R2_Mean = np.mean(Mean_vet_r2)

std_R2_MQO = np.std(MQO_vet_r2)
std_R2_Mean = np.std(Mean_vet_r2)

max_R2_MQO = np.max(MQO_vet_r2)
max_R2_Mean = np.max(Mean_vet_r2)

min_R2_MQO = np.min(MQO_vet_r2)
min_R2_Mean = np.min(Mean_vet_r2)


print("\n===== RESULTADOS (R²) =====\n")

print(f"{'Modelo':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-"*60)

print(f"{'MQO':<10} {mean_R2_MQO:>12.4f} {std_R2_MQO:>12.4f} {min_R2_MQO:>12.4f} {max_R2_MQO:>12.4f}")
print(f"{'Mean':<10} {mean_R2_Mean:>12.4f} {std_R2_Mean:>12.4f} {min_R2_Mean:>12.4f} {max_R2_Mean:>12.4f}")


bp = 1