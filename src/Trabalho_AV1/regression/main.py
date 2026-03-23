import os
import numpy as np
import matplotlib.pyplot as plt
from libs.models.LinearRegression import LinearRegression
from libs.models.MeanModel import MeanModel
from libs.validation.RandomSubsampling import RandomSubsamplingValidation

caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "aerogerador.dat")
data = np.loadtxt(caminho_arquivo, delimiter='\t')

x = data[:,0]
y = data[:,1]

plt.figure(figsize=(10,8))
plt.subplot(2,2,(1,2))
plt.scatter(x, y, edgecolors='k')
plt.title("Gráfico de dispersão", fontsize=18)
plt.xlabel("Velocidade do vento", fontsize=18)
plt.ylabel("Potência gerada", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)


plt.subplot(2,2,3)
plt.hist(x, bins=30, edgecolor='black')
plt.title("Histograma - Velocidade do vento", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.subplot(2,2,4)
plt.hist(y, bins=30, edgecolor='black')
plt.title("Histograma - Potência gerada")

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

X_data = x.reshape(-1,1)
Y_data = y.reshape(-1,1)

models = {
    "MQO": LinearRegression,
    "Mean": MeanModel
}

validator = RandomSubsamplingValidation(X_data, Y_data, models, R=500)
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

y_pred_mqo = LinearRegression(X_data, Y_data)
y_pred_mean = MeanModel(X_data, Y_data)
y_pred_mqo.fit()
y_pred_mean.fit()


plt.figure(figsize=(10,8))
plt.subplot(2,1,1)

plt.scatter(x, y, color='gray', edgecolors='black', label="Dados")
plt.plot(x, y_pred_mqo.predict(X_data),linewidth=2.5,label="MQO",color='#1f77b4')
plt.plot(x, y_pred_mean.predict(X_data), linewidth=2.5, linestyle="--", label="Média",color='#ff7f0e')
plt.xlabel("Velocidade do vento", fontsize=18)
plt.ylabel("Potência gerada", fontsize=18)
plt.title("Ajuste dos modelos", fontsize=18)
plt.legend()

plt.subplot(2,2,3)
plt.boxplot([
    results["MQO"]["mse"],
    results["Mean"]["mse"]
],
labels=["MQO", "Média"])
plt.yscale("log") 
plt.title("MSE", fontsize=18)
plt.ylabel("Erro", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.subplot(2,2,4)
plt.boxplot([
    results["MQO"]["r2"],
    results["Mean"]["r2"]
],
labels=["MQO", "Média"])
plt.title("R²")
plt.ylabel("Coeficiente")
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()

"""
Modelo             Mean          Std          Min          Max
------------------------------------------------------------
MQO              798.77       163.41       443.79      1313.17
Mean           11137.68       608.45      9313.19     13172.48

Modelo             Mean          Std          Min          Max
------------------------------------------------------------
MQO              0.9282       0.0134       0.8842       0.9588
Mean            -0.0026       0.0035      -0.0333      -0.0000
"""