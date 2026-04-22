import os
import numpy as np
import matplotlib.pyplot as plt
from libs.models.LinearRegression import LinearRegression
from libs.models.LinearRegression import LinearRegressionRegular
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
    "tiknov λ = 0": LinearRegressionRegular,
    "tiknov λ = 0.25": LinearRegressionRegular,
    "tiknov λ = 0.5": LinearRegressionRegular,
    "tiknov λ = 0.75": LinearRegressionRegular,
    "tiknov λ = 1": LinearRegressionRegular,
    "Mean": MeanModel
}

validator = RandomSubsamplingValidation(X_data, Y_data, models, R=500)
results = validator.run()

print("\n===== RESULTADOS (MSE) =====\n")
print(f"{'Modelo':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-"*60)

for model_name in results:
    mse_values = results[model_name]["mse"]
    mean = np.mean(mse_values)
    std = np.std(mse_values)
    min_val = np.min(mse_values)
    max_val = np.max(mse_values)
    
    print(f"{model_name:<15} {mean:>12.2f} {std:>12.2f} {min_val:>12.2f} {max_val:>12.2f}")

print("\n===== RESULTADOS (R²) =====\n")
print(f"{'Modelo':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("-"*60)

for model_name in results:
    r2_values = results[model_name]["r2"]
    mean = np.mean(r2_values)
    std = np.std(r2_values)
    min_val = np.min(r2_values)
    max_val = np.max(r2_values)
    
    print(f"{model_name:<15} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f}")

models_plot = {
    "MQO": LinearRegression(X_data, Y_data),
    "tiknov λ = 0": LinearRegressionRegular(X_data, Y_data),       
    "tiknov λ = 0.25": LinearRegressionRegular(X_data, Y_data),
    "tiknov λ = 0.5": LinearRegressionRegular(X_data, Y_data),  
    "tiknov λ = 0.75": LinearRegressionRegular(X_data, Y_data),
    "tiknov λ = 1": LinearRegressionRegular(X_data, Y_data),
    "Mean": MeanModel(X_data, Y_data)
}

for name, model in models_plot.items():
    if name.__contains__("tiknov"):
        lamb = float(str.split(name, "tiknov λ =")[1])
        model.fit(lamb)
        continue

    model.fit()

plt.figure(figsize=(14, 10)) 
plt.subplot(2, 1, 1)
plt.scatter(x, y, color='gray', edgecolors='black', label="Dados")
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

for (nome, modelo), cor in zip(models_plot.items(), cores):
    estilo_linha = "--" if nome == "Mean" else "-"
    plt.plot(x, modelo.predict(X_data), linewidth=2.5, linestyle=estilo_linha, label=nome, color=cor)

plt.xlabel("Velocidade do vento", fontsize=18)
plt.ylabel("Potência gerada", fontsize=18)
plt.title("Ajuste dos modelos", fontsize=18)
plt.legend(ncol=2, fontsize=12) 

nomes_modelos = list(results.keys())
mse_data = [results[nome]["mse"] for nome in nomes_modelos]
r2_data = [results[nome]["r2"] for nome in nomes_modelos]

# --- SUBPLOT 3: Boxplot do MSE ---
plt.subplot(2, 2, 3)
plt.boxplot(mse_data, labels=nomes_modelos)
plt.yscale("log") 
plt.title("MSE", fontsize=18)
plt.ylabel("Erro", fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=12) 
plt.yticks(fontsize=14)

# --- SUBPLOT 4: Boxplot do R² ---
plt.subplot(2, 2, 4)
plt.boxplot(r2_data, labels=nomes_modelos)
plt.title("R²", fontsize=18)
plt.ylabel("Coeficiente", fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=12) 

plt.tight_layout() 
plt.show()