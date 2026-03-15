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

plt.scatter(x, y)
plt.title("Gráfico de dispersão")
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
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

bp = 1