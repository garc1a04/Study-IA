import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from libs.models import ADALINE as a, Perceptron as per, MLP as mlp
from libs.validation.RandomSubsampling import RandomSubsamplingValidation

caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spiral_d.csv")
data = np.loadtxt(caminho_arquivo, delimiter=',')

def plotar_curva_aprendizado(historico_erros, titulo, ax):
    ax.plot(range(1, len(historico_erros) + 1), historico_erros, marker='o', color='#b20000', markersize=4)
    ax.set_title(titulo)
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Erro / MSE')
    ax.grid(True, linestyle='--', alpha=0.7)

def plotar_curvas_aprendizado(erros_treino, erros_validacao, titulo, ax):
    epocas = range(1, len(erros_treino) + 1)
    ax.plot(epocas, erros_treino, label='Treinamento (MSE)', color='#0044cc', linewidth=2)
    ax.plot(epocas, erros_validacao, label='Validação (MSE)', color='#b20000', linestyle='--', linewidth=2)
    
    ax.set_title(titulo)
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Erro (MSE)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def plotar_matriz_confusao(y_teste, y_pred, titulo, ax):
    y_t = y_teste.flatten()
    y_p = y_pred.flatten()

    TP = np.sum((y_t == 1) & (y_p == 1))
    TN = np.sum((y_t == -1) & (y_p == -1))
    FP = np.sum((y_t == -1) & (y_p == 1))
    FN = np.sum((y_t == 1) & (y_p == -1))
    
    matriz = np.array([[TN, FP], 
                       [FN, TP]])
    
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto -1', 'Previsto +1'], 
                yticklabels=['Real -1', 'Real +1'], ax=ax)
    
    ax.set_title(titulo)
    ax.set_ylabel('Valores Reais')
    ax.set_xlabel('Valores Previstos')

def plotar_boxplot_metricas(resultados_modelos, nome_metrica="acuracia"):
    plt.figure(figsize=(8, 6))
    dados_para_plot = []
    nomes = []
    
    for nome_modelo, dados in resultados_modelos.items():
        lista_valores = dados["metricas"][nome_metrica]
        dados_para_plot.append(lista_valores)
        nomes.append(nome_modelo)
        
    sns.boxplot(data=dados_para_plot, palette="Set2")
    plt.xticks(ticks=range(len(nomes)), labels=nomes)
    plt.title(f'Distribuição de {nome_metrica.capitalize()} nas 500 Rodadas')
    plt.ylabel(nome_metrica.capitalize())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plotar_diagnostico_barras(labels, acc_treino, acc_teste):
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, acc_treino, width, label="Treino", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, acc_teste, width, label="Teste", color="#DD8452")

    ax.set(title="Diagnóstico de Underfitting / Overfitting – MLP",
           ylabel="Acurácia Média", xticks=x, xticklabels=labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for barras in [bars1, bars2]:
        for bar in barras:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


def plotar_boxplots_acuracia(labels, dados_treino, dados_teste):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição da Acurácia – Variações do MLP")

    configs = [
        (axes[0], dados_treino, "Treino"),
        (axes[1], dados_teste, "Teste")
    ]

    for ax, dados, titulo in configs:
        sns.boxplot(data=dados, palette=["#4C72B0", "#DD8452"], ax=ax)
        ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel="Acurácia", title=f"Acurácia no {titulo}")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    X_all = data[:, :-1]
    Y_all = data[:, -1]

    classes = [1, -1]
    cores = ["#FF00BF", "#00FF40"]

    plt.title("Gráfico de espalhamento do dataset")
    for i, classe in enumerate(classes):
        X_plot = data[data[:, -1] == classe, :-1]    
        plt.scatter(X_plot[:, 0], X_plot[:, 1], c=cores[i], label=f'CLASSE {classe}', edgecolors='k')

    plt.grid(True)
    plt.legend()
    plt.show()

    models = {
        # "Perceptron": per.Perceptron,
        # "Adaline": a.ADALINE,
        "MLP": mlp.MultilayerPerceptron
    }

    validator = RandomSubsamplingValidation(
        X=X_all, 
        y=Y_all, 
        models=models,
        learning_rate=0.001,
        R=500
    )

    results = validator.run(5000, max_workers=15)
    comparativo_final = {}

    METRICAS_AUXILIARES = {"acuracia_treino"}

    for nome_modelo, dados_modelo in results.items():
        print(f"\nExtraindo métricas e gráficos para o modelo: {nome_modelo}")
        comparativo_final[nome_modelo] = {}

        for nome_metrica, lista_valores in dados_modelo["metricas"].items():
            if nome_metrica in METRICAS_AUXILIARES:
                continue

            idx_melhor = int(np.argmax(lista_valores))
            idx_pior   = int(np.argmin(lista_valores))

            media        = np.mean(lista_valores)
            desvio       = np.std(lista_valores)
            melhor_valor = lista_valores[idx_melhor]
            pior_valor   = lista_valores[idx_pior]

            comparativo_final[nome_modelo][nome_metrica] = {
                "media":         media,
                "desvio_padrao": desvio,
                "maior_valor":   melhor_valor,
                "menor_valor":   pior_valor,
                "rodada_melhor": idx_melhor,
                "rodada_pior":   idx_pior
            }

            print(f"[{nome_metrica.upper()}] Média: {media:.4f} (+/- {desvio:.4f}) "
                f"| Máx: {melhor_valor:.4f} (Rod. {idx_melhor}) "
                f"| Min: {pior_valor:.4f} (Rod. {idx_pior})")

            y_teste_melhor  = dados_modelo["artefatos"]["y_testes"][idx_melhor]
            y_pred_melhor   = dados_modelo["artefatos"]["y_preds"][idx_melhor]
            historico_melhor = dados_modelo["artefatos"]["historicos"][idx_melhor]

            y_teste_pior    = dados_modelo["artefatos"]["y_testes"][idx_pior]
            y_pred_pior     = dados_modelo["artefatos"]["y_preds"][idx_pior]
            historico_pior  = dados_modelo["artefatos"]["historicos"][idx_pior]

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
            fig.suptitle(f"Análise de {nome_metrica.capitalize()} - Modelo: {nome_modelo}", fontsize=16)

            plotar_matriz_confusao(y_teste_melhor, y_pred_melhor, "Matriz: Melhor Caso", ax=axes[0,0])
            plotar_curva_aprendizado(historico_melhor, "Curva: Melhor Caso", ax=axes[0, 1])
            plotar_matriz_confusao(y_teste_pior,   y_pred_pior,   "Matriz: Pior Caso", ax=axes[1, 0])
            plotar_curva_aprendizado(historico_pior,   "Curva: Pior Caso", ax=axes[1, 1])
            plt.tight_layout()
            plt.show()


    print("\nGerando Boxplots comparativos finais...")
    plotar_boxplot_metricas(results, "acuracia")
    plotar_boxplot_metricas(results, "f1_score")

    TOPOLOGIAS_MLP = {
        "MLP_Underfitting": "Underfitting\n[2]",
        "MLP_Overfitting": "Overfitting\n[64, 64, 32]"
    }

    topologias_presentes = [t for t in TOPOLOGIAS_MLP if t in results]

    if topologias_presentes:
        print("\nGerando diagnóstico de Underfitting / Overfitting para o MLP...")
        
        labels = [TOPOLOGIAS_MLP[t] for t in topologias_presentes]
        
        dados_treino = [results[t]["metricas"]["acuracia_treino"] for t in topologias_presentes]
        dados_teste  = [results[t]["metricas"]["acuracia"] for t in topologias_presentes]
        
        medias_treino = [np.mean(dt) for dt in dados_treino]
        medias_teste  = [np.mean(dt) for dt in dados_teste]

        plotar_curvas_aprendizado(labels, medias_treino, medias_teste)
        plotar_boxplots_acuracia(labels, dados_treino, dados_teste)
        print(f"\n{'Topologia':<22} {'Treino (média)':<18} {'Teste (média)':<18} {'Gap (T-t)'}")
        print("-" * 70)
        
        for label, tr, te in zip(labels, medias_treino, medias_teste):
            label_limpo = label.replace('\n', ' ')
            print(f"{label_limpo:<22} {tr:<18.4f} {te:<18.4f} {tr - te:.4f}")