import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from libs.models import ADALINE as a, Perceptron as per, MLP as mlp
from libs.validation.RandomSubsampling import RandomSubsamplingValidation


def plotar_curva_aprendizado(historico_erros, titulo, ax):
    ax.plot(range(1, len(historico_erros) + 1), historico_erros,
            marker='o', color='#b20000', markersize=3)
    ax.set_title(titulo)
    ax.set_xlabel('Épocas')
    ax.set_ylabel('Erro / MSE')
    ax.grid(True, linestyle='--', alpha=0.7)


def plotar_matriz_confusao_multiclasse(y_teste, y_pred, titulo, ax, nomes=None):
    # y_teste vem como one-hot array, y_pred já vem como o ID da classe (argmax)
    y_t_idx = np.argmax(y_teste, axis=1) if y_teste.ndim > 1 else y_teste
    y_p_idx = y_pred

    C = len(nomes) if nomes else len(np.unique(y_t_idx))
    matriz = np.zeros((C, C), dtype=int)
    
    for real, pred in zip(y_t_idx, y_p_idx):
        matriz[real, pred] += 1

    tick_labels = nomes if nomes else [str(i) for i in range(C)]
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels,
                ax=ax, cbar=False, linewidths=0.3)
    
    ax.set_title(titulo, fontsize=10)
    ax.set_ylabel('Real')
    ax.set_xlabel('Previsto')
    ax.tick_params(axis='x', rotation=45, labelsize=6)
    ax.tick_params(axis='y', rotation=0,  labelsize=6)


def plotar_boxplot_acuracia(resultados_modelos):
    plt.figure(figsize=(8, 6))
    dados = []
    nomes = []
    for nome, dados_modelo in resultados_modelos.items():
        dados.append(dados_modelo["metricas"]["acuracia"])
        nomes.append(nome)
    
    sns.boxplot(data=dados, palette="Set2")
    plt.xticks(ticks=range(len(nomes)), labels=nomes)
    plt.title("Distribuição de Acurácia nas Rodadas – Reconhecimento Facial")
    plt.ylabel("Acurácia")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plotar_violin_acuracia(resultados_modelos):
    plt.figure(figsize=(8, 6))
    dados = []
    nomes = []
    for nome, dados_modelo in resultados_modelos.items():
        dados.append(dados_modelo["metricas"]["acuracia"])
        nomes.append(nome)
        
    sns.violinplot(data=dados, palette="Set2", inner="box")
    plt.xticks(ticks=range(len(nomes)), labels=nomes)
    plt.title("Violin Plot de Acurácia – Reconhecimento Facial")
    plt.ylabel("Acurácia")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def imprimir_tabela_resultados(comparativo):
    print("\n" + "=" * 72)
    print(f"{'Modelo':<30} {'Média':>8} {'Desvio':>8} {'Máx':>8} {'Mín':>8}")
    print("=" * 72)
    for nome, stats in comparativo.items():
        acc = stats["acuracia"]
        print(f"{nome:<30} {acc['media']:>8.4f} {acc['desvio']:>8.4f} "
              f"{stats['max']:>8.4f} {stats['min']:>8.4f}")
    print("=" * 72)


if __name__ == '__main__':

    caminho_pasta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "RecFac")

    linhas_img, colunas_img = 30, 30
    p = linhas_img * colunas_img
    num_classes = 20
    imagens_por_classe = 32
    num_amostras = num_classes * imagens_por_classe 

    X_all = np.zeros((num_amostras, p))
    Y_all = -np.ones((num_amostras, num_classes))

    print("Iniciando o carregamento imperativo das imagens...")

    pastas_pessoas = sorted(os.listdir(caminho_pasta))
    idx_amostra = 0
    nomes_classes = []

    for classe_idx in range(num_classes):
        nome_pasta = pastas_pessoas[classe_idx]
        nomes_classes.append(nome_pasta)
        caminho_pessoa = os.path.join(caminho_pasta, nome_pasta)
        arquivos_img = sorted(os.listdir(caminho_pessoa))

        for img_idx in range(imagens_por_classe):
            nome_img = arquivos_img[img_idx]
            caminho_img = os.path.join(caminho_pessoa, nome_img)

            img_cinza = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
            img_redimensionada = cv2.resize(img_cinza, (colunas_img, linhas_img))
            vetor_atributos = img_redimensionada.flatten() / 255.0

            X_all[idx_amostra, :] = vetor_atributos
            Y_all[idx_amostra, classe_idx] = 1
            idx_amostra += 1

    print(f"Carregamento concluído!")
    print(f"Dimensão de X_all: {X_all.shape}  (Esperado: {num_amostras}, {p})")
    print(f"Dimensão de Y_all: {Y_all.shape}  (Esperado: {num_amostras}, {num_classes})")
    print("-" * 50)
    
    models = {
        "Perceptron": per.Perceptron,
        "Adaline":    a.ADALINE,
        "MLP":        mlp.MultilayerPerceptron,
    }

    validator = RandomSubsamplingValidation(
        X=X_all,
        y=Y_all,
        models=models,
        learning_rate=0.0001,
        R=10
    )

    print("\nIniciando validação Monte Carlo com R=10 rodadas...\n")
    results = validator.run(epochs=10000, max_workers=10)  

    comparativo_final = {}

    for nome_modelo, dados_modelo in results.items():
        print(f"\n{'─'*60}")
        print(f" Modelo: {nome_modelo}")
        print(f"{'─'*60}")

        # Extração das métricas
        m = dados_modelo["metricas"]
        acc_test = m["acuracia"]
        acc_train = m["acuracia_treino"]

        idx_melhor = int(np.argmax(acc_test))
        idx_pior   = int(np.argmin(acc_test))
        
        comparativo_final[nome_modelo] = {
            "acuracia":   {"media": np.mean(acc_test),  "desvio": np.std(acc_test)},
            "acc_treino": {"media": np.mean(acc_train), "desvio": np.std(acc_train)},
            "max": acc_test[idx_melhor],
            "min": acc_test[idx_pior]
        }

        print(f"[TESTE ] Acc: {np.mean(acc_test):.4f}")
        print(f"[TREINO] Acc: {np.mean(acc_train):.4f}")
        print(f"Melhor rodada: {idx_melhor} ({acc_test[idx_melhor]:.4f}) | Pior rodada: {idx_pior} ({acc_test[idx_pior]:.4f})")

        y_teste_melhor   = dados_modelo["artefatos"]["y_testes"][idx_melhor]
        y_pred_melhor    = dados_modelo["artefatos"]["y_preds"][idx_melhor]
        historico_melhor = dados_modelo["artefatos"]["historicos"][idx_melhor]

        y_teste_pior     = dados_modelo["artefatos"]["y_testes"][idx_pior]
        y_pred_pior      = dados_modelo["artefatos"]["y_preds"][idx_pior]
        historico_pior   = dados_modelo["artefatos"]["historicos"][idx_pior]

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 11))
        fig.suptitle(f"Reconhecimento Facial - Modelo: {nome_modelo}", fontsize=14)

        plotar_matriz_confusao_multiclasse(
            y_teste_melhor, y_pred_melhor,
            f"Matriz de Confusão - Melhor Caso (Acc={acc_test[idx_melhor]:.4f})",
            ax=axes[0, 0], nomes=nomes_classes
        )
        
        plotar_curva_aprendizado(
            historico_melhor,
            "Curva de Aprendizado - Melhor Caso",
            ax=axes[0, 1]
        )
        
        plotar_matriz_confusao_multiclasse(
            y_teste_pior, y_pred_pior,
            f"Matriz de Confusão - Pior Caso (Acc={acc_test[idx_pior]:.4f})",
            ax=axes[1, 0], nomes=nomes_classes
        )
        
        plotar_curva_aprendizado(
            historico_pior,
            "Curva de Aprendizado - Pior Caso",
            ax=axes[1, 1]
        )

        plt.tight_layout()
        plt.show()

    imprimir_tabela_resultados(comparativo_final)

    print("\nGerando Gráficos Comparativos Finais...")
    plotar_boxplot_acuracia(results)
    plotar_violin_acuracia(results)