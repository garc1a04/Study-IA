import numpy as np
import concurrent.futures
import time

class RandomSubsamplingValidation:
    def __init__(self, X, y, models, R=500, train_size=0.8, learning_rate=0.5):
        self.X = X
        self.y = y
        self.models = models
        self.R = R
        self.train_size = train_size
        self.results = {}
        self.learning_rate = learning_rate
        self.mlp_topologies = {
            # "MLP": [16, 8],
            "MLP_Underfitting": [2],
            "MLP_Overfitting":  [64, 64, 32],   
        }

    def _treinar_uma_rodada(self, rodada, epochs, random_seed):
        np.random.seed(random_seed)
        
        N = self.X.shape[0]
        split_idx = int(N * self.train_size)
        
        idx = np.random.permutation(N)
        X_embaralhado = self.X[idx]
        y_embaralhado = self.y[idx]

        X_treino = X_embaralhado[:split_idx]
        y_treino = y_embaralhado[:split_idx]
        X_teste = X_embaralhado[split_idx:]
        y_teste = y_embaralhado[split_idx:]

        X_treino_ajustado = X_treino.T
        y_treino_ajustado = y_treino.reshape(1, -1)

        resultados_desta_rodada = {}

        for name, model_class in self.models.items():
            if name.upper() == "MLP":
                for topo_name, topology in self.mlp_topologies.items():
                    model = model_class(
                        topology=topology.copy(),
                        X_train=X_treino_ajustado,
                        Y_train=y_treino_ajustado,
                        learning_rate=self.learning_rate,
                        max_epochs=epochs,
                        precision=1e-6
                    )
                    
                    historico = model.fit() 
                    y_pred_raw = model.predict(X_teste.T)
                    y_pred = np.where(y_pred_raw >= 0, 1, -1).flatten()
                    acc_teste, sens, esp, prec, f1 = self._compute_metrics(y_teste.flatten(), y_pred)

                    y_pred_tr_raw = model.predict(X_treino.T)
                    y_pred_tr = np.where(y_pred_tr_raw >= 0, 1, -1).flatten()
                    acc_treino, *_ = self._compute_metrics(y_treino.flatten(), y_pred_tr)

                    resultados_desta_rodada[topo_name] = {
                        "acc_teste": acc_teste, "acc_treino": acc_treino,
                        "sens": sens, "esp": esp, "prec": prec, "f1": f1,
                        "y_teste": y_teste, "y_pred": y_pred, "historico": historico,
                    }
            else:
                model = model_class(X_treino_ajustado, y_treino_ajustado, self.learning_rate)
                historico = model.fit(epochs)
                
                y_pred_raw = model.predict(X_teste.T)
                y_pred = np.where(y_pred_raw >= 0, 1, -1).flatten()
                acc_teste, sens, esp, prec, f1 = self._compute_metrics(y_teste.flatten(), y_pred)
                
                y_pred_tr_raw = model.predict(X_treino.T)
                y_pred_tr = np.where(y_pred_tr_raw >= 0, 1, -1).flatten()
                acc_treino, *_ = self._compute_metrics(y_treino.flatten(), y_pred_tr)

                resultados_desta_rodada[name] = {
                    "acc_teste": acc_teste, "acc_treino": acc_treino,
                    "sens": sens, "esp": esp, "prec": prec, "f1": f1,
                    "y_teste": y_teste, "y_pred": y_pred, "historico": historico
                }

        return rodada, resultados_desta_rodada

    def run(self, epochs, max_workers=None):
        for name in self.models:
            if name.upper() != "MLP":
                self.results[name] = {
                    "metricas": {"acuracia": [], "acuracia_treino": [], "sensibilidade": [], "especificidade": [], "precisao": [], "f1_score": []},
                    "artefatos": {"y_testes": [], "y_preds": [], "historicos": []}
                }

        for topo_name in self.mlp_topologies:
            self.results[topo_name] = {
                "metricas": {"acuracia": [], "acuracia_treino": [], "sensibilidade": [], "especificidade": [], "precisao": [], "f1_score": []},
                "artefatos": {"y_testes": [], "y_preds": [], "historicos": []}
            }

        inicio = time.perf_counter()
        print(f"\nIniciando treinamento PARALELO com {self.R} rodadas...")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futuros = []

            for rodada in range(self.R):
                seed = np.random.randint(0, 2**31 - 1)
                futuro = executor.submit(self._treinar_uma_rodada, rodada, epochs, seed)
                futuros.append(futuro)

            for futuro in concurrent.futures.as_completed(futuros):
                rodada_concluida, resultados_rodada = futuro.result()
                print(f"Rodada concluída: {rodada_concluida + 1}/{self.R}")
                

                for model_key, dados in resultados_rodada.items():
                    r = self.results[model_key]
                    r["metricas"]["acuracia"].append(dados["acc_teste"])
                    r["metricas"]["acuracia_treino"].append(dados["acc_treino"])
                    r["metricas"]["sensibilidade"].append(dados["sens"])
                    r["metricas"]["especificidade"].append(dados["esp"])
                    r["metricas"]["precisao"].append(dados["prec"])
                    r["metricas"]["f1_score"].append(dados["f1"])
                    r["artefatos"]["y_testes"].append(dados["y_teste"])
                    r["artefatos"]["y_preds"].append(dados["y_pred"])
                    r["artefatos"]["historicos"].append(dados["historico"])

        fim = time.perf_counter()
        print(f"\n✅ O treinamento paralelo demorou {fim - inicio:.4f} segundos.")
        return self.results
    
    def _compute_metrics(self, y_t, y_p):
        TP = np.sum((y_p == 1)  & (y_t == 1))
        TN = np.sum((y_p == -1) & (y_t == -1))
        FP = np.sum((y_p == 1)  & (y_t == -1))
        FN = np.sum((y_p == -1) & (y_t == 1))

        acc          = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
        sensibilidade = TP / (TP + FN)           if (TP + FN) > 0 else 0.0
        especificidade = TN / (TN + FP)          if (TN + FP) > 0 else 0.0
        precisao      = TP / (TP + FP)           if (TP + FP) > 0 else 0.0
        f1_score      = (2 * precisao * sensibilidade / (precisao + sensibilidade)
                         if (precisao + sensibilidade) > 0 else 0.0)
        return acc, sensibilidade, especificidade, precisao, f1_score