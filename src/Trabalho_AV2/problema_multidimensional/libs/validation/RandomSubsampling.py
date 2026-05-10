import numpy as np
import concurrent.futures
import time

class RandomSubsamplingValidation:
    def __init__(self, X, y, models, R=10, train_size=0.8, learning_rate=0.001, precision=1e-6, mlp_topologies=None):
        self.X = X
        self.y = y
        self.models = models
        self.R = R
        self.train_size = train_size
        self.learning_rate = learning_rate
        self.precision = precision
        self.results = {}
        self.mlp_topologies = mlp_topologies if mlp_topologies else {"MLP": [64, 32]}

    def _treinar_uma_rodada(self, rodada, epochs, random_seed):
        np.random.seed(random_seed)
        N = self.X.shape[0]
        split_idx = int(N * self.train_size)
        idx = np.random.permutation(N)
        
        X_tr, y_tr = self.X[idx[:split_idx]], self.y[idx[:split_idx]]
        X_te, y_te = self.X[idx[split_idx:]], self.y[idx[split_idx:]]

        resultados_rodada = {}

        for name, model_class in self.models.items():
            if name.upper() == "MLP":
                for topo_name, topology in self.mlp_topologies.items():
                    model = model_class(
                        topology=topology.copy(), X_train=X_tr.T, Y_train=y_tr.T,
                        learning_rate=self.learning_rate, max_epochs=epochs, precision=self.precision
                    )
                    self._processar(model, topo_name, X_tr, y_tr, X_te, y_te, resultados_rodada)
            else:
                model = model_class(X_tr.T, y_tr.T, self.learning_rate)
                self._processar(model, name, X_tr, y_tr, X_te, y_te, resultados_rodada, epochs)

        return rodada, resultados_rodada

    def _processar(self, model, label, X_tr, y_tr, X_te, y_te, container, epochs=None):
        hist = model.fit(epochs) if epochs else model.fit()
        
        y_p_raw_te = model.predict(X_te.T)
        y_p_raw_tr = model.predict(X_tr.T)

        y_p_te = y_p_raw_te.T if y_p_raw_te.shape[0] == y_te.shape[1] else y_p_raw_te
        y_p_tr = y_p_raw_tr.T if y_p_raw_tr.shape[0] == y_tr.shape[1] else y_p_raw_tr

        acc_te = np.mean(np.argmax(y_te, axis=1) == np.argmax(y_p_te, axis=1))
        acc_tr = np.mean(np.argmax(y_tr, axis=1) == np.argmax(y_p_tr, axis=1))

        container[label] = {
            "acc_teste": acc_te, 
            "acc_treino": acc_tr,
            "y_teste": y_te, 
            "y_pred": np.argmax(y_p_te, axis=1), # Guardamos apenas o ID da pessoa
            "historico": hist
        }

    def run(self, epochs, max_workers=4):
        self.results = {}
        inicio = time.perf_counter()
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futuros = [executor.submit(self._treinar_uma_rodada, r, epochs, np.random.randint(0, 10**6)) for r in range(self.R)]
            
            for fut in concurrent.futures.as_completed(futuros):
                r_idx, res = fut.result()
                print(f"Rodada {r_idx+1}/{self.R} concluída.")
                
                for mod, dados in res.items():
                    if mod not in self.results:
                        self.results[mod] = {
                            "metricas": {"acuracia": [], "acuracia_treino": []},
                            "artefatos": {"y_testes": [], "y_preds": [], "historicos": []}
                        }
                    self.results[mod]["metricas"]["acuracia"].append(dados["acc_teste"])
                    self.results[mod]["metricas"]["acuracia_treino"].append(dados["acc_treino"])
                    self.results[mod]["artefatos"]["y_testes"].append(dados["y_teste"])
                    self.results[mod]["artefatos"]["y_preds"].append(dados["y_pred"])
                    self.results[mod]["artefatos"]["historicos"].append(dados["historico"])
        
        print(f"✅ Finalizado em {time.perf_counter() - inicio:.2f}s")
        return self.results