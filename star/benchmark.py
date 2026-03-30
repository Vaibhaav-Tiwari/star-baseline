"""
star/benchmark.py
=================
StabilityBenchmark: multi-seed evaluation harness.

Works with both NumpyGraphAutoEncoder and PyTorch GraphAutoEncoder / StaRWrapper.
"""

import numpy as np
from typing import Callable, Dict, List
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import logging

from .data import SpatialDataset

log = logging.getLogger("StaR.benchmark")


class StabilityBenchmark:
    """
    Systematic multi-seed evaluation.

    n_seeds       : number of random seeds
    n_clusters    : k for k-means (set to ground-truth n_layers)
    kmeans_n_init : k-means restarts per seed
    """

    def __init__(self, n_seeds: int = 100,
                 n_clusters: int = 7,
                 kmeans_n_init: int = 10):
        self.n_seeds       = n_seeds
        self.n_clusters    = n_clusters
        self.kmeans_n_init = kmeans_n_init
        self._le           = LabelEncoder()

    def run(self, model_factory: Callable,
            dataset: SpatialDataset,
            label: str = "model",
            verbose: bool = True) -> Dict:
        """
        Run model_factory(seed) for every seed and collect ARI / NMI.

        model_factory : callable(seed:int) → model
            model must expose get_embedding(X, edge_index) → np.ndarray
            X and edge_index are passed as numpy arrays;
            the model is responsible for converting them to tensors if needed.
        """
        true         = self._le.fit_transform(dataset.labels)
        ari_list, nmi_list = [], []

        for seed in range(self.n_seeds):
            if verbose and seed % max(1, self.n_seeds // 10) == 0:
                log.info(f"    {label}: seed {seed}/{self.n_seeds}")

            np.random.seed(seed)
            model = model_factory(seed)

            # Always pass raw numpy arrays — get_embedding() handles conversion
            z = model.get_embedding(dataset.X, dataset.edge_index)

            km = KMeans(n_clusters=self.n_clusters,
                        random_state=seed,
                        n_init=self.kmeans_n_init).fit_predict(z)

            ari_list.append(adjusted_rand_score(true, km))
            nmi_list.append(normalized_mutual_info_score(true, km))

        ari = np.array(ari_list, np.float32)
        nmi = np.array(nmi_list, np.float32)
        q25, q75 = np.percentile(ari, [25, 75])

        return dict(
            label=label, ari_list=ari, nmi_list=nmi,
            mean_ari=float(ari.mean()), var_ari=float(ari.var()),
            std_ari=float(ari.std()),   iqr_ari=float(q75 - q25),
            p60=float((ari > 0.60).mean()),
            p50=float((ari > 0.50).mean()),
            min_ari=float(ari.min()), max_ari=float(ari.max()),
            mean_nmi=float(nmi.mean()),
        )

    def print_table(self, results: List[Dict]):
        hdr = (f"{'Model':<30} {'MeanARI':>9} {'VarARI':>9} "
               f"{'StdARI':>9} {'IQR':>8} {'P>0.60':>8}")
        print("\n" + "=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            print(f"{r['label']:<30} {r['mean_ari']:>9.4f} "
                  f"{r['var_ari']:>9.4f} {r['std_ari']:>9.4f} "
                  f"{r['iqr_ari']:>8.4f} {r['p60']*100:>7.1f}%")
        print("=" * len(hdr))
        if len(results) >= 2:
            r = results[0]["var_ari"] / results[-1]["var_ari"]
            d = (results[-1]["mean_ari"] - results[0]["mean_ari"]) * 100
            print(f"\nVariance reduction: x{r:.2f}   Mean ARI delta: {d:+.1f}%")
