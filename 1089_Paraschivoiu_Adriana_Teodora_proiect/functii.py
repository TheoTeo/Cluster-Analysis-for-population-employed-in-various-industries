import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from scipy.cluster.hierarchy import linkage
from grafice import f_dendrograma
from sklearn.cluster import AgglomerativeClustering


def nan_replace(t):
    assert isinstance(t, pd.DataFrame)
    nume_variabile = list(t.columns)
    for v in nume_variabile:
        if any(t[v].isna()):
            if is_numeric_dtype(t[v]):
                t[v].fillna(t[v].mean(), inplace=True)
            else:
                modulul = t[v].mode()[0]
                t[v].fillna(modulul, inplace=True)


class hclust():
    def __init__(self, t, variabile, metoda="ward"):
        self.x = t[variabile].values
        self.metoda = metoda
        self.instante = list(t.index)
        self.h = linkage(self.x, method=metoda)
        print(self.h)

    def partitie(self, nume, nr_clusteri=None):
        p = self.h.shape[0]
        if nr_clusteri is None:
            k_dif_max = np.argmax(self.h[1:, 2] - self.h[:(p - 1), 2])
            nr_clusteri = p - k_dif_max
        else:
            k_dif_max = p - nr_clusteri

        distanta_prag = (self.h[k_dif_max, 2] + self.h[k_dif_max + 1, 2]) / 2
        f_dendrograma(self.h, self.instante, nume, distanta_prag)
        model_skl = AgglomerativeClustering(n_clusters=nr_clusteri, linkage=self.metoda)
        model_skl.fit(self.x)
        coduri = model_skl.labels_
        return np.array(["c" + str(i + 1) for i in coduri])


def adaugare_partitie(t, p, nume):
    t[nume] = p
