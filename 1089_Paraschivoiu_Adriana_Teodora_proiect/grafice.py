import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from seaborn import scatterplot


def f_dendrograma(h, instante, titlu, distanta_prag=0):
    fig = plt.figure(titlu, figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontsize=18, color='c')
    dendrogram(h, labels=instante, ax=ax, color_threshold=distanta_prag)
    plt.show()


def histograma2(t, var, partitia):
    titlu = "Histograma -" + var
    fig = plt.figure(titlu, figsize=(14, 8))
    assert isinstance(fig, plt.Figure)
    fig.suptitle(titlu, fontsize=18, color='b')
    clusteri = np.unique(partitia)
    q = len(clusteri)
    axe = fig.subplots(1, q, sharey=True)
    for i in range(q):
        axa = axe[i]
        assert isinstance(axa, plt.Axes)
        axa.set_xlabel(clusteri[i])
        x = t[partitia == clusteri[i]][var].values
        axa.hist(x, rwidth=0.9, range=(min(t[var]), max(t[var])))
    plt.show()


def scatter2d(z, partitie, instante=None, titlu="Plot partitie in axe principale", aspect=1):
    fig = plt.figure(figsize=(10, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel("z1", fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel("z2", fontdict={"fontsize": 12, "color": "b"})
    ax.set_aspect(aspect)
    scatterplot(x=z[:, 0], y=z[:, 1], hue=partitie, ax=ax)
    if instante:
        n = len(instante)
        for i in range(n):
            ax.text(z[i, 0], z[i, 1], instante[i])
    plt.show()
