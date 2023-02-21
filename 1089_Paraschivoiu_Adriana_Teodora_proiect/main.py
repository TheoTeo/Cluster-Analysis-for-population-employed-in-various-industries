import pandas as pd
from functii import *
from grafice import f_dendrograma, histograma2, scatter2d
from sklearn.decomposition import PCA
from tkinter import *

t = pd.read_csv("datead.csv", index_col=1)
nan_replace(t)

variabile = list(t)[1:]
model = hclust(t, variabile)


def Dendrograma():
    f_dendrograma(model.h, model.instante, "Plot ierarhie")


t_partitii = pd.DataFrame(index=model.instante)
p_o = model.partitie("Partitia optimala")
adaugare_partitie(t_partitii, p_o, "Partitia optimala")
p_3 = model.partitie("Partitia cu 3 clusteri", nr_clusteri=3)
adaugare_partitie(t_partitii, p_3, "Partitia cu 3 clusteri")
t_partitii.to_csv("partitii.csv")


def HistogrameOpt():
    for v in variabile:
        histograma2(t, v, p_o)


# Histograme
def Histograme():
    for v in variabile:
        histograma2(t, v, p_3)


acp = PCA(2)
acp.fit(model.x)
z = acp.transform(model.x)


def Scatter():
    scatter2d(z, p_3, model.instante)


window = Tk()
window.geometry("500x500")
btnDendrograma = Button(window, text="Grafic Dendrograma")
btnDendrograma.config(command=Dendrograma)
btnDendrograma.pack()

btnHistograma = Button(window, text="Grafice Histograma dupa Partitia cu 3 clusteri");
btnHistograma.config(command=Histograme)
btnHistograma.pack()

btnHistograma2 = Button(window, text="Grafice Histograma dupa Partitia Optimala");
btnHistograma2.config(command=HistogrameOpt)
btnHistograma2.pack()

btnScatter = Button(window, text="Grafic Scatter");
btnScatter.config(command=Scatter)
btnScatter.pack()

window.mainloop()
