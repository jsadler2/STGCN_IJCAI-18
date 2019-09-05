# coding: utf-8
from utils.math_graph import weight_matrix
import matplotlib.pyplot as plt
import pandas as pd
from utils.math_graph import scaled_laplacian, cheb_poly_approx


def plot_matrix(matrix, color_bar_label, filename):
    ax = plt.imshow(matrix, cmap="viridis_r")
    fig = plt.gcf()
    fig.colorbar(ax)
    axes = plt.gca()
    axes.set_ylabel('Station ID')
    axes.set_xlabel('Station ID')
    fig.text(0.97, 0.5, color_bar_label, rotation=270)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.cla()
    plt.close()


data_file = "data_loader/PeMS-M/PeMS-M/W_228.csv"

dfw = pd.read_csv(data_file, header=-1)
W = weight_matrix(data_file)
L = scaled_laplacian(W)
Ks = 3
n = 228
Lk = cheb_poly_approx(L, Ks, n)

plot_matrix(dfw.values, 'Distance', 'raw distance')
plot_matrix(L, 'laplacian', 'laplacian')
plot_matrix(W, 'Weighted Distance', 'weighted_distance')
for i in range(3):
    plot_matrix(Lk[:, i*n:(i+1)*n], f'cheb {i}', f'cheb_{i}')
