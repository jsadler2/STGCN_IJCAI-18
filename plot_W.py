# coding: utf-8
from utils.math_graph import weight_matrix
import matplotlib.pyplot as plt
import pandas as pd

data_file = "data_loader/PeMS-M/PeMS-M/W_228.csv" 
# plot unweighted
dfw = pd.read_csv(data_file, header=-1)
ax = plt.imshow(dfw.values, cmap="viridis_r")
fig = plt.gcf()
fig.colorbar(ax)
axes = plt.gca()
axes.set_ylabel('Station ID')
axes.set_xlabel('Station ID')
fig.text(0.97, 0.5, 'distance [?]', rotation=270)
plt.tight_layout()
plt.show()
plt.clf()
plt.cla()
plt.close()


W = weight_matrix(data_file)
ax = plt.imshow(W, vmin=0.5, cmap="viridis_r")
fig = plt.gcf()
fig.colorbar(ax)
axes = plt.gca()
axes.set_ylabel('Station ID')
axes.set_xlabel('Station ID')
fig.text(0.97, 0.5, 'weighted distance ', rotation=270)
plt.tight_layout()
plt.show()
