# %%
import numpy as np
import matplotlib.pyplot as plt

sv = np.load("/home/scherbelam20/develop/sparse_wf/singular_values_0.npy")
sv = sv.squeeze()

fig, ax = plt.subplots()
ax.semilogy(sv, marker="o")
fig.show()
