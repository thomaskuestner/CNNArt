import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


x = np.arange(0, np.pi, 0.01)
y = np.arange(0, 2*np.pi, 0.01)
X, Y = np.meshgrid(x, y)
Z = np.cos(X) * np.sin(Y) * 10


colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # R -> G -> B
n_bins = [3, 10, 100, 1000]  # Discretizes the interpolation into bins
cmap_name = 'my_list'

fig, axs = plt.subplots(2, 2, figsize=(6, 9))
fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)

for n_bin, ax in zip(n_bins, axs.ravel()):
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # Fewer bins will result in "coarser" colomap interpolation
    im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cm)

    ax.set_title("N bins: %s" % n_bin)
    fig.colorbar(im, ax=ax)

plt.show()


