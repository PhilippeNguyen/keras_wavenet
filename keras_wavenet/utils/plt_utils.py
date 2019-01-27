import numpy as np
import matplotlib.pyplot as plt

def draw_heatmap(data,name):
    heatmap, xedges, yedges = np.histogram2d(data[:,0],data[:,1], bins=50)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title(name)
    plt.show()