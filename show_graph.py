import matplotlib.pyplot as plt
import networkx as nx
import sys
from os import path
from math import floor
from pylab import Rectangle

plt.figure()
nfigs = len(sys.argv)-1
fwid = int(floor(nfigs**0.5))
fhi = (nfigs-1)// fwid + 1

for i in range(nfigs):
    ax=plt.subplot(fwid, fhi, i+1)
    #  ax.set_title(path.split(sys.argv[i+1])[1])

    gr = nx.read_adjlist(sys.argv[i+1])
    pos = nx.drawing.nx_agraph.graphviz_layout(gr, prog="fdp")
    nx.draw_networkx(gr, pos, node_color='1.0', node_size=550, font_size=8, with_labels=True)

    autoAxis = ax.axis()
    rec = Rectangle(
            (autoAxis[0],autoAxis[2]),
            (autoAxis[1]-autoAxis[0]),
            (autoAxis[3]-autoAxis[2]),
            fill=False,lw=1)
    rec = ax.add_patch(rec)
    rec.set_clip_on(False)
plt.show()

