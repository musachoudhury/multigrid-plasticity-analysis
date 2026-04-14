import pyamg
from pyamg.vis import aggviz
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

data = pyamg.gallery.load_example("unit_square")
A = data["A"].tocsr()
V = data["vertices"]
E = data["elements"]

# print(E)

ml = pyamg.smoothed_aggregation_solver(
    A, max_coarse=10, keep=True, strength=("classical", {"theta": 0.1})
)

AggOp0 = ml.levels[0].AggOp
AggOp1 = ml.levels[1].AggOp
fig, ax = plt.subplots(ncols=2)

for axs in ax:
    axs.triplot(V[:, 0], V[:, 1], E, lw=0.5, color="tab:gray")
    axs.axis(False)
    # plt.show


# aggviz.plotaggs(AggOp0, V, sp.sparse.csr_matrix(A), ax[0], buffer=(0.1, -0.09))

# aggviz.plotaggs(AggOp0 @ AggOp1, V, sp.sparse.csr_matrix(A), ax[1], buffer=(0.2, -0.2))
# plt.show()

pyamg.vis.vis_coarse.vis_aggregate_groups(
    V=V,
    E2V=E,
    AggOp=AggOp0,
    mesh_type="tri",
    fname="output_aggs0.vtu",
)
