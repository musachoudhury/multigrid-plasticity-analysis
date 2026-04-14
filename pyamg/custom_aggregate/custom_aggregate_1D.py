import numpy as np
from scipy.sparse import csr_matrix

# Example: simple 1D Poisson (6x6)
from pyamg.gallery import poisson
from pyamg import smoothed_aggregation_solver

A = poisson((6,), format="csr")  # 6-node 1D grid

# 6 nodes → 3 aggregates of size 2
# nodes 0,1 → aggregate 0
# nodes 2,3 → aggregate 1
# nodes 4,5 → aggregate 2
my_agg = np.array([0, 0, 1, 1, 2, 2])

ml = smoothed_aggregation_solver(A, aggregate=[my_agg])
# Note: wrap in a list — one entry per level

b = np.ones(A.shape[0])
x = ml.solve(b, tol=1e-10)
print(x)
print(ml)  # shows hierarchy info
