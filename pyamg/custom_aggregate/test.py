import numpy as np
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson
import pyamg

n = 6
A = poisson((n, n), format="csr")

# Build the 1D assignment array
agg = np.zeros(n * n, dtype=int)
agg_id = 0
for bi in range(0, n, 2):
    for bj in range(0, n, 2):
        for di in range(2):
            for dj in range(2):
                agg[(bi + di) * n + (bj + dj)] = agg_id
        agg_id += 1

n_nodes = n * n  # 36
n_aggs = agg_id  # 9

# Convert to AggOp sparse matrix (n_nodes × n_aggs)
rows = np.arange(n_nodes)  # one entry per node
cols = agg  # aggregate each node belongs to
data = np.ones(n_nodes, dtype=float)

AggOp = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_aggs))
print(AggOp.toarray())
