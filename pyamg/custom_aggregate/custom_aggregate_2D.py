import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyamg
from pyamg.gallery import poisson
from scipy.sparse import csr_matrix
from claude_agg_vis import vis_vector_aggregate_groups

# ── 1. Build the matrix ──────────────────────────────────────────────────────
n = 36  # 6×6 grid → 36 nodes

A = poisson((n, n), format="csr")

V = np.array([[j, i] for i in range(n) for j in range(n)], dtype=float)

elements = []
for i in range(n - 1):
    for j in range(n - 1):
        # Corner node indices
        n00 = i * n + j  # (i,   j  )
        n10 = (i + 1) * n + j  # (i+1, j  )
        n01 = i * n + (j + 1)  # (i,   j+1)
        n11 = (i + 1) * n + (j + 1)  # (i+1, j+1)

        elements.append([n00, n10, n01])  # lower-left triangle
        elements.append([n10, n11, n01])  # upper-right triangle

E2V = np.array(elements, dtype=int)

# ── 2. Build custom aggregates (2×2 blocks) ───────────────────────────────────
# agg is a 1D array of shape (n*n,); agg[k] = aggregate ID for node k
agg = np.zeros((n * n), dtype=int)
jump = 4
agg_id = 0
for bi in range(0, n, jump):
    for bj in range(0, n, jump):
        for di in range(jump):
            for dj in range(jump):
                agg[(bi + di) * n + (bj + dj)] = agg_id
        agg_id += 1

n_nodes = n * n
n_aggs = agg_id

rows = np.arange(n_nodes)
cols = agg
data = np.ones(n_nodes, dtype=float)


print(rows.shape)
print(cols.shape)
print(rows)
print(cols)
print(n_nodes)
print(n_aggs)

AggOp0 = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_aggs))
print(AggOp0.toarray())

n_nodes = n_aggs
n_aggs = 4

agg = np.zeros((n_nodes), dtype=int)
agg_id = 0
mid = len(agg) // 4
agg[:mid] = 0
agg[mid : 2 * mid] = 1
agg[2 * mid : 3 * mid] = 2
agg[3 * mid :] = 3

rows = np.arange(n_nodes)
cols = agg
data = np.ones(n_nodes, dtype=float)

AggOp1 = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_aggs))

print(AggOp1.toarray())

# print(agg.reshape(n, n))
# print(agg_csr.shape)
# print(agg_csr)

# # ── 3. Build solver — use keep=True to retain AggOp on levels ────────────────
ml = pyamg.smoothed_aggregation_solver(
    A,
    aggregate=[("predefined", {"AggOp": AggOp0}), ("predefined", {"AggOp": AggOp1})],
    keep=True,
    max_levels=3,
)

print(ml.levels[0].AggOp)
print(ml.levels[1].AggOp)

# Aggregate 0
vis_vector_aggregate_groups(
    V=V,
    E2V=E2V,
    AggOp=AggOp0,
    ndof=1,
    fname="Agg0.png",
)

# Aggregate 1
vis_vector_aggregate_groups(
    V=V,
    E2V=E2V,
    AggOp=AggOp0 @ AggOp1,
    ndof=1,
    fname="Agg1.png",
)
# print(ml)
# print(E)
# exit()

# # ── 4. Solve ──────────────────────────────────────────────────────────────────
# b = np.random.rand(A.shape[0])
# x = ml.solve(b, tol=1e-10, accel="cg")
# print(f"Relative residual: {np.linalg.norm(b - A @ x) / np.linalg.norm(b):.2e}")

# # ── 5. Inspect AggOp (sparse CSR, shape n_nodes × n_aggregates) ──────────────
# AggOp = ml.levels[0].AggOp
# print(f"AggOp shape: {AggOp.shape}")  # (36, 9)
# # AggOp.indices[k] gives the aggregate ID for node k (same as our agg array)

# # ── 6. Visualise using the inner/outer edge idiom from pyamg-examples ─────────
# # Build edge list from the sparsity pattern of A
# coo = A.tocoo()
# E = np.vstack((coo.row, coo.col)).T
# E = E[E[:, 0] < E[:, 1]]  # keep each edge once

# # Inner edges: both endpoints in the same aggregate
# inner = AggOp.indices[E[:, 0]] == AggOp.indices[E[:, 1]]
# outer = ~inner

# fig, ax = plt.subplots(figsize=(6, 6))

# # Draw outer (inter-aggregate) edges in red
# segs_outer = V[E[outer].ravel()].reshape(-1, 2, 2)
# col_outer = mpl.collections.LineCollection(
#     segs_outer, color=[232 / 255, 74 / 255, 39 / 255], linewidth=1.0, zorder=1
# )
# ax.add_collection(col_outer, autolim=True)

# # Draw inner (intra-aggregate) edges in dark blue, thicker
# segs_inner = V[E[inner].ravel()].reshape(-1, 2, 2)
# col_inner = mpl.collections.LineCollection(
#     segs_inner, color=[19 / 255, 41 / 255, 75 / 255], linewidth=4.0, zorder=2
# )
# ax.add_collection(col_inner, autolim=True)

# # Draw nodes coloured by aggregate
# ax.scatter(
#     V[:, 0],
#     V[:, 1],
#     c=AggOp.indices,
#     cmap="tab10",
#     s=200,
#     zorder=3,
#     edgecolors="k",
#     linewidths=0.5,
# )

# ax.autoscale_view()
# ax.set_aspect("equal")
# ax.set_title(
#     f"2D Poisson ({n}×{n}): custom 2×2 block aggregates\n"
#     f"Dark blue = intra-aggregate edges, Red = inter-aggregate edges"
# )
# plt.tight_layout()
# plt.show()
