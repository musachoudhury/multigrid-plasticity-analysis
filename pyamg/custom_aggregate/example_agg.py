import numpy as np
import matplotlib.pyplot as plt
from pyamg.gallery import poisson
from pyamg import smoothed_aggregation_solver

# ── 1. Build the matrix ──────────────────────────────────────────────────────
n = 4  # 6×6 grid → 36 nodes
A = poisson((n, n), format="csr")

# ── 2. Build aggregates based on 2×2 blocks ──────────────────────────────────
# Node (i,j) maps to linear index: i*n + j
# We assign each node to a 2×2 block aggregate

agg = np.zeros(n * n, dtype=int)

agg_id = 0
for bi in range(0, n, 2):  # block row: 0, 2, 4
    for bj in range(0, n, 2):  # block col: 0, 2, 4
        for di in range(2):  # within-block row offset
            for dj in range(2):  # within-block col offset
                node = (bi + di) * n + (bj + dj)
                agg[node] = agg_id
        agg_id += 1

# Result: 9 aggregates of 4 nodes each (for a 6×6 grid)
print(f"Aggregates: {agg_id}, nodes per agg: {np.bincount(agg)}")

# ── 3. Visualise ─────────────────────────────────────────────────────────────
agg_grid = agg.reshape(n, n)

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(agg_grid, cmap="tab10", origin="upper")
plt.colorbar(im, ax=ax, label="Aggregate ID")

# Annotate each cell with its aggregate ID
for i in range(n):
    for j in range(n):
        ax.text(
            j,
            i,
            str(agg_grid[i, j]),
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            fontweight="bold",
        )

# Draw block boundaries
for k in range(0, n + 1, 2):
    ax.axhline(k - 0.5, color="black", linewidth=2)
    ax.axvline(k - 0.5, color="black", linewidth=2)

ax.set_title("2×2 Block Aggregates on 6×6 Grid")
ax.set_xticks(range(n))
ax.set_yticks(range(n))
plt.tight_layout()
plt.show()

# ┌───┬───┬───┐
# │ 0 │ 1 │ 2 │
# ├───┼───┼───┤
# │ 3 │ 4 │ 5 │
# ├───┼───┼───┤
# │ 6 │ 7 │ 8 │
# └───┴───┴───┘

# ── 4. Build solver with custom aggregates ───────────────────────────────────
ml = smoothed_aggregation_solver(A, aggregate=[agg])
print(ml)

# ── 5. Solve ─────────────────────────────────────────────────────────────────
b = np.ones(A.shape[0])
x = ml.solve(b, tol=1e-10, accel="cg")

residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
print(f"Relative residual: {residual:.2e}")
