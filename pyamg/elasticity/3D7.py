"""
2D Linear Elasticity Finite Element Method (FEM) solver
using triangular elements on a rectangular mesh.

Solves for displacement field (ux, uy) under applied boundary conditions
using the stiffness matrix assembly and direct linear solve.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyamg
import scipy.sparse as sp
from agg_vis import vis_aggregate_groups
from claude_agg_vis import vis_vector_aggregate_groups

# Suppress scientific notation in numpy print output (e.g. 1e-5 → 0.00001)
np.set_printoptions(suppress=True)


# ---------------------------------------------------------------------------
# 1. MESH GENERATION
# ---------------------------------------------------------------------------
# Build a structured rectangular grid of nx*ny nodes, divided into triangles.
# Each quad cell is split into 2 triangles, giving (nx-1)*(ny-1)*2 triangles total.

nx = 51  # number of nodes in x direction
ny = 51  # number of nodes in y direction

# geometry: shape (num_nodes, 2) — stores (x, y) coordinate of each node
num_nodes = nx * ny
geometry = np.zeros((num_nodes, 2), dtype=float)

c = 0  # node counter
for i in range(nx):
    for j in range(ny):
        geometry[c] = [
            float(i / (nx - 1)),
            float(j / (ny - 1)),
        ]  # [float(i), float(j)]
        c += 1


# topology: shape (num_triangles, 3) — stores the 3 node indices for each triangle
ntri = (nx - 1) * (ny - 1) * 2
topology = np.zeros((ntri, 3), dtype=int)

c = 0  # triangle counter
for i in range(nx - 1):
    for j in range(ny - 1):
        # ij is the bottom-left node of the current quad cell
        ij = j + i * ny
        # Split the quad into two triangles:
        #   Triangle 1: bottom-left, bottom-right, top-right
        topology[c] = [ij, ij + ny, ij + ny + 1]
        #   Triangle 2: bottom-left+1, bottom-left, top-right
        topology[c + 1] = [ij + 1, ij, ij + ny + 1]
        c += 2

# Pack geometry and topology together for convenience
mesh = (geometry, topology)


# ---------------------------------------------------------------------------
# 2. MESH PLOTTING UTILITY
# ---------------------------------------------------------------------------


def plot(mesh, data=None):
    """
    Plot the triangular mesh, optionally with a scalar field overlaid.

    Parameters
    ----------
    mesh : tuple of (geometry, topology)
    data : None, or array of length num_nodes (scalar field),
           or array of length 2*num_nodes (displacement [ux0, uy0, ux1, uy1, ...])
           For displacement data the magnitude sqrt(ux^2 + uy^2) is displayed.
    """
    geom, topo = mesh
    x = geom[:, 0]
    y = geom[:, 1]

    if data is not None:
        if len(data) == len(geom):
            # One scalar value per node — plot directly
            mag = data
        else:
            # Two values per node (ux, uy) interleaved — compute displacement magnitude
            assert len(data) == 2 * len(geom), (
                "data must have length num_nodes or 2*num_nodes"
            )
            ux = data[0::2]  # every other entry starting at 0 → x displacements
            uy = data[1::2]  # every other entry starting at 1 → y displacements
            mag = np.sqrt(ux**2 + uy**2)

        # Filled contour plot of the scalar/magnitude field on the triangulation
        plt.tricontourf(x, y, topo, mag, 40)

    # Draw the triangle edges in semi-transparent black
    plt.triplot(x, y, topo, color="k", alpha=0.5)

    # Add 10% padding around the mesh extents
    xmax, xmin = x.max(), x.min()
    ymax, ymin = y.max(), y.min()
    dx = 0.1 * (xmax - xmin)
    dy = 0.1 * (ymax - ymin)
    plt.xlim(xmin - dx, xmax + dx)
    plt.ylim(ymin - dy, ymax + dy)


# Plot the bare mesh to verify it looks correct
plt.figure(figsize=(8, 8))
plt.gca().set_aspect("equal")
plot(mesh)
plt.title("Undeformed Mesh")


# ---------------------------------------------------------------------------
# 3. ELEMENT STIFFNESS MATRIX
# ---------------------------------------------------------------------------


def Kmat(Darray, p, geometry):
    """
    Compute the 6x6 local stiffness matrix for a single triangular element.

    For a CST (Constant Strain Triangle) element in 2D elasticity, the
    displacement field within the element is linear and the strain is constant.

    Parameters
    ----------
    D        : (3, 3) ndarray — constitutive (material) matrix
    p        : (3,) int array — indices of the triangle's three nodes
    geometry : (num_nodes, 2) ndarray — node coordinates

    Returns
    -------
    K : (6, 6) ndarray — local element stiffness matrix
        DOF ordering: [ux0, uy0, ux1, uy1, ux2, uy2]
    """
    assert len(p) == 3, "Triangle must have exactly 3 nodes"

    x0, y0 = geometry[p[0]]
    x1, y1 = geometry[p[1]]
    x2, y2 = geometry[p[2]]

    b = 0.54
    if x0 < b and x1 < b and x2 < b:
        D = Darray[0]
    else:
        D = Darray[1]
    #    D = Darray[1]

    # Element area using the cross-product formula for a triangle
    # Ae = 0.5 * |det([x1-x0, x2-x0; y1-y0, y2-y0])|
    Ae = 0.5 * abs((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))

    # B matrix (strain-displacement matrix) maps nodal displacements → strains
    # Strains: [eps_xx, eps_yy, gamma_xy] = B @ [ux0, uy0, ux1, uy1, ux2, uy2]
    # The B matrix is constant within a CST element (strain is uniform).
    B = np.array(
        [
            # Row 0: d(ux)/dx terms (eps_xx)
            [y1 - y2, 0.0, y2 - y0, 0.0, y0 - y1, 0.0],
            # Row 1: d(uy)/dy terms (eps_yy)
            [0.0, x2 - x1, 0.0, x0 - x2, 0.0, x1 - x0],
            # Row 2: d(ux)/dy + d(uy)/dx terms (gamma_xy)
            [x2 - x1, y1 - y2, x0 - x2, y2 - y0, x1 - x0, y0 - y1],
        ]
    ) / (2 * Ae)

    # Local stiffness matrix: K = Ae * B^T D B
    # Integrating B^T D B over the element area (constant integrand → multiply by Ae)
    K = Ae * np.matmul(B.T, np.matmul(D, B))
    return K


# ---------------------------------------------------------------------------
# 4. GLOBAL STIFFNESS MATRIX ASSEMBLY
# ---------------------------------------------------------------------------


def assemble_matrix(mesh, Darray):
    """
    Assemble the global stiffness matrix by summing local element matrices.

    Each node has 2 DOFs (ux, uy), so the global matrix is (2*num_nodes) square.
    DOF ordering: node i → DOFs [2i, 2i+1] = [ux_i, uy_i].

    Parameters
    ----------
    mesh  : tuple of (geometry, topology)
    Dmat  : (3, 3) ndarray — constitutive matrix (same for all elements)

    Returns
    -------
    Kglobal : (2*num_nodes, 2*num_nodes) ndarray — assembled global stiffness matrix
    """
    geom, topo = mesh

    ndof = len(geom) * 2  # total degrees of freedom
    Kglobal = np.zeros((ndof, ndof))

    for tri in topo:
        # Compute local 6x6 stiffness matrix for this triangle
        K = Kmat(Darray, tri, geom)

        # Map local DOF indices [0..5] to global DOF indices
        # For nodes [n0, n1, n2], global DOFs are [2n0, 2n0+1, 2n1, 2n1+1, 2n2, 2n2+1]
        entries = np.empty(6, dtype=int)
        entries[0::2] = tri * 2  # x DOFs for each node
        entries[1::2] = tri * 2 + 1  # y DOFs for each node

        # Scatter local K into the global K (direct stiffness method)
        for i, idx in enumerate(entries):
            for j, jdx in enumerate(entries):
                Kglobal[idx, jdx] += K[i, j]

    return Kglobal


# ---------------------------------------------------------------------------
# 5. MATERIAL PROPERTIES & GLOBAL ASSEMBLY
# ---------------------------------------------------------------------------

# Plane-stress constitutive matrix D for isotropic linear elastic material.
# Relates engineering strains [eps_xx, eps_yy, gamma_xy] to stresses [sig_xx, sig_yy, tau_xy].
E = 200e3  # Young's modulus (arbitrary units)
nu = 0.3  # Poisson's ratio

Dmat0 = (E / (1 - nu**2)) * np.array(
    [
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2.0],
    ]
)

E = E / 100
Dmat1 = (E / (1 - nu**2)) * np.array(
    [
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2.0],
    ]
)

Darray = [Dmat0, Dmat1]

# Assemble the global stiffness matrix
Kglobal = assemble_matrix(mesh, Darray)
print("Global stiffness matrix K:")
print(Kglobal)

# Visualise the sparsity pattern of K (non-zero entries)
plt.figure(figsize=(8, 8))
plt.imshow((Kglobal != 0), interpolation="nearest", cmap="Greys")
plt.title("Non-zero structure of global stiffness matrix")


# ---------------------------------------------------------------------------
# 6. BOUNDARY CONDITIONS
# ---------------------------------------------------------------------------

geom, topo = mesh

rho = 20
g = 9.81
# Global force/load vector (zero = no body forces or surface tractions, only BCs)
# fglobal = np.zeros(len(geom) * 2)
fglobal = np.tile([0, -rho * g], len(geom))


def set_bc(K, f, row, val):
    """
    Enforce an essential (Dirichlet) boundary condition u[row] = val
    by zeroing the row and placing 1 on the diagonal, and setting f[row] = val.

    This is the 'row replacement' or 'penalty-free' BC method; it modifies
    the linear system so that the constrained DOF takes the prescribed value.

    Parameters
    ----------
    K   : global stiffness matrix (modified in-place)
    f   : global force vector (modified in-place)
    row : DOF index to constrain
    val : prescribed displacement value
    """
    K[row] = 0.0  # zero entire row
    K[row, row] = 1.0  # unit diagonal → equation becomes: 1 * u[row] = val
    f[row] = val


# Apply boundary conditions:
#   - Bottom edge (y == 0):       ux = 0, uy = 0  (fully fixed)
#   - Left edge (x == 0):         ux = 0.5        (prescribed horizontal pull)
#   - Top-right corner (15, 11):  ux = 0, uy = 0  (pinned to prevent rigid-body rotation)
for i, x in enumerate(geometry):
    # if x[1] == 0:  # bottom edge — fully fixed
    #     set_bc(Kglobal, fglobal, i * 2, 0.0)  # ux = 0
    #     set_bc(Kglobal, fglobal, i * 2 + 1, 0.0)  # uy = 0
    # if x[0] == 0:  # left edge — horizontal displacement prescribed
    #     set_bc(Kglobal, fglobal, i * 2, 0.5)  # ux = 0.5
    # if x[0] == 15 and x[1] == 11:  # top-right corner — pinned
    #     set_bc(Kglobal, fglobal, i * 2, 0.0)  # ux = 0
    #     set_bc(Kglobal, fglobal, i * 2 + 1, 0.0)  # uy = 0

    # if x[1] == 0:  # bottom edge — fully fixed
    #     set_bc(Kglobal, fglobal, i * 2, 0.0)  # ux = 0
    #     set_bc(Kglobal, fglobal, i * 2 + 1, 0.0)  # uy = 0
    if x[0] == 0:  # left edge — fully fixed
        set_bc(Kglobal, fglobal, i * 2, 0.0)  # ux = 0.0
        set_bc(Kglobal, fglobal, i * 2 + 1, 0.0)  # ux = 0.0
    if x[0] == 1.0:  # right edge - fully fixed
        set_bc(Kglobal, fglobal, i * 2, 0.0)  # ux = 0
        set_bc(Kglobal, fglobal, i * 2 + 1, 0.0)  # uy = 0


# ---------------------------------------------------------------------------
# 7. SOLVE THE LINEAR SYSTEM
# ---------------------------------------------------------------------------

# Solve K * u = f for the displacement vector u
# u is interleaved: [ux0, uy0, ux1, uy1, ..., ux_N, uy_N]
u = np.linalg.solve(Kglobal, fglobal)
print(f"Displacement range: max={u.max():.4f}, min={u.min():.4f}")

# Separate interleaved displacement vector into x and y components
ux = u[0::2]  # x-displacements (even indices)
uy = u[1::2]  # y-displacements (odd indices)

# Plot ux and uy side by side on the undeformed mesh
plt.figure(figsize=(16, 6))
plt.subplot(121, adjustable="box", aspect=1)
plt.title("x displacement (ux)")
plot(mesh, ux)
plt.colorbar()

plt.subplot(122, adjustable="box", aspect=1)
plt.title("y displacement (uy)")
plot(mesh, uy)
plt.colorbar()


# ---------------------------------------------------------------------------
# 8. DEFORMED MESH VISUALISATION
# ---------------------------------------------------------------------------


def move_mesh(mesh, u, alpha=1):
    """
    Displace mesh nodes by the computed displacement field.

    Parameters
    ----------
    mesh : tuple of (geometry, topology) — geometry is modified in-place
    u    : (2*num_nodes,) ndarray — interleaved [ux0, uy0, ux1, uy1, ...]
    """
    geom, topo = mesh
    num_nodes = len(geom)
    assert len(u) == num_nodes * 2, "u must have 2 entries per node"

    # Reshape u from flat [ux0, uy0, ...] to (num_nodes, 2) array of [ux, uy] per node
    v = u.reshape((num_nodes, 2))

    # Add displacement to each node coordinate in-place
    for x, dx in zip(geom, v):
        x += alpha * dx


# Work on a copy so the original mesh is preserved
mesh_copy = (np.array(mesh[0]), np.array(mesh[1]))
move_mesh(mesh_copy, u)

# Plot the deformed mesh coloured by displacement magnitude
plt.figure(figsize=(8, 8))
plt.gca().set_aspect("equal")
plt.title("Deformed mesh (coloured by displacement magnitude)")
plot(mesh_copy, u)
plt.colorbar()

plt.tight_layout()
# plt.show()

A = sp.csr_matrix(Kglobal)

ml = pyamg.smoothed_aggregation_solver(
    A,
    max_levels=5,
    max_coarse=10,
    keep=True,
    strength=("symmetric", {"theta": 0.3}),
)

AggOpArr = []
for idx, level in enumerate(ml.levels):
    if hasattr(level, "AggOp"):
        AggOpArr.append(level.AggOp)
        AggOpProduct = AggOpArr[0]

        if idx != 0:
            for a in AggOpArr[1:]:
                AggOpProduct = AggOpProduct @ a

        vis_vector_aggregate_groups(
            V=geometry,
            E2V=topology,
            AggOp=AggOpProduct,
            ndof=2,
            fname="agg" + str(idx) + ".png",
        )

# pyamg.vis.vis_coarse.vis_aggregate_groups(
#     V=geometry,
#     E2V=topology,
#     AggOp=AggOp0 @ AggOp1,
#     mesh_type="tri",
#     fname="output_aggs1.vtu",
# )

# pyamg.vis.vis_coarse.vis_aggregate_groups(
#     V=geometry,
#     E2V=topology,
#     AggOp=AggOp0 @ AggOp1 @ AggOp2,
#     mesh_type="tri",
#     fname="output_aggs2.vtu",
# )

pyamg.vis.vtk_writer.write_basic_mesh(
    V=geometry, E2V=topology, mesh_type="tri", fname="output_mesh.vtu"
)
print(geometry)
