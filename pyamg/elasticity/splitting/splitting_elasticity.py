import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction

from mpi4py import MPI

from dolfinx import fem, io, plot, mesh

from dolfinx.fem import assemble_matrix
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType, GhostMode
from pathlib import Path
import pyvista
from petsc4py import PETSc
from utils import nullspace_elasticty
import ufl
from dolfinx.io import gmsh as gmshio
import gmsh
import pyamg
import matplotlib as mplt
import matplotlib.pyplot as plt
import pyamg.vis
from pyamg.vis import aggviz
import scipy as sp

dtype = PETSc.ScalarType

size = MPI.COMM_WORLD.Get_size()
if MPI.COMM_WORLD.size > 1:
    print("This demo works only in serial.")
    exit(0)

length, height = 1.0, 1.0
Nx, Ny = 2, 2
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, height])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.none,
)

# gmsh.initialize()
# mesh_data = gmshio.model_to_mesh(
#     gmsh_square(),
#     MPI.COMM_WORLD,
#     rank=0,
#     gdim=2,
# )
# gmsh.finalize()

# domain = mesh_data.mesh

dim = domain.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 1
shape = (dim,)  # this means we want a vector field of size `dim`
V = fem.functionspace(domain, ("P", degree, shape))

domain.topology.create_connectivity(2, 0)

# element_to_vertex = domain.topology.connectivity(2, 0)
# print(element_to_vertex)
# print(element_to_vertex.array.reshape(-1, 3))
# exit()
V_mat = fem.functionspace(domain, ("P", degree))
nullspace_elasticty(V)


u_sol = fem.Function(V, name="Displacement")

# E = fem.Constant(domain, 210e3)
# nu = fem.Constant(domain, 0.3)


# lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
# mu = E / 2 / (1 + nu)


def epsilon(v):
    return sym(grad(v))


# def vector_expr(x):
#     # Must return shape (gdim, N)
#     values = np.zeros((domain.geometry.dim, x.shape[1]), dtype=np.float64)
#     values[0] = np.sin(np.pi * x[0])  # u_x component
#     values[1] = np.cos(np.pi * x[1])  # u_y component

#     return values

N_CHECK = 2


def checkerboard(x, n=N_CHECK):
    """
    Returns +1.0 or -1.0 at every point depending on which sub-square it falls in.

    Parameters
    ----------
    x : np.ndarray, shape (3, N)
        DOLFINx coordinate array: x[0]=x-coords, x[1]=y-coords.
    n : int
        Number of sub-squares per side.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    a = 210e0
    b = 210e3
    i = np.floor(n * x[0]).astype(int)  # column index of sub-square
    j = np.floor(n * x[1]).astype(int)  # row    index of sub-square
    return np.where((i + j) % 2 == 0, a, b)


E = fem.Function(V_mat)
E.interpolate(checkerboard)

# E = fem.Constant(domain, 210e3)


def sigma(v, E_val, nu_val):
    # E = fem.Constant(domain, E_val)
    # E = fem.Function(V)
    nu = fem.Constant(domain, nu_val)

    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    return lmbda * tr(epsilon(v)) * Identity(dim) + 2 * mu * epsilon(v)


u = TrialFunction(V)
v = TestFunction(V)

rho = 20
g = 9.81
f = fem.Constant(domain, np.array([0, -rho * g * 1000]))

cell_indices, cell_markers = [], []

# cellsinfo = [
#     (0, lambda x: np.less_equal(x[0], 0.5)),
#     (1, lambda x: np.greater(x[0], 0.49)),
# ]

cellsinfo = [
    (
        0,
        lambda x: np.logical_and(np.less_equal(x[0], 0.5), np.less_equal(x[1], 0.5)),
    ),  # bottom-left
    (
        1,
        lambda x: np.logical_and(np.greater(x[0], 0.49), np.less_equal(x[1], 0.5)),
    ),  # bottom-right
    (
        2,
        lambda x: np.logical_and(np.less_equal(x[0], 0.5), np.greater(x[1], 0.49)),
    ),  # top-left
    (
        3,
        lambda x: np.logical_and(np.greater(x[0], 0.49), np.greater(x[1], 0.49)),
    ),  # top-right
]

tdim = domain.topology.dim
for marker, locator in cellsinfo:
    cells_ = mesh.locate_entities(domain, tdim, locator)
    cell_indices.append(cells_)
    cell_markers.append(np.full(len(cells_), marker))

cell_indices = np.array(np.hstack(cell_indices), dtype=np.int32)
cell_markers = np.array(np.hstack(cell_markers), dtype=np.int32)
sorted_facets = np.argsort(cell_indices)
cell_tags = mesh.meshtags(
    domain, tdim, cell_indices[sorted_facets], cell_markers[sorted_facets]
)

metadata = {
    "quadrature_degree": 2,
    "quadrature_scheme": "default",
}

dx = Measure("dx", domain=domain, subdomain_data=cell_tags, metadata=metadata)


# a = (
#     inner(sigma(u, 210e3, 0.3), epsilon(v)) * dx(0)
#     + inner(sigma(u, 210e3, 0.3), epsilon(v)) * dx(1)
#     + inner(sigma(u, 210e3, 0.3), epsilon(v)) * dx(2)
#     + inner(sigma(u, 210e3, 0.3), epsilon(v)) * dx(3)
# )

a = inner(sigma(u, 210e3, 0.3), epsilon(v)) * dx

L = inner(f, v) * dx

a_form = fem.form(a)
L_form = fem.form(L)


def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], length)


left_dofs = fem.locate_dofs_geometrical(V, left)
right_dofs = fem.locate_dofs_geometrical(V, right)

bcs = [
    fem.dirichletbc(np.zeros((2,)), left_dofs, V),
    fem.dirichletbc(np.zeros((2,)), right_dofs, V),
]


b = fem.petsc.create_vector(V)

solver = PETSc.KSP().create(
    domain.comm,
)

solver.setMonitor(
    lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
)

solver.setComputeSingularValues(True)
solver.getComputeSingularValues()

solver.setFromOptions()
# solver.setOperators(A)


uh = fem.Function(V)


# print("finished step")
A_scipy = assemble_matrix(a_form, bcs).to_scipy()
# V_coords = domain.geometry.x[:, :2]
dof_coords = V.tabulate_dof_coordinates()


A = A_scipy.tocsr()
# E = np.vstack((A_csr.tocoo().row, A_csr.tocoo().col)).T  # edges of the matrix graph
# print(E)
domain.topology.create_connectivity(2, 0)
element_to_vertex = domain.topology.connectivity(2, 0)
E2V = element_to_vertex.array.reshape(-1, 4)

# V = dof_coords
E = E2V

# Use Ruge-Stuben Splitting Algorithm (use 'keep' in order to retain the splitting)
ml = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=1, CF="RS", keep=True)
print(ml)

# The CF splitting, 1 == C-node and 0 == F-node
splitting = ml.levels[0].splitting
C_nodes = splitting == 1
F_nodes = splitting == 0

fig, ax = plt.subplots()
alledges = dof_coords[E.ravel(), :].reshape((-1, 2, 2))
col = mplt.collections.LineCollection(alledges, color=[0.7, 0.7, 0.7], linewidth=1.0)
ax.add_collection(col, autolim=True)
ax.autoscale_view()

print(dof_coords[:, 0])
print(C_nodes)
print(len((C_nodes)))

# ax.scatter(
#     dof_coords[:, 0][C_nodes],
#     dof_coords[:, 1][C_nodes],
#     color=[232.0 / 255, 74.0 / 255, 39.0 / 255],
#     s=100.0,
#     label="C-pts",
#     zorder=10,
# )
# ax.scatter(
#     dof_coords[:, 0][F_nodes],
#     dof_coords[:, 1][F_nodes],
#     color=[19.0 / 255, 41.0 / 255, 75.0 / 255],
#     s=100.0,
#     label="F-pts",
#     zorder=10,
# )

# ax.axis("square")
# ax.set_xlabel("$x$")
# ax.set_ylabel("$y$")

# plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2)

# figname = "./output/splitting.png"
# import sys

# if len(sys.argv) > 1:
#     if sys.argv[1] == "--savefig":
#         plt.savefig(figname, bbox_inches="tight", dpi=150)
# else:
#     plt.show()
