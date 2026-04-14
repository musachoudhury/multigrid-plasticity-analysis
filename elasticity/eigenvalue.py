# Code to evaluate eigenvalues of elasticity matrix

import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction

from mpi4py import MPI

from dolfinx import fem, io, plot, mesh
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType, GhostMode
from pathlib import Path
import pyvista
from petsc4py import PETSc
from utils import build_nullspace, plot_graph, gmsh_square
from gamg_opts import set_solver_options_gamg, set_solver_options_icc
import ufl
from dolfinx.io import gmsh as gmshio
import gmsh
from slepc4py import SLEPc

from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import diags
from scipy.sparse import csr_matrix
from scipy.linalg import eigh


def petsc_to_scipy(A_petsc):
    """Convert a PETSc Mat to a scipy csr_matrix."""
    ai, aj, av = A_petsc.getValuesCSR()
    return csr_matrix((av, aj, ai), shape=A_petsc.getSize())


dtype = PETSc.ScalarType

try:
    import pyvista

    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False

if PETSc.IntType == np.int64 and MPI.COMM_WORLD.size > 1:
    print(
        "This solver fails with PETSc and 64-bit integers because of memory errors in MUMPS."
    )
    # Note: when PETSc.IntType == np.int32, superlu_dist is used
    # rather than MUMPS and does not trigger memory failures.
    exit(0)

try:
    from dolfinx.io import VTXWriter

    has_vtx = True
except ImportError:
    print("VTXWriter not available, solution will not be saved.")
    has_vtx = False

size = MPI.COMM_WORLD.Get_size()

length, height = 1.0, 1.0
Nx, Ny = 50, 50
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
V_mat = fem.functionspace(domain, ("P", degree))
build_nullspace(V)


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
    a = 210e3
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
f = fem.Constant(domain, np.array([0, -rho * g * 10]))

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

problem = fem.petsc.LinearProblem(
    a,
    L,
    u=u_sol,
    bcs=bcs,
    petsc_options_prefix="elasticity",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

problem_iterative = fem.petsc.LinearProblem(
    a,
    L,
    u=u_sol,
    bcs=bcs,
    petsc_options_prefix="elasticity",
    petsc_options={"ksp_type": "gmres", "pc_type": "gamg"},
)

ksp_it = problem_iterative.solver

ksp_it.setMonitor(
    lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
)


# u_it = problem_iterative.solve()
# exit()
u_direct = problem.solve()


# set_solver_options_cg()
# set_solver_options_gamg()


A = fem.petsc.create_matrix(a_form)
ns = build_nullspace(V)
A.setNearNullSpace(ns)
# A.setOption(PETSc.Mat.Option.SPD, True)

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
solver.setOperators(A)


uh = fem.Function(V)


uh.x.petsc_vec.zeroEntries()
b.zeroEntries()

f.value[1] = -rho * g * 10  # * i * 0.1
A.zeroEntries()
fem.petsc.assemble_matrix(A, a_form, bcs=bcs)
A.assemble()

A_scipy = petsc_to_scipy(A)

A_dense = A_scipy.toarray()

eigenvalues, eigenvectors = np.linalg.eigh(A_dense)  # all eigenvalues, sorted

eigenvalues, eigenvectors = eigh(A_dense)

print(f"Got {len(eigenvalues)} eigenvalues")

# for e in eigenvalues:
#     print(e)

# remove bc artefact
eigenvalues = eigenvalues[np.abs(eigenvalues - 1.0) > 1e-6]
eigenvalues = np.sort(eigenvalues)

print(f"First sorted eigenvalue: {eigenvalues[0]}")
print(f"Min eigenvalue: {np.min(np.abs(eigenvalues))}")
print(f"Max eigenvalue: {np.max(np.abs(eigenvalues))}")
print(f"Condition number: {np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))}")

idx = np.argmin(np.abs(eigenvalues))
print(f"Smallest absolute eigenvalue: {eigenvalues[idx]}")
