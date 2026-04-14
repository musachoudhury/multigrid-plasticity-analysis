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
from utils import nullspace_elasticty, plot_graph, gmsh_square
from gamg_opts import set_solver_options_gamg, set_solver_options_icc
import ufl
from dolfinx.io import gmsh as gmshio
import gmsh
import pyamg
import matplotlib as mplt
import matplotlib.pyplot as plt
import pyamg.vis
from pyamg.vis import aggviz
import scipy as sp
from claude_agg_vis import vis_vector_aggregate_groups


dtype = PETSc.ScalarType

size = MPI.COMM_WORLD.Get_size()
if MPI.COMM_WORLD.size > 1:
    print("This demo works only in serial.")
    exit(0)

length, height = 1.0, 1.0
Nx, Ny = 99, 99
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, height])],
    [Nx, Ny],
    cell_type=CellType.triangle,
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

tdim = domain.topology.dim


metadata = {
    "quadrature_degree": 2,
    "quadrature_scheme": "default",
}

dx = Measure("dx", domain=domain, metadata=metadata)

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

u_direct = problem.solve()

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


domain.topology.create_connectivity(2, 0)
element_to_vertex = domain.topology.connectivity(2, 0)
E2V = element_to_vertex.array.reshape(-1, 3)

ml = pyamg.smoothed_aggregation_solver(
    A,
    B=nullspace_elasticty(V),
    max_levels=4,
    max_coarse=10,
    keep=True,
    strength=("symmetric", {"theta": 0.1}),
)

fig, ax = plt.subplots(ncols=3)

V = dof_coords
E = E2V

AggOpArr = []
for idx, level in enumerate(ml.levels):
    if hasattr(level, "AggOp"):
        AggOpArr.append(level.AggOp)
        AggOpProduct = AggOpArr[0]

        if idx != 0:
            for a in AggOpArr[1:]:
                AggOpProduct = AggOpProduct @ a

        vis_vector_aggregate_groups(
            V=dof_coords[:, 0:2],
            E2V=E2V,
            AggOp=AggOpProduct,
            ndof=2,
            fname="agg" + str(idx) + ".png",
        )

# create the vtk file for a mesh
pyamg.vis.vtk_writer.write_basic_mesh(
    V=dof_coords, E2V=E2V, mesh_type="tri", fname="output_mesh.vtu"
)
