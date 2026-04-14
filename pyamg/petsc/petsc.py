import sys
import petsc4py

petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

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
from claude_agg_vis import vis_vector_aggregate_groups


dtype = PETSc.ScalarType

size = MPI.COMM_WORLD.Get_size()
if MPI.COMM_WORLD.size > 1:
    print("This demo works only in serial.")
    exit(0)

length, height = 1.0, 1.0
Nx, Ny = 100, 100
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
    b = 210e20
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
    return np.isclose(x[0], 0.0)


def right(x):
    return np.isclose(x[0], length)


def line(x):
    return np.isclose(x[0], 0.4)


left_dofs = fem.locate_dofs_geometrical(V, left)
right_dofs = fem.locate_dofs_geometrical(V, right)
line_dofs = fem.locate_dofs_geometrical(V, line)

bcs = [
    fem.dirichletbc(np.zeros((2,)), left_dofs, V),
    fem.dirichletbc(np.zeros((2,)), right_dofs, V),
    # fem.dirichletbc(np.zeros((2,)), line_dofs, V),
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

theta_values = [0.12121]  # [0.113]  # [0.12121]  # np.linspace(0, 1, 100)  # 0.12121
results = {}


for theta in theta_values:
    ml = pyamg.smoothed_aggregation_solver(
        A,
        B=nullspace_elasticty(V),
        max_levels=11,
        max_coarse=10,
        keep=True,
        strength=("symmetric", {"theta": theta}),
        aggregate="naive",
    )
    print(f"\n--- theta = {theta} ---")
    print(ml)

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
                fname=f"plots/agg_theta{round(theta, 5)}_level{idx}.png",
            )
    results[theta] = ml
    # print((ml.levels[0].AggOp.shape))


b = np.random.rand(A.shape[0], 1)
accelerated_residuals = []
x = ml.solve(b, tol=1e-10, accel="cg", residuals=accelerated_residuals)

# Compute relative residuals
accelerated_residuals = np.array(accelerated_residuals) / accelerated_residuals[0]

print(f"KSP its: {len(accelerated_residuals)}")
# Plot convergence history
fig, ax = plt.subplots()
ax.set_title("Convergence History")
ax.set_xlabel("Iteration")
ax.set_ylabel("Relative Residual")
ax.semilogy(accelerated_residuals, label="Accelerated", linestyle="None", marker=".")
ax.legend()

# figname = "./output/convergence.png"
# import sys

# if "--savefig" in sys.argv:
#     plt.savefig(figname, bbox_inches="tight", dpi=150)
# else:
#     plt.show()

# --- Build a simple 1D Laplacian as a test matrix ---
# n = 64
# A = PETSc.Mat().createAIJ([n, n], nnz=3)
# for i in range(n):
#     if i > 0:
#         A.setValue(i, i - 1, -1.0)
#     A.setValue(i, i, 2.0)
#     if i < n - 1:
#         A.setValue(i, i + 1, -1.0)
# A.assemblyBegin()
# A.assemblyEnd()

# --- Set up KSP/PC with GAMG ---
ksp = PETSc.KSP().create()
ksp.setOperators(A)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.GAMG)
pc.setGAMGType("agg")

# Use 0 smoothing steps to get the raw (unsmoothed) aggregation operator,
# i.e. the closest equivalent to PyAMG's AggOp
pc.setFromOptions()
PETSc.Options()["pc_gamg_agg_nsmooths"] = 0

# Force PC setup without a full solve
ksp.setFromOptions()
pc.setUp()

# --- Extract the prolongation operators ---
nlevels = pc.getMGLevels()
print(f"Number of MG levels: {nlevels}")

for level in range(1, nlevels):
    P = pc.getMGInterpolation(level)  # prolongation from level-1 to level
    print(f"Level {level}: P shape = {P.getSize()}")

    # Convert to a scipy sparse matrix if needed
    ai, aj, av = P.getValuesCSR()
    nrows, ncols = P.getSize()
    from scipy.sparse import csr_matrix

    P_scipy = csr_matrix((av, aj, ai), shape=(nrows, ncols))
    print(f"  P as scipy CSR: {P_scipy.shape}, nnz={P_scipy.nnz}")

##########################

vis_vector_aggregate_groups(
    V=dof_coords[:, 0:2],
    E2V=E2V,
    AggOp=P_scipy,
    ndof=2,
    fname="test.png",
)
