# %%
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
import pyvista


msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(100, 100),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 2))


tdim = msh.topology.dim
fdim = tdim - 1
facets = mesh.locate_entities_boundary(
    msh,
    dim=fdim,
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)


bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

dx = ufl.Measure("dx", domain=msh)

# +
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

f0 = 2 * ufl.sin(ufl.pi * x[0])
f1 = 750 * ufl.sin(20 * ufl.pi * x[0])

N_CHECK = 4


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
    a = 4000
    b = 10000
    i = np.floor(n * x[0]).astype(int)  # column index of sub-square
    j = np.floor(n * x[1]).astype(int)  # row    index of sub-square
    return np.where((i + j) % 2 == 0, a, b)


k = fem.Function(V)
k.interpolate(checkerboard)

a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
L = ufl.inner(f0, v) * dx

a_form = fem.form(a)
L_form = fem.form(L)

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="demo_poisson_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
    },
)
# Direct solution
u_direct = problem.solve()

# Iterative solution

opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["ksp_rtol"] = 1.0e-8
opts["ksp_max_it"] = 100
opts["ksp_atol"] = 1.0e-8
opts["pc_type"] = "none"

A = fem.petsc.create_matrix(a_form)
b = fem.petsc.create_vector(V)

solver = PETSc.KSP().create(
    msh.comm,
)
solver.setFromOptions()
solver.setOperators(A)

A.zeroEntries()
fem.petsc.assemble_matrix(A, a_form, bcs=[bc])
A.assemble()


fem.petsc.assemble_vector(b, L_form)

fem.petsc.apply_lifting(b, [a_form], [[bc]], alpha=-1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# Set du|_bc = u_{i-1}-u_D
fem.petsc.set_bc(b, [bc])

uh = fem.Function(V)
solver.setMonitor(
    lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
)
solver.solve(b, uh.x.petsc_vec)


reason = solver.getConvergedReason()

print(reason)
if reason < 0:
    print("Krylov solver did not converge!")

error = fem.Function(V)
error.x.array[:] = u_direct.x.array[:] - uh.x.array[:]

assert isinstance(uh, fem.Function)


out_folder = Path("out_poisson")
out_folder.mkdir(parents=True, exist_ok=True)


def plot_graph(s):
    cells, types, x = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = s.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        plotter.screenshot(out_folder / "uh_poisson.png")
    else:
        plotter.show()
        plotter.close()


plot_graph(u_direct)
plot_graph(uh)
plot_graph(error)
