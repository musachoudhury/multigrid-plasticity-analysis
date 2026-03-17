# %%
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
import pyvista


msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(100, 100),
    cell_type=mesh.CellType.triangle,
)
V = fem.functionspace(msh, ("Lagrange", 1))

tdim = msh.topology.dim
fdim = tdim - 1
facets = mesh.locate_entities_boundary(
    msh,
    dim=fdim,
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=facets)


bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

cell_indices, cell_markers = [], []

cellsinfo = [
    (0, lambda x: np.less_equal(x[1], 0.5)),
    (1, lambda x: np.greater(x[1], 0.5)),
]

for marker, locator in cellsinfo:
    cells_ = mesh.locate_entities(msh, tdim, locator)
    cell_indices.append(cells_)
    cell_markers.append(np.full(len(cells_), marker))

cell_indices = np.array(np.hstack(cell_indices), dtype=np.int32)
cell_markers = np.array(np.hstack(cell_markers), dtype=np.int32)
sorted_facets = np.argsort(cell_indices)
cell_tags = mesh.meshtags(
    msh, tdim, cell_indices[sorted_facets], cell_markers[sorted_facets]
)

dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)

# +
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

# f = 10*ufl.sin(2*ufl.pi*x[0])+10*ufl.sin(10*ufl.pi*x[0])
# #f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
# f = 10*ufl.sin(2*ufl.pi*x[0])+10*ufl.sin(10*ufl.pi*x[0])
# f = 10*ufl.sin(5*ufl.pi*x[0])
f0 = 20 * ufl.sin(2 * ufl.pi * x[0])
f1 = 750 * ufl.sin(20 * ufl.pi * x[0])

# g = ufl.sin(1000 * x[0])
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f0, v) * dx(0) + ufl.inner(f1, v) * dx(1)  # + ufl.inner(g, v) * ufl.ds

a_form = fem.form(a)
L_form = fem.form(L)

# -
# +
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

# Set solver options
opts = PETSc.Options()
opts["ksp_type"] = "richardson"
opts["ksp_rtol"] = 1.0e-8
opts["ksp_max_it"] = 100
# opts["ksp_atol"] = 1.0e-8
opts["pc_type"] = "jacobi"


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

error = fem.Function(V)
error.x.array[:] = u_direct.x.array[:] - uh.x.array[:]

assert isinstance(uh, fem.Function)
# assert isinstance(error, fem.Function)
# -

# The solution can be written to a {py:class}`XDMFFile
# <dolfinx.io.XDMFFile>` file visualization with [ParaView](https://www.paraview.org/)
# or [VisIt](https://visit-dav.github.io/visit-website/):

# +


out_folder = Path("out_poisson")
out_folder.mkdir(parents=True, exist_ok=True)
with io.XDMFFile(msh.comm, out_folder / "poisson.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_function(error)
# -

# and displayed using [pyvista](https://docs.pyvista.org/).


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


# %%
