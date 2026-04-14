import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction

from mpi4py import MPI

from dolfinx import fem, io, plot, la
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType
from pathlib import Path
import pyvista
from petsc4py import PETSc

dtype = PETSc.ScalarType


def build_nullspace(V: fem.FunctionSpace):
    """Build PETSc nullspace for 2D elasticity."""
    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(3)]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(2)]

    # Set the three translational rigid body modes
    for i in range(2):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[2][dofs[0]] = -x1
    b[2][dofs[1]] = x0

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=2, comm=V.mesh.comm)
        for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


length, height = 1.0, 1.0
Nx, Ny = 100, 100
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, height])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)

dim = domain.topology.dim
print(f"Mesh topology dimension d={dim}.")

degree = 2
shape = (dim,)  # this means we want a vector field of size `dim`
V = fem.functionspace(domain, ("P", degree, shape))
build_nullspace(V)


u_sol = fem.Function(V, name="Displacement")

E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return sym(grad(v))


def sigma(v):
    return lmbda * tr(epsilon(v)) * Identity(dim) + 2 * mu * epsilon(v)


# print("mu (UFL):\n", mu)
# print("epsilon (UFL):\n", epsilon(u_sol))
# print("sigma (UFL):\n", sigma(u_sol))

u = TrialFunction(V)
v = TestFunction(V)

rho = 2000
g = 9.81
f = fem.Constant(domain, np.array([0, -rho * g]))

dx = Measure("dx", domain=domain)
a = inner(sigma(u), epsilon(v)) * dx
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
    petsc_options_prefix="demo_2D_elasticity",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)
u_direct = problem.solve()

# Set solver options
opts = PETSc.Options()
# opts["ksp_type"] = "richardson"
# opts["ksp_rtol"] = 1.0e-8
# opts["ksp_atol"] = 1.0e-8
# opts["ksp_max_it"] = 10000
# # opts["ksp_atol"] = 1.0e-8
# opts["pc_type"] = "sor"
# opts["pc_sor_omega"] = 1.0  # omega=1.0 → Gauss-Seidel (not SOR)
# opts["pc_sor_symmetric"] = None  # symmetric GS (forward + backward sweep)
# opts["pc_sor_its"] = 1  # GS sweeps per PC application

opts["ksp_type"] = "preonly"
opts["ksp_rtol"] = 1.0e-8
opts["ksp_atol"] = 1.0e-8
opts["pc_type"] = "gamg"
opts["ksp_max_it"] = 0

A = fem.petsc.create_matrix(a_form)
ns = build_nullspace(V)
A.setNearNullSpace(ns)
# A.setOption(PETSc.Mat.Option.SPD, True)

b = fem.petsc.create_vector(V)

solver = PETSc.KSP().create(
    domain.comm,
)
solver.setFromOptions()
solver.setOperators(A)

A.zeroEntries()
fem.petsc.assemble_matrix(A, a_form, bcs=bcs)
A.assemble()


fem.petsc.assemble_vector(b, L_form)

fem.petsc.apply_lifting(b, [a_form], [bcs], alpha=-1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# Set du|_bc = u_{i-1}-u_D
fem.petsc.set_bc(b, bcs)

uh = fem.Function(V)
solver.setMonitor(
    lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}")
)
solver.solve(b, uh.x.petsc_vec)

error = fem.Function(V)
error.x.array[:] = u_direct.x.array[:] - uh.x.array[:]
# V_uy, mapping = V.sub(1).collapse()
# right_dofs_uy = fem.locate_dofs_geometrical((V.sub(1), V_uy), right)

# uD_y = fem.Function(V_uy)
# bcs2 = [
#     fem.dirichletbc(np.zeros((2,)), left_dofs, V),
#     fem.dirichletbc(uD_y, right_dofs_uy, V.sub(1)),
# ]

# problem = fem.petsc.LinearProblem(
#     a,
#     L,
#     u=u_sol,
#     bcs=bcs2,
#     petsc_options_prefix="demo_2D_elasticity",
#     petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
# )
# problem.solve()


vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
vtk.write_function(u_sol)
vtk.close()


out_folder = Path("out_poisson")
out_folder.mkdir(parents=True, exist_ok=True)
with io.XDMFFile(domain.comm, out_folder / "poisson.xdmf", "w") as file:
    file.write_mesh(domain)
    # file.write_function(error)


def plot_graph(s):
    plotter = pyvista.Plotter()
    cells, types, x = plot.vtk_mesh(s.function_space)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = s.x.array.reshape((x.shape[0], dim))
    # plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_scalar("u", factor=100)
    plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()

    if pyvista.OFF_SCREEN:
        plotter.screenshot(out_folder / "deflection.png")
    else:
        plotter.show()
        plotter.close()


# plot_graph(u_sol)
# plot_graph(uh)
plot_graph(error)
