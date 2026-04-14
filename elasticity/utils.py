import numpy as np
from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction

from mpi4py import MPI

from dolfinx import fem, la, plot

from dolfinx.mesh import create_rectangle, CellType

from petsc4py import PETSc
import pyvista
import gmsh

dtype = PETSc.ScalarType
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio


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


def plot_graph(s):
    plotter = pyvista.Plotter()
    cells, types, x = plot.vtk_mesh(s.function_space)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = s.x.array.reshape((x.shape[0], s.function_space.mesh.topology.dim))
    # plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_scalar("u", factor=100)
    plotter.add_mesh(warped, show_edges=False)
    plotter.show_axes()

    if pyvista.OFF_SCREEN:
        plotter.screenshot("results/deflection.png")
    else:
        plotter.show()
        plotter.close()


def create_mesh(comm: MPI.Comm, model: gmsh.model, name: str, filename: str, mode: str):
    """Create a DOLFINx from a Gmsh model and output to file.

    Args:
        comm: MPI communicator top create the mesh on.
        model: Gmsh model.
        name: Name (identifier) of the mesh to add.
        filename: XDMF filename.
        mode: XDMF file mode. "w" (write) or "a" (append).
    """
    mesh_data = gmshio.model_to_mesh(model, comm, rank=0)
    mesh_data.mesh.name = name
    if mesh_data.cell_tags is not None:
        mesh_data.cell_tags.name = f"{name}_cells"
    if mesh_data.facet_tags is not None:
        mesh_data.facet_tags.name = f"{name}_facets"
    if mesh_data.ridge_tags is not None:
        mesh_data.ridge_tags.name = f"{name}_ridges"
    if mesh_data.peak_tags is not None:
        mesh_data.peak_tags.name = f"{name}_peaks"
    with XDMFFile(mesh_data.mesh.comm, filename, mode) as file:
        mesh_data.mesh.topology.create_connectivity(2, 3)
        mesh_data.mesh.topology.create_connectivity(1, 3)
        mesh_data.mesh.topology.create_connectivity(0, 3)
        file.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            file.write_meshtags(
                mesh_data.cell_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.facet_tags is not None:
            file.write_meshtags(
                mesh_data.facet_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.ridge_tags is not None:
            file.write_meshtags(
                mesh_data.ridge_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )
        if mesh_data.peak_tags is not None:
            file.write_meshtags(
                mesh_data.peak_tags,
                mesh_data.mesh.geometry,
                geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{name}']/Geometry",
            )


def gmsh_square(lc: float = 0.0035 / 4, output: str = "unit_square.msh") -> None:
    """
    Create a unit-square [0,1]^2 triangular mesh in Gmsh and write it to *output*.

    Parameters
    ----------
    lc     : characteristic element size (smaller → finer mesh)
    output : path for the .msh file
    """

    gmsh.model.add("unit_square_tets")

    # ------------------------------------------------------------------
    # Geometry: four corners of [0,1]^2
    # ------------------------------------------------------------------
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

    # Boundary lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom  (y = 0)
    l2 = gmsh.model.geo.addLine(p2, p3)  # right   (x = 1)
    l3 = gmsh.model.geo.addLine(p3, p4)  # top     (y = 1)
    l4 = gmsh.model.geo.addLine(p4, p1)  # left    (x = 0)

    # Surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # ------------------------------------------------------------------
    # Physical groups (required by DOLFINx / gmshio)
    # ------------------------------------------------------------------
    gmsh.model.addPhysicalGroup(2, [surf], tag=2)
    gmsh.model.setPhysicalName(2, 2, "domain")

    # gmsh.model.addPhysicalGroup(1, [l1], tag=10)
    # gmsh.model.setPhysicalName(1, 10, "bottom")

    # gmsh.model.addPhysicalGroup(1, [l2], tag=11)
    # gmsh.model.setPhysicalName(1, 11, "right")

    # gmsh.model.addPhysicalGroup(1, [l3], tag=12)
    # gmsh.model.setPhysicalName(1, 12, "top")

    # gmsh.model.addPhysicalGroup(1, [l4], tag=13)
    # gmsh.model.setPhysicalName(1, 13, "left")

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    # gmsh.write(output)
    # print(f"Mesh written to '{output}'  (lc = {lc})")

    return gmsh.model()


# gmsh.initialize()
# gmsh.option.setNumber("General.Terminal", 0)

# # Create model
# model = gmsh_square()

# gmsh.finalize()

# create_mesh(MPI.COMM_WORLD, model, "tri_p1", "out_gmsh/mesh.xdmf", "w")
