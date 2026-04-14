import warnings
import numpy as np
from scipy.sparse import csr_array, coo_array, triu
from pyamg.vis.vtk_writer import write_vtu


def vis_aggregate_groups(V, E2V, AggOp, mesh_type, fname="output.vtu"):
    """Coarse grid visualization of aggregate groups.

    Create .vtu files for use in Paraview or display with Matplotlib.

    Parameters
    ----------
    V : {array}
        coordinate array (N x D)
    E2V : {array}
        element index array (Nel x Nelnodes)
    AggOp : {csr_array}
        sparse matrix for the aggregate-vertex relationship (N x Nagg)
    mesh_type : {string}
        type of elements: vertex, tri, quad, tet, hex (all 3d)
    fname : {string, file object}
        file to be written, e.g. 'output.vtu'

    Returns
    -------
    Writes data to .vtu file for use in paraview (xml 0.1 format) or
    displays to screen using matplotlib

    Notes
    -----
    Works for both 2d and 3d elements.  Element groupings are colored
    with data equal to 2.0 and stringy edges in the aggregate are colored
    with 3.0

    Examples
    --------
    >>> from pyamg.aggregation import standard_aggregation
    >>> from pyamg.vis.vis_coarse import vis_aggregate_groups
    >>> from pyamg.gallery import load_example
    >>> data = load_example('unit_square')
    >>> A = data['A'].tocsr()
    >>> V = data['vertices']
    >>> E2V = data['elements']
    >>> AggOp = standard_aggregation(A)[0]
    >>> vis_aggregate_groups(V=V, E2V=E2V, AggOp=AggOp,
    ...                      mesh_type='tri', fname='output.vtu')
    >>> from pyamg.aggregation import standard_aggregation
    >>> from pyamg.vis.vis_coarse import vis_aggregate_groups
    >>> from pyamg.gallery import load_example
    >>> data = load_example('unit_cube')
    >>> A = data['A'].tocsr()
    >>> V = data['vertices']
    >>> E2V = data['elements']
    >>> AggOp = standard_aggregation(A)[0]
    >>> vis_aggregate_groups(V=V, E2V=E2V, AggOp=AggOp,
    ...                      mesh_type='tet', fname='output.vtu')

    """
    check_input(V=V, E2V=E2V, AggOp=AggOp, mesh_type=mesh_type)
    map_type_to_key = {"tri": 5, "quad": 9, "tet": 10, "hex": 12}
    if mesh_type not in map_type_to_key:
        raise ValueError(f"Unknown mesh_type={mesh_type}")
    key = map_type_to_key[mesh_type]

    AggOp = csr_array(AggOp)

    # remove elements with dirichlet BCs
    if E2V.max() >= AggOp.shape[0]:
        E2V = E2V[E2V.max(axis=1) < AggOp.shape[0]]

    # 1 #
    # Find elements with all vertices in same aggregate

    # account for 0 rows.  Mark them as solitary aggregates
    if len(AggOp.indices) != AggOp.shape[0]:
        full_aggs = ((AggOp.indptr[1:] - AggOp.indptr[:-1]) == 0).nonzero()[0]
        new_aggs = np.array(AggOp.sum(axis=1), dtype=int).ravel()
        new_aggs[full_aggs == 1] = AggOp.indices  # keep existing aggregate IDs
        new_aggs[full_aggs == 0] = AggOp.shape[1]  # fill in singletons maxID+1
        ElementAggs = new_aggs[E2V]
    else:
        ElementAggs = AggOp.indices[E2V]

    # 2 #
    # find all aggregates encompassing full elements
    # mask[i] == True if all vertices in element i belong to the same aggregate
    mask = np.where(abs(np.diff(ElementAggs)).max(axis=1) == 0)[0]
    # mask = (ElementAggs[:,:] == ElementAggs[:,0]).all(axis=1)
    E2V_a = E2V[mask, :]  # elements where element is full
    Nel_a = E2V_a.shape[0]

    # 3 #
    # find edges of elements in the same aggregate (brute force)

    # construct vertex to vertex graph
    col = E2V.ravel()
    row = np.kron(np.arange(0, E2V.shape[0]), np.ones((E2V.shape[1],), dtype=int))
    data = np.ones((len(col),))
    if len(row) != len(col):
        raise ValueError("Problem constructing vertex-to-vertex map")
    V2V = coo_array((data, (row, col)), shape=(E2V.shape[0], E2V.max() + 1))
    V2V = V2V.T @ V2V
    V2V = triu(V2V, 1).tocoo()

    # get all the edges
    edges = np.vstack((V2V.row, V2V.col)).T

    # all the edges in the same aggregate
    E2V_b = edges[AggOp.indices[V2V.row] == AggOp.indices[V2V.col]]
    Nel_b = E2V_b.shape[0]

    # 3.5 #
    # single node aggregates
    sums = np.array(AggOp.sum(axis=0)).ravel()
    E2V_c = np.argwhere(np.isin(AggOp.indices, np.where(sums == 1)[0])).ravel()
    Nel_c = len(E2V_c)

    # 4 #
    # now write out the elements and edges
    colors_a = 3 * np.ones((Nel_a,))  # color triangles with threes
    colors_b = 2 * np.ones((Nel_b,))  # color edges with twos
    colors_c = 1 * np.ones((Nel_c,))  # color the vertices with ones

    cells = {1: E2V_c, 3: E2V_b, key: E2V_a}
    cdata = {1: colors_c, 3: colors_b, key: colors_a}  # make sure it's a tuple
    write_vtu(V=V, cells=cells, fname=fname, cdata=cdata)


def check_input(V=None, E2V=None, AggOp=None, A=None, splitting=None, mesh_type=None):
    """Check input for local functions."""
    if V is not None and not np.issubdtype(V.dtype, np.floating):
        raise ValueError("V should be of type float")

    if E2V is not None:
        if not np.issubdtype(E2V.dtype, np.integer):
            raise ValueError("E2V should be of type integer")
        if E2V.min() != 0:
            warnings.warn(f"Element indices begin at {E2V.min()}", stacklevel=2)

    if AggOp is not None and AggOp.shape[1] > AggOp.shape[0]:
        raise ValueError("AggOp should be of size N x Nagg")

    if A is not None and AggOp is None:
        raise ValueError("problem with check_input")

    if (
        A is not None
        and AggOp is not None
        and ((A.shape[0] != A.shape[1]) or (A.shape[0] != AggOp.shape[0]))
    ):
        raise ValueError("expected square matrix A and compatible with AggOp")

    if splitting is not None and V is None:
        raise ValueError("problem with check_input")

    if splitting is not None:
        splitting = splitting.ravel()
        if V is not None and (len(splitting) % V.shape[0]) != 0:
            raise ValueError("splitting must be a multiple of N")

    if mesh_type is not None:
        valid_mesh_types = ("vertex", "tri", "quad", "tet", "hex")
        if mesh_type not in valid_mesh_types:
            raise ValueError(f"mesh_type should one of {valid_mesh_types}")
