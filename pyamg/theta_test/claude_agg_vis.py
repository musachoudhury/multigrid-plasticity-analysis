"""
vis_vector_aggregates.py
------------------------
Matplotlib-based visualisation of AMG aggregate groups for vector-valued
problems (multiple degrees of freedom per mesh vertex).

The key difference from the original vis_aggregate_groups is the introduction
of a `ndof` parameter.  When ndof > 1, the AggOp has ndof*N rows rather than
N rows, so the original direct indexing `AggOp.indices[E2V]` breaks.

Strategy
--------
For an interleaved DOF ordering (v0_x, v0_y, v1_x, v1_y, ...) with ndof
components per vertex, DOF row indices for vertex i are:
    ndof*i,  ndof*i+1,  ...,  ndof*i + ndof-1

We build a per-vertex aggregate ID array by taking the aggregate assigned to
the *first* DOF of each vertex and dividing by ndof.  This works because
standard_aggregation processes each DOF independently but preserves the
interleaved block structure, so DOF-aggregates naturally pair up.

For problems where that assumption does not hold (e.g. custom reorderings),
the function also accepts a pre-computed `vertex_agg` array directly.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection, PolyCollection
from scipy.sparse import csr_array, coo_array, triu


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dof_aggop_to_vertex_aggop(AggOp, ndof):
    """Collapse a DOF-level AggOp (ndof*N x Nagg) to a vertex-level array (N,).

    Parameters
    ----------
    AggOp : sparse matrix, shape (ndof*N, Nagg)
        Aggregation operator from standard_aggregation on the vector system.
    ndof : int
        Number of degrees of freedom per vertex.

    Returns
    -------
    vertex_agg : ndarray, shape (N,)
        Aggregate index for each vertex.  Vertices with no assignment get -1.
    """
    AggOp_csr = csr_array(AggOp)
    total_dofs = AggOp_csr.shape[0]
    if total_dofs % ndof != 0:
        raise ValueError(
            f"AggOp has {total_dofs} rows which is not divisible by ndof={ndof}."
        )
    N = total_dofs // ndof
    vertex_agg = np.full(N, -1, dtype=int)

    for i in range(N):
        first_dof_row = ndof * i
        cols = AggOp_csr.indices[
            AggOp_csr.indptr[first_dof_row] : AggOp_csr.indptr[first_dof_row + 1]
        ]
        if len(cols):
            # Divide by ndof to map DOF-aggregate index -> vertex-aggregate index
            vertex_agg[i] = cols[0] // ndof

    return vertex_agg


def _build_vertex_to_vertex(E2V):
    """Build upper-triangle vertex-to-vertex edge list from element table."""
    col = E2V.ravel()
    row = np.kron(
        np.arange(E2V.shape[0]),
        np.ones(E2V.shape[1], dtype=int),
    )
    data = np.ones(len(col))
    V2E = coo_array((data, (row, col)), shape=(E2V.shape[0], E2V.max() + 1))
    V2V = triu(V2E.T @ V2E, 1).tocoo()
    edges = np.column_stack((V2V.row, V2V.col))
    return edges, V2V


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def vis_vector_aggregate_groups(
    V,
    E2V,
    AggOp,
    ndof=1,
    vertex_agg=None,
    fname=None,
    ax=None,
    figsize=(8, 7),
    cmap="tab20",
    show_mesh=True,
    show_singleton_markers=True,
):
    """Visualise AMG aggregate groups for scalar or vector-valued problems.

    Produces a Matplotlib figure in which each aggregate is filled with a
    distinct colour.  Edges that connect two vertices in the *same* aggregate
    are drawn solidly; edges that cross aggregate boundaries are drawn faintly
    as background mesh.  Singleton aggregates are marked with a cross.

    Parameters
    ----------
    V : ndarray, shape (N, 2)
        Vertex coordinate array (2-D meshes only).
    E2V : ndarray, shape (Nel, Nelnodes), int
        Element-to-vertex connectivity (triangles or quads).
    AggOp : sparse matrix
        Aggregation operator.  For scalar problems: shape (N, Nagg).
        For vector problems: shape (ndof*N, Nagg_dof).
    ndof : int, optional
        Number of degrees of freedom per vertex.  Default 1 (scalar).
        When ndof > 1 the function automatically reduces AggOp to a
        per-vertex assignment before visualising.
    vertex_agg : ndarray of int, shape (N,), optional
        Pre-computed per-vertex aggregate IDs.  If supplied, AggOp and ndof
        are ignored for the aggregate assignment step.  Useful when the DOF
        ordering is non-standard.
    fname : str or None, optional
        If given, save the figure to this path (PNG, PDF, etc.).
    ax : matplotlib Axes or None, optional
        Axes to draw into.  A new figure is created when None.
    figsize : tuple, optional
        Figure size in inches, used only when ax is None.
    cmap : str, optional
        Matplotlib colourmap name for aggregate colours.  Default ``'tab20'``.
    show_mesh : bool, optional
        Draw faint background mesh edges.  Default True.
    show_singleton_markers : bool, optional
        Mark singleton aggregates with a cross symbol.  Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes

    Examples
    --------
    Scalar problem (original behaviour):

    >>> from pyamg.gallery import load_example
    >>> from pyamg.aggregation import standard_aggregation
    >>> data = load_example('unit_square')
    >>> AggOp = standard_aggregation(data['A'].tocsr())[0]
    >>> fig, ax = vis_vector_aggregate_groups(
    ...     V=data['vertices'], E2V=data['elements'],
    ...     AggOp=AggOp, ndof=1, fname='scalar_aggs.png')

    Vector problem (2 DOFs per node, e.g. 2-D elasticity):

    >>> from scipy.sparse import kron, eye
    >>> A2 = kron(data['A'].tocsr(), eye(2, format='csr'), format='csr')
    >>> AggOp2 = standard_aggregation(A2)[0]
    >>> fig, ax = vis_vector_aggregate_groups(
    ...     V=data['vertices'], E2V=data['elements'],
    ...     AggOp=AggOp2, ndof=2, fname='vector_aggs.png')
    """
    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("V must be an (N, 2) array of 2-D coordinates.")
    if not np.issubdtype(E2V.dtype, np.integer):
        raise ValueError("E2V must be an integer array.")
    if ndof < 1:
        raise ValueError("ndof must be >= 1.")

    N = V.shape[0]

    # ------------------------------------------------------------------
    # Step 1: resolve per-vertex aggregate IDs
    # ------------------------------------------------------------------
    if vertex_agg is not None:
        vertex_agg = np.asarray(vertex_agg, dtype=int).ravel()
        if len(vertex_agg) != N:
            raise ValueError(
                f"vertex_agg has length {len(vertex_agg)}, expected N={N}."
            )
    elif ndof == 1:
        # Scalar path — identical to the original code
        AggOp_csr = csr_array(AggOp)
        if AggOp_csr.shape[0] != N:
            raise ValueError(
                f"For ndof=1 AggOp must have {N} rows, got {AggOp_csr.shape[0]}."
            )
        # Handle zero rows (unassigned vertices) as in the original
        if len(AggOp_csr.indices) != N:
            vertex_agg = np.full(N, AggOp_csr.shape[1], dtype=int)  # sentinel
            for i in range(N):
                cols = AggOp_csr.indices[AggOp_csr.indptr[i] : AggOp_csr.indptr[i + 1]]
                if len(cols):
                    vertex_agg[i] = cols[0]
        else:
            vertex_agg = AggOp_csr.indices.copy()
    else:
        # Vector path
        expected_rows = ndof * N
        # print(expected_rows)
        # print(AggOp.shape[0])
        # exit()
        if AggOp.shape[0] != expected_rows:
            raise ValueError(
                f"For ndof={ndof} AggOp must have {expected_rows} rows "
                f"(ndof * N = {ndof} * {N}), got {AggOp.shape[0]}."
            )

        vertex_agg = _dof_aggop_to_vertex_aggop(AggOp, ndof)

    # Remap aggregate IDs to a contiguous range [0, Nagg)
    unique_aggs, vertex_agg_mapped = np.unique(vertex_agg, return_inverse=True)
    Nagg = len(unique_aggs)

    # ------------------------------------------------------------------
    # Step 2: classify elements as "full" (all vertices same aggregate)
    # ------------------------------------------------------------------
    # Remove elements that reference vertices outside our range
    valid_mask = E2V.max(axis=1) < N
    E2V_clean = E2V[valid_mask]

    ElementAggs = vertex_agg_mapped[E2V_clean]  # (Nel, Nelnodes)
    full_mask = (np.diff(ElementAggs, axis=1) == 0).all(axis=1)
    E2V_full = E2V_clean[full_mask]

    # ------------------------------------------------------------------
    # Step 3: classify edges as intra-aggregate
    # ------------------------------------------------------------------
    edges, V2V = _build_vertex_to_vertex(E2V_clean)
    intra_mask = vertex_agg_mapped[V2V.row] == vertex_agg_mapped[V2V.col]
    edges_intra = edges[intra_mask]
    edges_cross = edges[~intra_mask]

    # ------------------------------------------------------------------
    # Step 4: singleton aggregates
    # ------------------------------------------------------------------
    agg_sizes = np.bincount(vertex_agg_mapped, minlength=Nagg)
    singleton_agg_ids = np.where(agg_sizes == 1)[0]
    singleton_vertices = np.array(
        [np.where(vertex_agg_mapped == a)[0][0] for a in singleton_agg_ids]
    )

    # ------------------------------------------------------------------
    # Step 5: draw
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    cmap_obj = plt.get_cmap(cmap)
    colours = cmap_obj(np.linspace(0, 1, Nagg))

    # Background mesh
    if show_mesh and len(edges_cross):
        segs_cross = V[edges_cross]  # (M, 2, 2)
        lc_cross = LineCollection(
            segs_cross, linewidths=0.4, colors="lightgrey", zorder=1
        )
        ax.add_collection(lc_cross)

    # Filled triangles / quads coloured by aggregate
    if len(E2V_full):
        agg_ids_full = vertex_agg_mapped[E2V_full[:, 0]]
        face_colours = colours[agg_ids_full]
        poly = PolyCollection(
            V[E2V_full],
            facecolors=face_colours,
            edgecolors="none",
            alpha=0.55,
            zorder=2,
        )
        ax.add_collection(poly)

    # Intra-aggregate edges
    if len(edges_intra):
        agg_ids_edge = vertex_agg_mapped[edges_intra[:, 0]]
        edge_colours = colours[agg_ids_edge]
        segs_intra = V[edges_intra]
        lc_intra = LineCollection(
            segs_intra, linewidths=1.5, colors=edge_colours, zorder=3
        )
        ax.add_collection(lc_intra)

    # Singleton markers
    if show_singleton_markers and len(singleton_vertices):
        ax.scatter(
            V[singleton_vertices, 0],
            V[singleton_vertices, 1],
            marker="x",
            s=60,
            c="black",
            linewidths=1.5,
            zorder=5,
            label=f"Singletons ({len(singleton_vertices)})",
        )

    # Axis cosmetics
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ndof_label = f"ndof={ndof}" if ndof > 1 else "scalar"
    ax.set_title(
        f"AMG Aggregate Groups  ({ndof_label},  {Nagg} aggregates, {N} vertices)"
    )

    # ax.axvline(
    #     x=0.54, color="black", linewidth=1.2, linestyle="--", zorder=4, label="x = 0.5"
    # )
    # (
    #     plt.Line2D(
    #         [0], [0], color="black", linewidth=1.2, linestyle="--", label="x = 0.5"
    #     ),
    # )

    legend_handles = [
        mpatches.Patch(facecolor="C0", alpha=0.55, label="Full element (intra-agg)"),
        plt.Line2D([0], [0], color="C0", linewidth=1.5, label="Intra-agg edge"),
        plt.Line2D([0], [0], color="lightgrey", linewidth=0.8, label="Cross-agg edge"),
    ]
    if show_singleton_markers and len(singleton_vertices):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="x",
                color="black",
                linewidth=0,
                markersize=8,
                label=f"Singletons ({len(singleton_vertices)})",
            )
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    if fname is not None:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved to {fname}")

    return fig, ax


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    from pyamg.gallery import load_example
    from pyamg.aggregation import standard_aggregation
    from scipy.sparse import kron, eye

    data = load_example("unit_square")
    A = data["A"].tocsr()
    V = data["vertices"]
    E2V = data["elements"]

    # --- Scalar (1 DOF / vertex) ---
    AggOp1 = standard_aggregation(A)[0]
    fig1, ax1 = vis_vector_aggregate_groups(
        V=V,
        E2V=E2V,
        AggOp=AggOp1,
        ndof=1,
        fname="/mnt/user-data/outputs/agg_scalar.png",
    )
    plt.close(fig1)

    # --- Vector (2 DOFs / vertex, e.g. 2-D elasticity) ---
    ndof = 2
    A2 = kron(A, eye(ndof, format="csr"), format="csr")
    AggOp2 = standard_aggregation(A2)[0]
    fig2, ax2 = vis_vector_aggregate_groups(
        V=V,
        E2V=E2V,
        AggOp=AggOp2,
        ndof=ndof,
        fname="/mnt/user-data/outputs/agg_vector_2dof.png",
    )
    plt.close(fig2)

    # --- Vector (3 DOFs / vertex, e.g. 3 components) ---
    ndof = 3
    A3 = kron(A, eye(ndof, format="csr"), format="csr")
    AggOp3 = standard_aggregation(A3)[0]
    fig3, ax3 = vis_vector_aggregate_groups(
        V=V,
        E2V=E2V,
        AggOp=AggOp3,
        ndof=ndof,
        fname="/mnt/user-data/outputs/agg_vector_3dof.png",
    )
    plt.close(fig3)

    print("All three plots saved.")
