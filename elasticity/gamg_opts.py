from petsc4py import PETSc

# Set solver options


def set_solver_options_gamg():
    opts = PETSc.Options()

    # KSP options
    opts["ksp_type"] = "cg"  # default: gmres  | Krylov solver type
    opts["ksp_gmres_restart"] = 10000
    opts["ksp_rtol"] = 1.0e-50  # default: 1e-5   | relative convergence tolerance
    opts["ksp_atol"] = 1.0e-8  # default: 1e-50  | absolute convergence tolerance
    opts["ksp_max_it"] = 200  # default: 10000  | maximum number of iterations

    # GAMG general options
    opts["pc_type"] = "gamg"

    # # opts["pc_svd_monitor"] = True

    # # opts["ksp_monitor_singular_value"] = True
    opts["pc_gamg_type"] = (
        "agg"  # default: agg        | one of: agg, geo, classical (only agg supported)
    )
    # # opts["pc_gamg_aggressive_coarsening"] = True
    # opts["pc_gamg_repartition"] = (
    #     False  # default: false      | repartition DOFs across coarse grids as they are determined
    # )
    # opts["pc_gamg_asm_use_agg"] = (
    #     False  # default: false      | use aggregates to define PCASM smoother subdomains on each level
    # )
    # opts["pc_gamg_process_eq_limit"] = (
    #     50  # default: 50        | target number of equations per MPI rank on coarse grids
    # )
    # opts["pc_gamg_coarse_eq_limit"] = (
    #     50  # default: 50        | maximum number of equations on the coarsest grid
    # )
    # opts["pc_gamg_reuse_interpolation"] = (
    #     True  # default: true       | reuse previously computed interpolations when rebuilding AMG
    # )
    opts["pc_gamg_threshold"] = (
        0.111  # 0.111  # 0.1110001  # default: -1        | remove small graph values before aggregating (< 0 = no filtering)
    )
    # opts["pc_gamg_threshold_scale"] = (
    #     1  # 0.9  # default: 1         | scaling of threshold on each coarser grid
    # )

    # # GAMG aggregation options (used when pc_gamg_type = "agg")
    # opts["pc_gamg_agg_nsmooths"] = (
    #     1  # default: 1         | number of smoothing steps for smooth aggregation prolongation
    # )
    # opts["pc_gamg_aggressive_coarsening"] = (
    #     1  # default: 1         | number of aggressive coarsening (MIS-2) levels from finest
    # )
    # opts["pc_gamg_aggressive_square_graph"] = (
    #     True  # default: true       | use A^T A for coarsening, otherwise MIS-k (k=2) is used
    # )
    # opts["pc_gamg_mis_k_minimum_degree_ordering"] = (
    #     False  # default: false | use minimum degree ordering in greedy MIS algorithm
    # )
    # opts["pc_gamg_asm_hem_aggs"] = (
    #     0  # default: 0         | number of HEM aggregation steps for PCASM smoother
    # )
    # opts["pc_gamg_aggressive_mis_k"] = (
    #     2  # default: 2         | distance k in MIS coarsening (> 2 is aggressive)
    # )

    # # Multigrid (PCMG) options — shared between PCMG and PCGAMG
    # # opts["pc_mg_levels"] = 4                      # default: automatic  | number of levels including finest; GAMG sets this via internal heuristic
    # opts["pc_mg_cycle_type"] = "v"  # default: v          | cycle type, one of: v, w
    # opts["pc_mg_type"] = (
    #     "multiplicative"  # default: multiplicative | one of: additive, multiplicative, full, kaskade
    # )
    # opts["pc_mg_log"] = (
    #     False  # default: false      | log time spent on each level of the solver
    # )
    # opts["pc_mg_distinct_smoothup"] = (
    #     False  # default: false      | configure up (post) and down (pre) smoothers separately with different option prefixes
    # )
    # opts["pc_mg_galerkin"] = (
    #     "both"  # default: both       | use Galerkin process to compute coarser operators A_c = R A R^T; one of: both, pmat, mat, none
    # )

    # opts["pc_mg_dump_matlab"] = (
    #     False  # default: false      | dump matrices and restriction/interpolation to PETSCVIEWERSOCKET for MATLAB
    # )
    # opts["pc_mg_dump_binary"] = (
    #     False  # default: false      | dump matrices and restriction/interpolation to binary file 'binaryoutput'
    # )

    # Level smoother options (applied to all levels except coarsest)
    opts["mg_levels_ksp_type"] = (
        "chebyshev"  # default: chebyshev  | Krylov smoother type on all levels
    )
    opts["mg_levels_pc_type"] = (
        "sor"  # default: sor        | preconditioner type for smoother on all levels
    )
    opts["mg_levels_ksp_max_it"] = (
        2  # default: 2         | number of smoother iterations on each level
    )

    opts["pc_mg_multiplicative_cycles"] = (
        1  # default: 1         | number of multigrid cycles to use as the preconditioner
    )

    # opts["mg_levels_ksp_chebyshev_esteig_steps"] = 1000

    # # Coarse grid solver options
    # # opts["mg_coarse_ksp_type"] = "preonly"        # default: preonly    | Krylov solver type on coarsest grid
    # # opts["mg_coarse_pc_type"] = "lu"              # default: lu         | preconditioner type on coarsest grid
    # opts["ksp_reuse_preconditioner"] = True


def set_solver_options_icc():
    opts = PETSc.Options()

    # KSP options
    opts["ksp_type"] = "gmres"  # default: gmres  | Krylov solver type
    opts["ksp_rtol"] = 1.0e-50  # default: 1e-5   | relative convergence tolerance
    opts["ksp_atol"] = 1.0e-8  # default: 1e-50  | absolute convergence tolerance
    opts["ksp_max_it"] = 100  # default: 10000  | maximum number of iterations

    # GAMG general options
    opts["pc_type"] = "hypre"
    opts["pc_hypre_type"] = "euclid"
    opts["pc_hypre_euclid_level"] = 70  # ILU(2)
    # opts["pc_factor_levels"] = 10
    # opts["pc_factor_mat_ordering_type"] = "nd"
    # default: lu         | preconditioner type on coarsest grid
    opts["ksp_reuse_preconditioner"] = True


# ---------------------------------------------------------------------------
# Option A  (recommended): HYPRE BoomerAMG
# ---------------------------------------------------------------------------
def set_solver_options_boomeramg():
    """
    CG + BoomerAMG with 'ext+i' interpolation.

    This is the most robust option for scalar or vector problems with
    large, heterogeneous coefficients.  The 'ext+i' interpolation
    (interpolation_type=6) uses distance-2 stencils and is specifically
    designed to be robust to large jumps.

    Iteration count is typically O(1) w.r.t. mesh refinement AND
    w.r.t. the contrast ratio.

    Requires: PETSc built with --download-hypre (standard in most packages).
    """
    opts = PETSc.Options()

    # --- Outer Krylov ---
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-6
    opts["ksp_atol"] = 1.0e-12
    opts["ksp_max_it"] = 500
    opts["ksp_norm_type"] = "unpreconditioned"  # easier to interpret

    # --- Preconditioner: HYPRE BoomerAMG ---
    opts["pc_type"] = "hypre"
    opts["pc_hypre_type"] = "boomeramg"

    # Coarsening: HMIS (distance-2 MIS) is more robust than RS for jumps
    opts["pc_hypre_boomeramg_coarsen_type"] = "HMIS"

    # Interpolation: 'ext+i' (type 6) — robust to large jumps
    # Alternative: 'FF1' (type 18) is lighter; 'extended' (type 14) is heavier
    opts["pc_hypre_boomeramg_interp_type"] = "ext+i"

    # Truncate interpolation to at most P_max entries per row
    # (controls operator complexity; 4–6 is a good balance)
    opts["pc_hypre_boomeramg_P_max"] = 4

    # Smoother: l1-scaled Jacobi — cheap and robust to heterogeneity
    # (standard Gauss-Seidel can stall on large contrasts)
    opts["pc_hypre_boomeramg_relax_type_all"] = "l1scaled-Jacobi"
    opts["pc_hypre_boomeramg_relax_sweeps_all"] = 1

    # Aggressive coarsening on the first level (reduces complexity)
    opts["pc_hypre_boomeramg_agg_nl"] = 1
    opts["pc_hypre_boomeramg_agg_num_paths"] = 2

    # Strength threshold — lower = stronger connections kept = more robust
    # For extreme contrasts set to 0.0 (keep everything) or 0.25
    opts["pc_hypre_boomeramg_strong_threshold"] = 0.25

    # Coarse grid solver: direct (for small coarse problems)
    opts["pc_hypre_boomeramg_max_coarse_size"] = 100


# ---------------------------------------------------------------------------
# Option D: BDDC  (scalable, parameter-robust, requires PETSc BDDC support)
# ---------------------------------------------------------------------------
def set_solver_options_bddc():
    """
    CG + BDDC.  BDDC is provably robust to coefficient jumps that are
    constant on each subdomain.  For the checkerboard case (one material
    per element or per coarse cell) it provides near-optimal iteration
    counts independent of the contrast ratio.

    Requires: PETSc built with MUMPS or SuperLU_dist for the subdomain and
    coarse solvers.  Works best with 1 subdomain per MPI rank.

    Usage note: you must supply dofmap/topology information via IS objects
    when building the PC — see PETSc PCBDDC documentation for details.
    This function only sets the option strings; the IS setup must be done
    in the calling code.
    """
    opts = PETSc.Options()

    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-6
    opts["ksp_atol"] = 1.0e-12
    opts["ksp_max_it"] = 200

    opts["pc_type"] = "bddc"

    # Subdomain solver: direct
    opts["pc_bddc_dirichlet_pc_type"] = "lu"
    opts["pc_bddc_neumann_pc_type"] = "lu"

    # Coarse solver
    opts["pc_bddc_coarse_pc_type"] = "lu"
    opts["pc_bddc_use_deluxe_scaling"] = True  # improves robustness to jumps
    opts["pc_bddc_symmetric"] = True


# ---------------------------------------------------------------------------
# Option B: PETSc GAMG with l1-Jacobi smoother  (no HYPRE dependency)
# ---------------------------------------------------------------------------
def set_solver_options_gamg_robust():
    """
    CG + GAMG-agg with l1-Jacobi smoother.

    Key changes vs the original gamg_opts.py:
      1. Outer solver: CG  (SPD system — no need for GMRES)
      2. Smoother: 'ksp_type chebyshev' + 'pc_type jacobi' replaced by
         a single l1-Jacobi application (more robust to heterogeneity).
      3. threshold = 0  →  do not drop any connections before aggregating.
         Dropping connections on a heterogeneous graph is the main reason
         for GAMG divergence on large-contrast problems.
      4. nsmooths = 1  (smooth aggregation prolongation steps).

    This will not match BoomerAMG quality for 18-order contrasts but is
    a significant improvement over the original settings and requires no
    additional libraries.
    """
    opts = PETSc.Options()

    # --- Outer Krylov ---
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1.0e-6
    opts["ksp_atol"] = 1.0e-12
    opts["ksp_max_it"] = 500

    # --- Preconditioner: GAMG ---
    opts["pc_type"] = "gamg"
    opts["pc_gamg_type"] = "agg"

    # Do NOT drop any graph edges — crucial for large-contrast problems
    opts["pc_gamg_threshold"] = 0.111
    opts["pc_gamg_threshold_scale"] = 1.0

    # Smooth aggregation prolongation: 1 smoothing step is standard
    opts["pc_gamg_agg_nsmooths"] = 1

    # Reuse interpolation when re-assembling (saves cost in nonlinear loops)
    opts["pc_gamg_reuse_interpolation"] = True

    # --- Level smoother: l1-Jacobi ---
    # 'ksp_type richardson' + 'pc_type jacobi' with l1 scaling is robust
    opts["mg_levels_ksp_type"] = "richardson"
    opts["mg_levels_ksp_richardson_scale"] = 1.0
    opts["mg_levels_pc_type"] = "jacobi"
    opts["mg_levels_pc_jacobi_type"] = "rowl1"  # l1-scaled diagonal
    opts["mg_levels_ksp_max_it"] = 2

    opts["pc_mg_cycle_type"] = "v"
    opts["pc_mg_multiplicative_cycles"] = 1
