from petsc4py import PETSc

# Set solver options


def set_solver_options_gamg():
    opts = PETSc.Options()

    # KSP options
    opts["ksp_type"] = "gmres"  # default: gmres  | Krylov solver type
    opts["ksp_gmres_restart"] = 10000
    opts["ksp_rtol"] = 1.0e-50  # default: 1e-5   | relative convergence tolerance
    opts["ksp_atol"] = 1.0e-8  # default: 1e-50  | absolute convergence tolerance
    opts["ksp_max_it"] = 2000  # default: 10000  | maximum number of iterations

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

    # opts["mg_levels_ksp_chebyshev_esteig_steps"] = 50

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
