#include "MyTest.H"

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();
    initData();
}

void
MyTest::solve ()
{
    // Based on solvePoisson() and solveABecLaplacian() (EWQ)

    LPInfo info;
    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setMaxCoarseningLevel(max_coarsening_level);

    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;

    const int nlevels = geom.size();

    if (composite_solve)
    {

        MLPoisson mlpoisson(geom, grids, dmap, info);

        mlpoisson.setMaxOrder(linop_maxorder);

        // This is a 3d problem with homogeneous Neumann BC
        mlpoisson.setDomainBC({AMREX_D_DECL(LinOpBCType::Neumann,
                                            LinOpBCType::Neumann,
                                            LinOpBCType::Neumann)},
                              {AMREX_D_DECL(LinOpBCType::Neumann,
                                            LinOpBCType::Neumann,
                                            LinOpBCType::Neumann)}); 
        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            // for problem with pure homogeneous Neumann BC, we could pass a nullptr
            mlpoisson.setLevelBC(ilev, nullptr);
        }

        MLMG mlmg(mlpoisson);
        mlmg.setMaxIter(max_iter);
        mlmg.setMaxFmgIter(max_fmg_iter);
        mlmg.setVerbose(verbose);
        mlmg.setBottomVerbose(bottom_verbose);
#ifdef AMREX_USE_HYPRE
        if (use_hypre) {
            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
            mlmg.setHypreInterface(hypre_interface);
        }
#endif
#ifdef AMREX_USE_PETSC
        if (use_petsc) {
            mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
        }
#endif

        mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), tol_rel, tol_abs);
    }
    else
    {
        for (int ilev = 0; ilev < nlevels; ++ilev)
        {
            MLPoisson mlpoisson({geom[ilev]}, {grids[ilev]}, {dmap[ilev]}, info);
            
            mlpoisson.setMaxOrder(linop_maxorder);
            
            // This is a 3d problem with homogeneous Neumann BC
            mlpoisson.setDomainBC({AMREX_D_DECL(LinOpBCType::Neumann,
                                                LinOpBCType::Neumann,
                                                LinOpBCType::Neumann)},
                                  {AMREX_D_DECL(LinOpBCType::Neumann,
                                                LinOpBCType::Neumann,
                                                LinOpBCType::Neumann)});
            
            if (ilev > 0) {
                mlpoisson.setCoarseFineBC(&solution[ilev-1], ref_ratio);
            }

            // for problem with pure homogeneous Neumann BC, we could pass a nullptr
            mlpoisson.setLevelBC(0, nullptr);

            MLMG mlmg(mlpoisson);
            mlmg.setMaxIter(max_iter);
            mlmg.setMaxFmgIter(max_fmg_iter);
            mlmg.setVerbose(verbose);
            mlmg.setBottomVerbose(bottom_verbose);
#ifdef AMREX_USE_HYPRE
            if (use_hypre) {
                mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
                mlmg.setHypreInterface(hypre_interface);
            }
#endif
#ifdef AMREX_USE_PETSC
            if (use_petsc) {
                mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
            }
#endif
            
            mlmg.solve({&solution[ilev]}, {&rhs[ilev]}, tol_rel, tol_abs);            
        }
    }

    // Since this problem has Neumann BC, solution + constant is also a
    // solution...
    //
}


void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("max_level", max_level);
    pp.query("spacing", spacing);
    pp.query("ref_ratio", ref_ratio);
    pp.query("max_grid_size", max_grid_size);

    pp.query("xmin", xmin);
    pp.query("xmax", xmax);
    pp.query("buffer", buffer);
    pp.query("zhub", zhub);
    pp.query("ground_effect", ground_effect);
    pp.query("rhs_prefix", rhs_prefix);
    pp.query("output", output);

    pp.query("composite_solve", composite_solve);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("linop_maxorder", linop_maxorder);
    pp.query("agglomeration", agglomeration);
    pp.query("consolidation", consolidation);
    pp.query("max_coarsening_level", max_coarsening_level);

#ifdef AMREX_USE_HYPRE
    pp.query("use_hypre", use_hypre);
    pp.query("hypre_interface", hypre_interface_i);
    if (hypre_interface_i == 1) {
        hypre_interface = Hypre::Interface::structed;
    } else if (hypre_interface_i == 2) {
        hypre_interface = Hypre::Interface::semi_structed;
    } else {
        hypre_interface = Hypre::Interface::ij;
    }
#endif
#ifdef AMREX_USE_PETSC
    pp.query("use_petsc", use_petsc);
#endif
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!(use_hypre && use_petsc),
                                     "use_hypre & use_petsc cannot be both true");
}

void
MyTest::initData ()
{
    int nlevels = max_level + 1;
    geom.resize(nlevels);
    grids.resize(nlevels);
    dmap.resize(nlevels);

    solution.resize(nlevels);
    rhs.resize(nlevels);

    // Calculate bounds (EWQ)
    x0 = xmin;
    x1 = xmax;
    y0 = -1 - buffer;
    y1 =  1 + buffer;
    if (ground_effect)
    {
        z0 = 0;
        z1 = zhub + 1 + buffer;
    }
    else
    {
        z0 = zhub - 1 - buffer;
        z1 = zhub + 1 + buffer;
    }
    amrex::Print() << "Target bounds: "
        << "(" << x0 << ", " << y0 << ", " << z0 << ") "
        << "(" << x1 << ", " << y1 << ", " << z1 << ") "
        << "\n";

    // Calculate number of cells (EWQ)
    // - ensure multiple of max_grid_size
    amrex::Real spacing0 = spacing * std::pow(2, max_level);
    int nxtot = std::ceil((x1-x0) / spacing0);
    int nytot = std::ceil((y1-y0) / spacing0);
    int nztot = std::ceil((z1-z0) / spacing0);
    amrex::Print() << "- adjusted grid size (for max spacing=" << spacing0 << "): " << nxtot << " " << nytot << " " << nztot << "\n";
    int nxdiff = std::ceil((float)nxtot / max_grid_size) * max_grid_size - nxtot; // extra cells needed?
    int nydiff = std::ceil((float)nytot / max_grid_size) * max_grid_size - nytot;
    int nzdiff = std::ceil((float)nztot / max_grid_size) * max_grid_size - nztot;
    amrex::Print() << "- grid expansion (max_grid_size=" << max_grid_size << "): " << nxdiff << " " << nydiff << " " << nzdiff << "\n";

    // Calculate actual bounds (EWQ)
    x0 -= nxdiff/2 * spacing0;
    x1 += nxdiff/2 * spacing0;
    y0 -= nydiff/2 * spacing0;
    y1 += nydiff/2 * spacing0;
    if (ground_effect)
    {
        z1 += nzdiff * spacing0;
    }
    else
    {
        z0 -= nzdiff/2 * spacing0;
        z1 += nzdiff/2 * spacing0;
    }
    amrex::Print() << "Actual bounds (init spacing=" << spacing0 << "): "
        << "(" << x0 << ", " << y0 << ", " << z0 << ") "
        << "(" << x1 << ", " << y1 << ", " << z1 << ") "
        << "\n";

    nxtot = (x1 - x0) / spacing0;
    nytot = (y1 - y0) / spacing0;
    nztot = (z1 - z0) / spacing0;

    RealBox rb({AMREX_D_DECL(x0,y0,z0)}, {AMREX_D_DECL(x1,y1,z1)}); // EWQ
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    // EWQ:
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)},
                IntVect{AMREX_D_DECL(nxtot-1,nytot-1,nztot-1)});  // cell-centered, by default
    amrex::Print() << "domain0 : " << domain0 << "\n"; // domain0 : ((0,0,0) (127,127,127) (0,0,0))

    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        amrex::Print() << "lvl " << ilev << " : " << geom[ilev] << "\n";
        domain.refine(ref_ratio);
    }

    domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        grids[ilev].define(domain);
        grids[ilev].maxSize(max_grid_size);
//        domain.grow(-n_cell/4);   // fine level cover the middle of the coarse domain
        domain.refine(ref_ratio); 
    }

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        // a: const BoxArray & bxs
        // b: const DistributionMapping & dm
        // c: int ncomp (# components)
        // d: int ngrow (# ghost cells)
        //                          a            b           c  d
        solution      [ilev].define(grids[ilev], dmap[ilev], 1, 1);
        rhs           [ilev].define(grids[ilev], dmap[ilev], 1, 0);
    }

    initProbPoisson();
}

