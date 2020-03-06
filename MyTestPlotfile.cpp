
#include "MyTest.H"
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void
MyTest::writePlotfile () const
{
//    ParmParse pp;
//    bool gpu_regtest = false;
//#ifdef AMREX_USE_GPU
//    pp.query("gpu_regtest", gpu_regtest);
//#endif

    const int nlevels = max_level+1;
    Vector<MultiFab> plotmf(nlevels);

    const int ncomp = 2;
    Vector<std::string> varname = {"solution", "rhs"};

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        plotmf[ilev].define(grids[ilev], dmap[ilev], ncomp, 0);
        MultiFab::Copy(plotmf[ilev], solution      [ilev], 0, 0, 1, 0);
        MultiFab::Copy(plotmf[ilev], rhs           [ilev], 0, 1, 1, 0);
    }
    WriteMultiLevelPlotfile("plot", nlevels, amrex::GetVecOfConstPtrs(plotmf),
                            varname, geom, 0.0, Vector<int>(nlevels, 0),
                            Vector<IntVect>(nlevels, IntVect{ref_ratio}));
}

