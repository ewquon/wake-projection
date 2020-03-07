
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

void
MyTest::writeFineLevelData () const
{
    const amrex::IntVect lower = geom[max_level].Domain().smallEnd();
    const amrex::IntVect upper = geom[max_level].Domain().bigEnd();
    const int Nx = upper[0] - lower[0] + 1;
    const int Ny = upper[1] - lower[1] + 1;
    const int Nz = upper[2] - lower[2] + 1;
    const int N = Nx*Ny*Nz;
    amrex::Vector<double> solnvec(N);

    // Get solution at all cell centers (i,j,k)
    for (MFIter mfi(solution[max_level], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // Get current box chunk
        const Box& bx = mfi.tilebox();
        auto solnfab = solution[max_level].array(mfi);
        amrex::Dim3 lo = lbound(bx);
        amrex::Dim3 hi = ubound(bx);

        for (int k = lo.z; k <= hi.z; ++k)
        {
            for (int j = lo.y; j <= hi.y; ++j)
            {
                for (int i = lo.x; i <= hi.x; ++i)
                {
                    solnvec[(k-lower[2])*(Nx*Ny) + (j-lower[1])*Nx + i-lower[0]] = solnfab(i,j,k);
                }
            }
        }
    }

    // now write the whole vector at once
    std::ofstream f(output, std::ios::binary | std::ios::out);
    f.write(reinterpret_cast<char*>(&solnvec[0]), N*sizeof(double));
    f.close();
    amrex::Print() << "Wrote output to " << output << "\n";
}
