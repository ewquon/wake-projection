
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProbPoisson ()
{
    for (int ilev = 0; ilev <= max_level; ++ilev)
    {
        const auto prob_lo = geom[ilev].ProbLoArray();
        const auto dx      = geom[ilev].CellSizeArray();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(rhs[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto rhsfab = rhs[ilev].array(mfi);
            auto exactfab = exact_solution[ilev].array(mfi);
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                actual_init_poisson(i,j,k,rhsfab,exactfab,prob_lo,dx);
            });
        }

        solution[ilev].setVal(0.0);
    }
}
