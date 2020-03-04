
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProbPoisson ()
{
    for (int ilev = 0; ilev <= max_level; ++ilev)
    {
        // DEBUG
        const auto prob_lo = geom[ilev].ProbLoArray();
        const auto dx      = geom[ilev].CellSizeArray();
        amrex::Print() << "init lvl " << ilev << " : "
                       << "small end at " << prob_lo[0] << " " << prob_lo[1] << " " << prob_lo[2] << " "
                       << "spacings = " << dx[0] << " " << dx[1] << " " << dx[2] << "\n";

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        const amrex::IntVect smallEnd = geom[ilev].Domain().smallEnd();
        const amrex::IntVect bigEnd = geom[ilev].Domain().bigEnd();
        int Nx = bigEnd[0] - smallEnd[0] + 1;
        int Ny = bigEnd[1] - smallEnd[1] + 1;
        int Nz = bigEnd[2] - smallEnd[2] + 1;
        amrex::Print() << Nx << " " << Ny << " " << Nz << "\n";
        int N = Nx*Ny*Nz;

        std::string fpath = rhs_prefix + std::to_string(ilev) + std::string(".rhs");
        amrex::Print() << "reading " << fpath << "\n";
        std::ifstream f(fpath, std::ios::binary | std::ios::in);
        amrex::Vector<double> rhsvec(N);
        f.read(reinterpret_cast<char*>(&rhsvec[0]), N*sizeof(double));
        f.close();

        // DEBUG
        double divmin=99999, divmax=-99999;
        for (int i=0; i < N; ++i)
        {
            if (rhsvec[i] > divmax) divmax = rhsvec[i];
            if (rhsvec[i] < divmin) divmin = rhsvec[i];
        }
        amrex::Print() << "RHS : [ " << divmin << " ... " << divmax << "]\n";

        for (MFIter mfi(rhs[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto rhsfab = rhs[ilev].array(mfi);
//            auto exactfab = exact_solution[ilev].array(mfi);
//            amrex::ParallelFor(bx,
//            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
//            {
//                actual_init_poisson(i,j,k,rhsfab,exactfab,prob_lo,dx);
//            });

            const auto lo = lbound(bx);
            const auto hi = ubound(bx);
            // DEBUG:
            if (ilev == 0)
            {
                amrex::Print() << "lo " << lo.x << " " << lo.y << " " << lo.z << "\n";
                amrex::Print() << "hi " << hi.x << " " << hi.y << " " << hi.z << "\n";
            }
            
            int idx0;
            for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    idx0 = (k-smallEnd[2])*(Nx*Ny) + (j-smallEnd[1])*Nx;
                    for (int i = lo.x; i <= hi.x; ++i) {
                        // DEBUG:
                        amrex::Real x = dx[0] * (i + 0.5);
                        amrex::Real y = dx[1] * (j + 0.5);
                        amrex::Real z = dx[2] * (k + 0.5);
                        if (std::abs(rhsvec[idx0+i-smallEnd[0]]) > 0.1)
                        {
                            amrex::Print() << "RHS at " << x << ", " << y << ", " << z << " "
                                << "(" << i << ", " << j << ", " << k << ")"
                                << " = " << rhsvec[idx0+i-smallEnd[0]] << "\n";
                        }
                        rhsfab(i,j,k) = rhsvec[idx0+i-smallEnd[0]];
                    }
                }
            }

        }

        solution[ilev].setVal(0.0);
    }
}
