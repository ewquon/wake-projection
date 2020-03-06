
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProbPoisson ()
{
    for (int ilev = 0; ilev <= max_level; ++ilev)
    {
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        // get number of cells in current level
        const amrex::IntVect lower = geom[ilev].Domain().smallEnd();
        const amrex::IntVect upper = geom[ilev].Domain().bigEnd();
        const int Nx = upper[0] - lower[0] + 1;
        const int Ny = upper[1] - lower[1] + 1;
        const int Nz = upper[2] - lower[2] + 1;
        const int N = Nx*Ny*Nz;

        // DEBUG
        {
            const auto prob_lo = geom[ilev].ProbLoArray();
            const auto dx      = geom[ilev].CellSizeArray();
            amrex::Print() << "init lvl " << ilev << " : "
                           << "lower bound = " << prob_lo[0] << " " << prob_lo[1] << " " << prob_lo[2] << ", "
                           << "spacings = " << dx[0] << " " << dx[1] << " " << dx[2] << ", "
                           << "N = " << Nx << " " << Ny << " " << Nz << "\n";
        }

        // read all RHS values
        std::string fpath = rhs_prefix + std::to_string(ilev) + std::string(".rhs");
        amrex::Print() << "reading " << fpath << "\n";
        std::ifstream f(fpath, std::ios::binary | std::ios::in);
        amrex::Vector<double> rhsvec(N);
        f.read(reinterpret_cast<char*>(&rhsvec[0]), N*sizeof(double));
        f.close();

        // DEBUG
        {
            double divmin=9e9, divmax=-9e9;
            for (int i=0; i < N; ++i)
            {
                if (rhsvec[i] > divmax) divmax = rhsvec[i];
                if (rhsvec[i] < divmin) divmin = rhsvec[i];
            }
            amrex::Print() << "RHS : [ " << divmin << " ... " << divmax << "]\n";
        }

        // Set all cell centers (i,j,k)
        for (MFIter mfi(rhs[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            // Get current box chunk
            const Box& bx = mfi.tilebox();
            auto rhsfab = rhs[ilev].array(mfi);
            amrex::Dim3 lo = lbound(bx);
            amrex::Dim3 hi = ubound(bx);

            for (int k = lo.z; k <= hi.z; ++k)
            {
                for (int j = lo.y; j <= hi.y; ++j)
                {
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        rhsfab(i,j,k) = 
                            rhsvec[(k-lower[2])*(Nx*Ny) + (j-lower[1])*Nx + i-lower[0]];
                    }
                }
            }
        }

        solution[ilev].setVal(0.0);
    }
}

