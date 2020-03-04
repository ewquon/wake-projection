#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>

#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

class MyTest
{
public:

    MyTest ();

    void solve ();
    void writePlotfile () const;

public: // make these public for cuda
    void initProbPoisson ();
    void initProbABecLaplacian ();
    void initProbABecLaplacianInhomNeumann ();

private:

    void readParameters ();
    void initData ();

//    int max_level = 2;
    int max_level = 0; // DEBUG
    int ref_ratio = 2;
    int max_grid_size = 64;
    
    // EWQ:
    amrex::Real x0,y0,z0,x1,y1,z1; // actual bounding box
    amrex::Real xmin = -10.0; // target streamwise extent, in rotor radii R
    amrex::Real xmax =  10.0; // target streamwise extent, in rotor radii R
    amrex::Real buffer = 10.0; // target lateral/vertical extent from rotor edge, in rotor radii R
//    amrex::Real spacing = 0.1;
    amrex::Real spacing = 0.4; // DEBUG
    amrex::Real zhub = 1.5;
    bool ground_effect = false;
    std::string rhs_prefix = "level"; 
    
    bool composite_solve = true;

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 100;
    int max_fmg_iter = 0;
    int linop_maxorder = 2;
    bool agglomeration = true;
    bool consolidation = true;
    int max_coarsening_level = 30;
    bool use_hypre = false;
    bool use_petsc = false;
    
#ifdef AMREX_USE_HYPRE
    int hypre_interface_i = 1;  // 1. structed, 2. semi-structed, 3. ij
    amrex::Hypre::Interface hypre_interface = amrex::Hypre::Interface::structed;
#endif

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;

    amrex::Vector<amrex::MultiFab> solution;
    amrex::Vector<amrex::MultiFab> rhs;
};

#endif
