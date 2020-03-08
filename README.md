# wake-projection
Wind turbine array blockage model based on AMReX

Starting with a modeled wake velocity field, compute a pseudo-pressure projection to enforce continuity in the remainder of the flow field. A corrected velocity field comes from solution of the Poisson equation with Neumann boundary conditions. 

References:
- Branlard, Quon, Forsting, King, and Moriarty, "Blockage effects in a wind farm: comparison of different engineering models", TORQUE 2020.
