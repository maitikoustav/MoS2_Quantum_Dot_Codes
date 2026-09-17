# MoS2_Quantum_Dot_Codes
Here I have uploaded all my latest codes to calculate real space tight binding model for triangular MoS2 quantum dot. After that all the codes associated with this to calculate exciton and trion binding energies.

Step 1.
This is the first step where we have to do the DFT calculation for the monolayer MoS2 for calculating the band structure. All the input files for VASP to calculate HSE06 band structure is attached with.

Step 2.
This is the step where we fit the DFT band structure with a Slater Koster tight binding model. Here we use our code tb_fitting.py.

Step 4.
After getting the TB parameters we use our real_tb_main.py code to calculate the real space tight binding model for triangular quantum dot and get the energy states and corresponding eigen functions.

Step 5.
In this step we calculate the wannier integrals using our code orbital_spread_new.py for every possible distances using the wannier functions from WANNIER90.

Step 6.
Then we calculate the configuration interaction matrix elements using our ci.py code and calculate the exciton and trion binding energies.


