INPUTS

# Input the number of different kind of atoms.

atom1=153
atom2=54
atom3=342

# Number of orbitals in each atom in the same order above.

orbital1=5
orbital2=5
orbital3=3

# This is the maximum length upto which we will take the hopping.

r1st=3.21
r2nd=4.10

###### Band Numbers #######
c1=1432
v1=1431    ###### valence and conduction band indices
Nc=12      ###### Number of valence bands and number of conduction bands  
Nv=12

NOTE:

1. poscar.txt file contains the position of atoms in cartesian for the structure and poscar_ref.txt has the positions for the largest structure.
2. At first we have to run real_tb_main.py, it will create Eigen.dat file which contains the eigenvalues of the real space tight binding Hamiltonian. eigval.npy and eigvec.npy are binary files which contains all the required eigenvalues and eigenvectors in their binary form.
3. In the next step we have to run dis_idx.py keeping all the output files from the previous run. It will create distance_idx.npy file which contains all the distance information which is required to use the precomputed wannier integrals.
4. In the last step we have to run ci.py which will calculate exciton and trion Hamiltonian and finally exciton and trion binding energies. V_ee.npy and V_eh.npy contains all the ee and eh integrals in binary format.
5. For speedup the ci.py code is a parallel code on MPI so you have to run this code on cluster.
