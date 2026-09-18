INPUTS

Nx=84
Ny=84
Nz=576

N=Nx*Ny*Nz

a1=np.array([9.3375003,0.0000000,0.0000000])
a2=np.array([-4.6687500,8.0865124,0.0000000])
a3=np.array([0.0000000,0.0000000,89.8437500])

global_origin= np.array([-1.631250,-2.825408,-30.156250])
Mo_pos=np.array([-0.0000016,1.8186543,3.0750000])
S1_pos=np.array([1.5750016,0.9093258,4.6503026])
S2_pos=np.array([1.5750016,0.9093258,1.4996976])

###### These Inputs are directly taken from WANNIER90 Output files

Nr=30
Ntheta=10
Nphi=10

###### These are the grids along spherical directions.

file1='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00001.xsf'
file2='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00002.xsf'
file3='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00003.xsf'
file4='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00004.xsf'
file5='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00005.xsf'
file6='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00006.xsf'
file7='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00007.xsf'
file8='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00008.xsf'
file9='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00009.xsf'
file10='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00010.xsf'
file11='/scratch/priya.snbose3/Koustav/python_codes/wannier_hse06/wann_xsf/wannier90_00011.xsf'

###### These are the locations of all the .xsf files which contains the information of the wannier functions. As our model has total 11 orbitals that's why there are 11 .xsf files. 

pos_index=np.arange(10,30)

###### This is the place where we have to put the indices of the distances at a time for which we calculate the wannier integral. 

After Calculating all integrals we can get all coulomb_integral_{ind}.dat files. Then we have to keep all files in a folder and pass that folder to the real space tight binding ci calculation step.


