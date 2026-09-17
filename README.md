Required Python Packages,
numpy
scipy
matplotlib.pyplot

Inputs for TB_fitting_code

Ef=-3.3396
NBANDS=192  

#### Ef and NBANDS are taken from the OUTCAR

near_1st=3.19
near_2nd=4.00  

#### These values are given from POSCAR file

E1=-2
E2=2  

##### E1 and E2 are the energy window where we want to fit the bands

total_type=2  

###### This is the number of types of atoms

atom_no=np.array([1,2])  

###### This is the number of atoms corresponding to every type

N_vbm=1
N_cbm=2  

###### These are the number of valence and conduction bands to fit.

near=np.array([0.01,near_1st,near_2nd])

gamma=[0.0,0.0,0.0]
X=[0.33,0.33,0.0]
M=[0.5,0.0000000000,0.0000000000]
points=40
s_points=[gamma,X,M,gamma]     

###### These are the symmetry points and points=40 means there are 40 points between each symmetry paths.

SK_params_1st=np.array([[-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-0.67,0.83,-0.31,-100.0,-100.0,100.0,-100.0],\
        [-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,3.01,-1.41],\
        [-100.0,100.0,-100.0,1.32,-0.00,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,100.0,-100.0]])

SK_params_2nd=np.array([[-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,100.0,-100.0],\
        [-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,0.90,-0.37],\
        [-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,100.0,-100.0]])

###### These are the irreducible Slater Koster initial guess parameters. 100.0 means our model don't required those parameters.

E0_1=np.array([100.0,100.0,100.0,100.0,-3.82,-3.00,100.0,100.0,-5.10])
E0_2=np.array([100.0,-5.45,100.0,-7.13,100.0,100.0,100.0,100.0,100.0])

###### These are the onsite energies. 

