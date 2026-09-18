import numpy as np
#import matplotlib.pyplot as plt
from numpy.linalg import eig
#from matplotlib import cm
import math
import cmath
from numpy import linalg
from scipy.interpolate import griddata
from scipy.constants import e, epsilon_0
from scipy.special import struve, y0
#from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
import gc
import scipy

# Input the number of diffrent kind of atoms.

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

Nc=12
Nv=12
norb=11

#----------------------------------------------------------INPUT END---------------------------------------------------------------

total=atom1+atom2+atom3
dim=atom1*orbital1+atom2*orbital2+atom3*orbital3


limit1=atom1
limit2=atom1+atom2
limit3=atom1+atom2+atom3

# Here we read the input poscar data file.

poscar=[]
file_name='poscar.txt'
with open(file_name,'r') as file:
    for line in file:
        columns=list(map(float,line.strip().split()))
        poscar.append(columns)

# Here we calculate the eigenvalues.
index=0
e_vec_atom=[]
e_vec_orbital=[]
orbital_index=[]
orb_ind=1
for i in range(total):
    if(i<limit1):
        e_vec_atom.append(i)
        e_vec_orbital.append('Mo_dxy')
        orbital_index.append(0)
        e_vec_atom.append(i)
        e_vec_orbital.append('Mo_dyz')
        orbital_index.append(1)
        e_vec_atom.append(i)
        e_vec_orbital.append('Mo_dzx')
        orbital_index.append(2)
        e_vec_atom.append(i)
        e_vec_orbital.append('Mo_dx2y2')
        orbital_index.append(3)
        e_vec_atom.append(i)
        e_vec_orbital.append('Mo_dz2')
        orbital_index.append(4)
    elif(i>=limit1 and i<limit2):
        e_vec_atom.append(i)
        e_vec_orbital.append('W_dxy')
        orbital_index.append(0)
        e_vec_atom.append(i)
        e_vec_orbital.append('W_dyz')
        orbital_index.append(1)
        e_vec_atom.append(i)
        e_vec_orbital.append('W_dzx')
        orbital_index.append(2)
        e_vec_atom.append(i)
        e_vec_orbital.append('W_dx2y2')
        orbital_index.append(3)
        e_vec_atom.append(i)
        e_vec_orbital.append('W_dz2')
        orbital_index.append(4)
    elif(i>=limit2 and i<limit3):
        if(orb_ind%2!=0):
            e_vec_atom.append(i)
            e_vec_orbital.append('S_x_upper')
            orbital_index.append(5)
            e_vec_atom.append(i)
            e_vec_orbital.append('S_y_upper')
            orbital_index.append(6)
            e_vec_atom.append(i)
            e_vec_orbital.append('S_z_upper')
            orbital_index.append(7)
        else:
            e_vec_atom.append(i)
            e_vec_orbital.append('S_x_down')
            orbital_index.append(8)
            e_vec_atom.append(i)
            e_vec_orbital.append('S_y_down')
            orbital_index.append(9)
            e_vec_atom.append(i)
            e_vec_orbital.append('S_z_down')
            orbital_index.append(10)
    orb_ind=orb_ind+1

eigen_value = np.load("eigval.npy")
eigen_vector = np.load("eigvec.npy")

####### Rearrange the band indices after the deletion ###########

c_band=[]
v_band=[]
for i in range(Nc):
    c_band.append(i)
for i in range(Nv):
    v_band.append(i+Nc)

############ Here we constructed the near distances and the the indices for the orbitals within the same atoms ##############

near_index=[]
near_index0=[]
near_d=[]

for i in range(dim):
    index=[]
    index0=[]
    d_index=[]
    for j in range(dim):
        d=((poscar[e_vec_atom[i]][0]-poscar[e_vec_atom[j]][0])**2+(poscar[e_vec_atom[i]][1]-poscar[e_vec_atom[j]][1])**2+(poscar[e_vec_atom[i]][2]-poscar[e_vec_atom[j]][2])**2)**0.5
        if(d<1e-8):
            index0.append(j)
        #index.append(j)
        d_index.append(d)

    #near_index.append(index)
    near_d.append(d_index)
    near_index0.append(index0)

################# Here we load the reference structures to get the distances for which the wannier integrals has been calculated ###################

poscar_ref=[]
file_name='poscar_ref.txt'
with open(file_name,'r') as file:
    for line in file:
        columns=list(map(float,line.strip().split()))
        poscar_ref.append(columns)

pos1=poscar_ref[3237]
near_dis=[]
pos=[]
for i in range(len(poscar_ref)):
    d=((pos1[0]-poscar_ref[i][0])**2+(pos1[1]-poscar_ref[i][1])**2+(pos1[2]-poscar_ref[i][2])**2)**0.5
    d=round(d,2)
    count=0
    for j in range(len(near_dis)):
       if(d==near_dis[j]):
            count=count+1
    if(count==0):
        near_dis.append(d)
        pos.append(poscar_ref[i])

near_dis=np.array(near_dis)
pos=np.array(pos)

idx=np.argsort(near_dis)

near_dis=near_dis[idx]
pos=pos[idx]

ci_basis=[]
for i in range(11):
    for j in range(i,11):
        ci_basis.append(np.array([i,j]))

####### This part will create a dictionary of Basis. It will search the basis index faster#######

####### ci_basis_dict basically stores the index for each orbital basis ##########

ci_basis_dict = {}
for idx, pair in enumerate(ci_basis):
    ci_basis_dict[tuple(pair)] = idx

basis_map=np.empty((norb,norb),dtype=np.int32)

for a in range(norb):
    for b in range(norb):
        i = min(a,b)
        j = max(a,b)
        basis_map[a,b] = ci_basis_dict[(i,j)]

######## cKDTree is a searching algorithm in python for quick search. Here we basically build the distance index searching. ###########

####### near_dis stores the reference distances and near_d stores the poscar distances ####################
tree=cKDTree(near_dis.reshape(-1,1))
distance_idx = np.full((dim,dim), -1, dtype=np.int16)

for i in range(dim):
    print(i)
    for j in range(dim):
        d = near_d[i][j]
        dist, idx = tree.query([[d]], k=1)
        if dist[0] < 0.1:
            distance_idx[i,j] = int(idx[0])

distance_idx = np.array(distance_idx,dtype=np.int16)
np.save("distance_idx",distance_idx)

######## We have deleted near_d to save ram ############
del near_d
gc.collect()
