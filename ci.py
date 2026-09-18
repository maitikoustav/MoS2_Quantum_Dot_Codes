import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from matplotlib import cm
import math
import cmath
from numpy import linalg
from scipy.interpolate import griddata
from scipy.constants import e, epsilon_0
from scipy.special import struve, y0
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
import gc
import scipy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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

# =====================================================
# Screening parameters
# =====================================================

k = 2.5
delta = 33.875/k
ratio = 1.0
delta_ee = delta/ratio

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
print(eigen_value)

####### Rearrange the band indices after the deletion ###########

c_band=[]
v_band=[]
for i in range(Nc):
    c_band.append(i)
for i in range(Nv):
    v_band.append(i+Nc)


####### Here we load all the files regarding coulomb integrals ########

cfiles = []

for i in range(3866):
    cfiles.append(f"../integrals/coulomb_integral_{i}.dat")

Nci = 66
Ndist = 3866

ci_arrays = [np.zeros((Nci, Nci)) for _ in range(Ndist)]

for p in range(len(ci_arrays)):
    void=[]
    with open(cfiles[p],'r') as file:
        for line in file:
            void.append([float(x.strip('[]')) for x in line.strip().split()])
    v_index=0
    for i in range(len(ci_arrays[0])):
        for j in range(i,len(ci_arrays[0])):
            ci_arrays[p][i][j]=void[v_index][0]
            ci_arrays[p][j][i]=ci_arrays[p][i][j]
            v_index=v_index+1

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

######## cKDTree is a searching algorithm in python for quick search. Here we basically build the distance index searching. ##########

distance_idx=np.load("distance_idx.npy")

######## This is the main function which calculates the direct integral. We have to just give 4 bands. ########

def direct(band1,band2,band3,band4,delta):
    v_direct=0.0+0.0j
    ev1 = eigen_vector[:,band1]
    ev2 = eigen_vector[:,band2]
    ev3 = eigen_vector[:,band3]
    ev4 = eigen_vector[:,band4]

    ######## We neglected those co-efficients below 1e-3 for speedup #########

    nz1 = np.abs(ev1) > 1e-3
    nz2 = np.abs(ev2) > 1e-3
    nz3 = np.abs(ev3) > 1e-3
    nz4 = np.abs(ev4) > 1e-3

    ######## Active i,j are only those indices which has higher co-efficients ##########

    active_i = np.where(nz1)[0]
    active_j = np.where(nz2)[0]

    ######## This is the part where we use mpi parallelization over active_i ##########

    for idx in range(rank, len(active_i), size):
        i=active_i[idx]
        oi = orbital_index[i]

        ####### we need active_s and active_t because at a fixed ri and rj there are several orbitals so we have to sum over every orbitals with same ri and rj #####

        active_s = [ss for ss in near_index0[i] if nz3[ss]]
        for j in active_j:
            ###### Here we get the near_d index which we have precomputed before direct ######
            dis_idx = distance_idx[i][j]
            if dis_idx < 0:
                continue
            op = orbital_index[j]
            ####### oi and op will create bra basis ########
            basis_idx1 = basis_map[oi,op]
            prefactor=np.conj(ev1[i])*np.conj(ev2[j])
            active_t = [tt for tt in near_index0[j] if nz4[tt]]
            for ss in active_s:
                os = orbital_index[ss]
                for tt in active_t:
                    ot = orbital_index[tt]
                    ######## os and ot will create ket basis ########
                    basis_idx2 = basis_map[os,ot]
                    v_direct=v_direct-ci_arrays[dis_idx][basis_idx1][basis_idx2]*prefactor*ev3[ss]*ev4[tt]

    ###### This is the part where will gather all mpi calculations and sum over them. ######
    v_direct = comm.allreduce(v_direct, op=MPI.SUM)
    v=v_direct

    return(v)

#################### This is the part where we will calculate all possible matrix elements. ####################

# =====================================================
# Select the bands you want to keep
# =====================================================

cb = c_band[:Nc]      # 12 conduction bands
vb = v_band[:Nv]      # 12 valence bands

# =====================================================
# Create maps from actual band index -> array index
# =====================================================

cb_map = {band:i for i,band in enumerate(cb)}
vb_map = {band:i for i,band in enumerate(vb)}
# Example:
#
# cb = [145,146,147,...]
#
# cb_map[145] = 0
# cb_map[146] = 1
# cb_map[147] = 2
#

# =====================================================
# Allocate arrays
# =====================================================

V_eh = np.zeros((Nc,Nv,Nc,Nv),dtype=complex)

V_ee = np.zeros((Nc,Nc,Nc,Nc),dtype=complex)

# =====================================================
# Electron-hole integrals
#
# V_eh[c,v,c',v']
# =====================================================

if(rank==0):
    print("Calculating e-h integrals ...")

eh_flag=0
for i,cband1 in enumerate(cb):
    for j,vband1 in enumerate(vb):
        for kk,cband2 in enumerate(cb):
            for ll,vband2 in enumerate(vb):
                eh_flag=eh_flag+1
                if(rank==0):
                    print(eh_flag)
                if (kk,ll,i,j) < (i,j,kk,ll):
                    continue
                val = direct(cband1,vband1,cband2,vband2,delta)
                V_eh[i,j,kk,ll] = val
                V_eh[kk,ll,i,j] = np.conj(val)

if(rank==0):
    print("e-h done")
# =====================================================
# Electron-electron integrals
#
# V_ee[c1,c2,c3,c4]
# =====================================================
if(rank==0):
    print("Calculating e-e integrals ...")

ee_flag=0
for i,cband1 in enumerate(cb):
    for j,cband2 in enumerate(cb):
        for kk,cband3 in enumerate(cb):
            for ll,cband4 in enumerate(cb):
                ee_flag=ee_flag+1
                if(rank==0):
                    print(ee_flag)
                # Hermitian symmetry
                if (kk,ll,i,j) < (i,j,kk,ll):
                    continue
                val = direct(cband1,cband2,cband3,cband4,delta_ee)
                V_ee[i,j,kk,ll] = val
                V_ee[kk,ll,i,j] = np.conj(val)

if(rank==0):
    print("e-e done")

# =====================================================
# Save to disk
# =====================================================

np.save("V_eh.npy",V_eh)
np.save("V_ee.npy",V_ee)

if(rank==0):
    print("Saved")

V_eh=np.load("V_eh.npy")
V_ee=np.load("V_ee.npy")
## exciton binding energy

def ci_exciton(bands):
    band1=bands[0]
    band2=bands[1]
    band3=bands[2]
    band4=bands[3]
    i  = cb_map[band1]
    j  = vb_map[band2]
    kk = cb_map[band3]
    ll = vb_map[band4]

    v_direct = V_eh[i,j,kk,ll]

    return(v_direct)
    
def ci_trion(bands,Eg_dot,ratio):
    #if(rank==0):
    #    print(bands)

    v_direct=0.0
    flag=0
    index=[]

    for p in range(3):
        if(bands[p]!=bands[p+3]):
            flag=flag+1
            index.append(p)

    if(flag>2):
        v_direct=0.0
    elif(flag==0):
        for p in range(3):
            for q in range(p+1,3):
                band1=bands[p]
                band2=bands[q]
                band3=bands[p+3]
                band4=bands[q+3]
                #print(band1,band2,band3,band4)
                if ((band1 in cb_map and band2 in vb_map)):
                    i  = cb_map[band1]
                    j  = vb_map[band2]
                    kk = cb_map[band3]
                    ll = vb_map[band4]
                    v_direct = v_direct+V_eh[i,j,kk,ll]

                elif ((band1 in vb_map and band2 in cb_map)):
                    i  = vb_map[band1]
                    j  = cb_map[band2]
                    kk = vb_map[band3]
                    ll = cb_map[band4]
                    v_direct = v_direct+V_eh[j,i,ll,kk]

                else:
                    i = cb_map[band1]
                    j = cb_map[band2]
                    kk = cb_map[band3]
                    ll = cb_map[band4]
                    v_direct = v_direct-(1/2)*(V_ee[i,j,kk,ll]-V_ee[i,j,ll,kk])

    elif(flag==1):
        band2=bands[index[0]]
        band4=bands[index[0]+3]
        for p in range(3):
            if(p!=index[0]):
                band1=bands[p]
                band3=bands[p+3]
                #print(band1,band2,band3,band4)
                if ((band1 in cb_map and band2 in vb_map)):
                    i  = cb_map[band1]
                    j  = vb_map[band2]
                    kk = cb_map[band3]
                    ll = vb_map[band4]
                    v_direct = v_direct+V_eh[i,j,kk,ll]

                elif ((band1 in vb_map and band2 in cb_map)):
                    i  = vb_map[band1]
                    j  = cb_map[band2]
                    kk = vb_map[band3]
                    ll = cb_map[band4]
                    v_direct = v_direct+V_eh[j,i,ll,kk]

                else:
                    i = cb_map[band1]
                    j = cb_map[band2]
                    kk = cb_map[band3]
                    ll = cb_map[band4]
                    v_direct = v_direct-(1/2)*(V_ee[i,j,kk,ll]-V_ee[i,j,ll,kk])

    elif(flag==2):
        band1=bands[index[0]+3]
        band2=bands[index[1]+3]
        band3=bands[index[0]]
        band4=bands[index[1]]
        #print(band1,band2,band3,band4)
        if ((band1 in cb_map and band2 in vb_map)):
            i  = cb_map[band1]
            j  = vb_map[band2]
            kk = cb_map[band3]
            ll = vb_map[band4]
            v_direct = v_direct+V_eh[i,j,kk,ll]

        elif ((band1 in vb_map and band2 in cb_map)):
            i  = vb_map[band1]
            j  = cb_map[band2]
            kk = vb_map[band3]
            ll = cb_map[band4]
            v_direct = v_direct+V_eh[j,i,ll,kk]

        else:
            i = cb_map[band1]
            j = cb_map[band2]
            kk = cb_map[band3]
            ll = cb_map[band4]
            v_direct = v_direct-(1/2)*(V_ee[i,j,kk,ll]-V_ee[i,j,ll,kk])

    return(v_direct)

Eg_dot=eigen_value[c_band[0]]-eigen_value[v_band[0]]

#Excitonic Hamiltonian

basis=[]

for i in range(len(c_band)):
    for j in range(len(v_band)):
        basis.append(np.array([c_band[i],v_band[j]]))
#print(basis)
H_ex=np.zeros((len(basis),len(basis)),dtype=complex)
for i in range(len(basis)):
    for j in range(i,len(basis)):
        bands=np.concatenate((basis[i],basis[j]))
        #if(rank==0):
        #    print(bands)
        H_ex[i][j]=ci_exciton(bands)
        if(i==j):
            self=basis[i]
            delta_Eg=eigen_value[self[0]]-eigen_value[self[1]]
            H_ex[i][j]=H_ex[i][j]+delta_Eg
for i in range(len(basis)):
    for j in range(i,len(basis)):
        H_ex[j][i]=np.conj(H_ex[i][j])


eh_value,eh_vector=linalg.eig(H_ex)
sorted_indices = np.argsort(eh_value)  # Ascending order
eh_value_sorted = eh_value[sorted_indices]
eh_vector_sorted = eh_vector[:, sorted_indices]
eh_value=eh_value_sorted
eh_vector=eh_vector_sorted

eh_value_ex=eh_value[0]

exciton_BE=Eg_dot-eh_value_ex
#if(rank==0):
print("Exciton_BE:",exciton_BE)
#print(eh_value)
file=open("exciton_be2","w")
file.write("{}\n".format(exciton_BE))
file.close()
    
###### Plot of exciton wavefunction #######

atom_index=[]
for i in range(len(poscar)):
    if(i<limit1):
        for j in range(orbital1):
            atom_index.append(i)
    elif(i>=limit1 and i<limit2):
        for j in range(orbital2):
            atom_index.append(i)
    elif(i>=limit2 and i<limit3):
        for j in range(orbital3):
            atom_index.append(i)
            
ex_vec=eh_vector[:,0]
print(np.sum(np.abs(ex_vec)**2))
print(np.abs(ex_vec).max())

idx=np.argsort(np.abs(ex_vec)**2)[::-1]

for k in range(10):
    print(basis[idx[k]],np.abs(ex_vec[idx[k]])**2)
hole_fixed=atom_index[np.argmax(abs(eigen_vector[:,v_band[0]])**2)]
print(hole_fixed)

wf=np.zeros(dim, dtype=complex)
for i in range(len(ex_vec)):
    for j in range(dim):
        c_ex=basis[i][0]
        v_ex=basis[i][1]
        c_vec=eigen_vector[:,c_ex]
        v_vec=eigen_vector[:,v_ex]
        if(atom_index[j]==hole_fixed):
            wf=wf+ex_vec[i]*c_vec*np.conj(v_vec[j])
    
rho_ex=np.abs(wf)**2

#Scatter Plot
x=np.zeros((total),dtype=float)
y=np.zeros((total),dtype=float)
prob=np.zeros((total),dtype=float)
for i in range(total):
    x[i]=poscar[i][0]
    y[i]=poscar[i][1]

for i in range(dim):
    prob[atom_index[i]]=prob[atom_index[i]]+rho_ex[i]

prob=prob/np.sum(prob)

# Plot
plt.figure(figsize=(7,7))

# Probability map
sc=plt.scatter(x,y,c=prob,cmap='YlOrRd',s=55,edgecolors='none')

# Colorbar
cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)

# Colorbar label
cbar.set_label('Probability', fontsize=22)

# Tick labels
cbar.ax.tick_params(labelsize=18, width=2, length=6, direction='in')

# Triangle boundary coordinates

x1,y1=0.00016,9.20976
x2,y2=-27.11484,56.18076
x3,y3=-25.51984,58.94376
x4,y4=28.71016,58.94376
x5,y5=30.30516,56.18076
x6,y6=3.19016,9.20976
triangle_x=[x1, x2, x3, x4, x5, x6, x1]
triangle_y=[y1, y2, y3, y4, y5, y6, y1]
plt.plot(triangle_x,triangle_y,color='black',linewidth=2)

plt.scatter(poscar[hole_fixed][0],poscar[hole_fixed][1],marker='x',s=200,c='blue')
#plt.xlim(-60, 60)
#plt.ylim(0, 120)
# Labels and title
plt.xlabel(r'X ($\mathrm{\AA}$)', fontsize=22)
plt.ylabel(r'Y ($\mathrm{\AA}$)', fontsize=22)
plt.title('Probability Map', fontsize=24)

# Tick label size
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Make axis border thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Make ticks thicker
ax.tick_params(axis='both', which='major',direction='in',length=7,width=2,labelsize=18)

plt.tight_layout()
plt.show()

#Trion Hamiltonian

basis=[]
for i in range(len(c_band)):
    for j in range(i+1,len(c_band)):
        for kk in range(len(v_band)):
            basis.append(np.array([c_band[i],c_band[j],v_band[kk]]))
#print(basis)

H_trion=np.zeros((len(basis),len(basis)),dtype=complex)
for i in range(len(basis)):
    for j in range(i,len(basis)):
        bands=np.concatenate((basis[i],basis[j]))
        #print(bands)
        H_trion[i][j]=ci_trion(bands,Eg_dot,ratio)
        if(i==j):
            self=basis[i]
            delta_Eg=eigen_value[self[0]]+eigen_value[self[1]]-eigen_value[self[2]]
            H_trion[i][j]=H_trion[i][j]+delta_Eg

for i in range(len(basis)):
    for j in range(i,len(basis)):
        H_trion[j][i]=np.conj(H_trion[i][j])

eh_value,eh_vector=linalg.eig(H_trion)
sorted_indices = np.argsort(eh_value)  # Ascending order
eh_value_sorted = eh_value[sorted_indices]
eh_vector_sorted = eh_vector[:, sorted_indices]

eh_value=eh_value_sorted
eh_vector=eh_vector_sorted

eh_value_tr=eh_value[0]

trion_BE=eh_value_ex+eigen_value[c_band[1]]-eh_value_tr

if(rank==0):
    print("Trion_BE & ratio :",trion_BE,ratio)
    file=open("trion_be2","w")
    file.write("{}\n".format(trion_BE))
    file.close()

