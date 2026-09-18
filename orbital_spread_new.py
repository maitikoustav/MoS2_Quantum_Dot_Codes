import numpy as np
#import matplotlib.pyplot as plt
from numpy.linalg import eig
#from matplotlib import cm
import math
import cmath
from numpy import linalg
#import h5py
from scipy.constants import e, epsilon_0
from scipy.special import struve, y0
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI
import sys

#dist_index = int(sys.argv[1])
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

def cart2frac(pos):
    pos=pos-global_origin
    M=np.column_stack((a1,a2,a3))

    frac=np.linalg.solve(M,pos)

    return(frac%1)

def inter(filename,atom):
    file1=filename
    rad=[]
    with open(file1,"r") as file:
        lines=file.readlines()
        for i in range(28,len(lines)-2):
            dat=np.array(list(map(float,lines[i].strip().split())))
            rad.extend(dat)
    rad=np.array(rad)
    coord=[]

    u=np.linspace(0.0, 1.0, Nx, endpoint=False)
    v=np.linspace(0.0, 1.0, Ny, endpoint=False)
    w=np.linspace(0.0, 1.0, Nz, endpoint=False)

    flag=0
    psi=np.zeros((Nx,Ny,Nz),dtype=float)

    for p in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                psi[i][j][p]=rad[flag]
                flag=flag+1

    interp = RegularGridInterpolator((u, v, w), psi, method='linear',bounds_error=False, fill_value=0)

    Nr=30
    Ntheta=10
    Nphi=10
    r0=6

    r=np.linspace(0,r0,Nr)
    theta=np.linspace(0,np.pi,Ntheta)
    phi=np.linspace(0,2*np.pi,Nphi,endpoint=False)

    dr=r[1]-r[0]
    dtheta=theta[1]-theta[0]
    dphi=phi[1]-phi[0]

    #print(dr,dtheta,dphi)

    scoord_cart=[]
    scoord_frac=[]
    dV=[]
    psi_new=[]
    psi_r=[]
    for i in range(len(r)):
        sum1=0.0
        count=0
        for j in range(len(theta)):
            for p in range(len(phi)):
                posx=r[i]*np.sin(theta[j])*np.cos(phi[p])
                posy=r[i]*np.sin(theta[j])*np.sin(phi[p])
                posz=r[i]*np.cos(theta[j])
                new_pos=atom+np.array([posx,posy,posz])
                scoord_cart.append(new_pos)
                scoord_frac.append(cart2frac(new_pos))
                psi_new.append(interp(cart2frac(new_pos)))
                sum1=sum1+abs(interp(cart2frac(new_pos)))
                count=count+1
                dV.append(r[i]**2*np.sin(theta[j])*dr*dtheta*dphi)
        psi_r.append(sum1/count)

    #plt.plot(r,psi_r,label='Interpolated')
    #plt.xlabel('r (Angstrom)')
    #plt.ylabel('Average |psi|')
    #plt.title('S_py')
    #plt.grid(True)
    #plt.legend()
    #plt.show()

    norm=0.0
    for i in range(len(psi_new)):
        norm=norm+abs(psi_new[i])**2*dV[i]

    W_normalized=psi_new/np.sqrt(norm)
    psi_new=W_normalized

    r_cut=2.5
    scoord=[]
    srad=[]
    sdV=[]

    for i in range(len(psi_new)):
        d=((scoord_cart[i][0]-atom[0])**2+(scoord_cart[i][1]-atom[1])**2+(scoord_cart[i][2]-atom[2])**2)**0.5
        if(d<=r_cut):
            scoord.append(scoord_cart[i])
            srad.append(psi_new[i])
            sdV.append(dV[i])
    return(scoord,srad,sdV)
def RK(r,r0,eps_env):
    """
    Rytova-Keldysh potential in SI units (Joules).
    
    Parameters:
        r (float or np.array): distance (meters)
        r0 (float): screening length (meters)
        eps_env (float): average dielectric constant of environment
    
    Returns:
        V (float or np.array): potential energy (Joules)
    """
    x=r/r0
    prefactor=e**2/(8*epsilon_0*eps_env*r0*(1e-10))
    return (prefactor*(struve(0, x)-y0(x)))/e

def downfold(rad,pos):
    # pos shape: (84,84,576,3)
    rad=np.array(rad)
    pos=np.array(pos)
    f0=f1=f2=3
    print(rad.shape)
    s0,s1,s2 = rad.shape
    print(s0,s1,s2)
    new0, new1, new2 = s0//f0, s1//f1, s2//f2

    # reshape into blocks
    rad_r = rad.reshape(new0, f0, new1, f1, new2, f2)
    pos_r = pos.reshape(new0, f0, new1, f1, new2, f2, 3)

    # average over the block axes (1,3,5)
    rad_new = rad_r.mean(axis=(1,3,5))
    pos_new = pos_r.mean(axis=(1,3,5))

    rad_1d = rad_new.reshape(-1)
    pos_1d = pos_new.reshape(-1,3)

    return rad_1d,pos_1d
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

### Read .xsf files ###

rad_dz2=[]
rad_dxy=[]
rad_dx2y2=[]
rad_dzx=[]
rad_dyz=[]
rad_px1=[]
rad_py1=[]
rad_pz1=[]
rad_px2=[]
rad_py2=[]
rad_pz2=[]

files=[file5,file3,file2,file4,file1,file7,file8,file6,file10,file11,file9]
rad_arrays=[rad_dxy,rad_dyz,rad_dzx,rad_dx2y2,rad_dz2,rad_px1,rad_py1,rad_pz1,rad_px2,rad_py2,rad_pz2]

srad_arrays=[]
scoord_arrays=[]
sdV_arrays=[]

for irad in range(len(rad_arrays)):
    filename=files[irad]
    if(irad<5):
        atom_pos=Mo_pos
    elif(irad>=5 and irad<8):
        atom_pos=S1_pos
    else:
        atom_pos=S2_pos

    scoord,srad,sdV=inter(filename,atom_pos)
    srad_arrays.append(srad)
    scoord_arrays.append(scoord)
    sdV_arrays.append(sdV)
poscar=[]
file_name='poscar.txt'
with open(file_name,'r') as file:
    for line in file:
        columns=list(map(float,line.strip().split()))
        poscar.append(columns)

pos1=poscar[3237]
k=2.5
delta=33.875/k
near_dis=[]
pos=[]

for i in range(len(poscar)):
    d=((pos1[0]-poscar[i][0])**2+(pos1[1]-poscar[i][1])**2+(pos1[2]-poscar[i][2])**2)**0.5
    d=round(d,2)
    #if(d<=delta):
    count=0
    for j in range(len(near_dis)):
       if(d==near_dis[j]):
            count=count+1
    if(count==0):
        near_dis.append(d)
        pos.append(poscar[i])


near_dis = np.array(near_dis)
pos = np.array(pos)

idx = np.argsort(near_dis)

near_dis = near_dis[idx]
pos = pos[idx]

#for i in range(len(near_dis)):
#    print(near_dis[i])
#print(near_dis)
#print(near_pos)

print(len(near_dis))
poscar=np.array([pos[0]])
pos_index=np.arange(10,30)
#pos_index=dist_index

#Basis Creation

basis=[]
for i in range(len(rad_arrays)):
    for j in range(i,len(rad_arrays)):
        basis.append(np.array([i,j]))
print(basis)

tasks=[]

for p in range(len(basis)):
    for q in range(p,len(basis)):
        tasks.append((p,q))

print("Total tasks =",len(tasks))

my_tasks = tasks[rank::size]

k=2.5
delta=33.875/k
#v_tab = RK(near_dis,delta,k)

if rank == 0:
    print("Reading Wannier orbitals", flush=True)
for ind in range(len(pos_index)):
    local_results=[]
    for p,q in my_tasks:
        v_rad=0.0
        orb1=srad_arrays[basis[p][0]]
        orb2=srad_arrays[basis[p][1]]
        orb3=srad_arrays[basis[q][0]]
        orb4=srad_arrays[basis[q][1]]
        coord1=scoord_arrays[basis[p][0]]
        coord2=scoord_arrays[basis[p][1]]

        for i in range(len(scoord_arrays[basis[p][0]])):
            for j in range(len(scoord_arrays[basis[p][1]])):
                if(abs(orb1[i])>0.0001 and abs(orb2[j])>0.0001 and abs(orb3[i])>0.0001 and abs(orb4[j])>0.0001):
                    r1=np.array([poscar[0][0]+coord1[i][0],poscar[0][1]+coord1[i][1],poscar[0][2]+coord1[i][2]])
                    r2=np.array([pos[pos_index[ind]][0]+coord2[j][0],pos[pos_index[ind]][1]+coord2[j][1],pos[pos_index[ind]][2]+coord2[j][2]])
                    d=((r1[0]-r2[0])**2+(r1[1]-r2[1])**2+(r1[2]-r2[2])**2)**0.5
                    if(d<=1e-8):
                        d=3.19
                    v_rad=v_rad+np.conj(orb1[i])*np.conj(orb2[j])*orb3[i]*orb4[j]*RK(d,delta,k)*sdV_arrays[basis[p][0]][i]*sdV_arrays[basis[p][1]][j]
                    #v_rad=v_rad+np.conj(orb1[i])*np.conj(orb2[j])*orb3[i]*orb4[j]*np.interp(d,near_dis,v_tab)*sdV_arrays[basis[p][0]][i]*sdV_arrays[basis[p][1]][j]
        if rank == 0:
            print(f"Finished p={p}, q={q}", flush=True)
        local_results.append((p,q,v_rad))

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        final_results = []

        for rank_results in all_results:
            final_results.extend(rank_results)

        final_results.sort(key=lambda x: (x[0], x[1]))

        with open(f"coulomb_integral_{ind}.dat","w") as file:
            for p, q, val in final_results:
                file.write(" {}\n".format(val))
