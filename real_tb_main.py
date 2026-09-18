import numpy as np
from numpy.linalg import eig
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

# At first we define all the interaction functions between all atoms. In main program if we call them they we return the corresponding matrix. 

#-----------SS Block-------------
def s_s(l,m,n,V1):
    Es_s=V1
    return(Es_s)

#-----------SP Block-------------
def s_x(l,m,n,V1):
    Es_x=l*V1
    return(Es_x)
def s_y(l,m,n,V1):
    Es_y=m*V1
    return(Es_y)
def s_z(l,m,n,V1):
    Es_z=n*V1
    return(Es_z)
def x_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Ex_s=s_x(l,m,n,V1)
    return(Ex_s)
def y_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Ey_s=s_y(l,m,n,V1)
    return(Ey_s)
def z_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Ez_s=s_z(l,m,n,V1)
    return(Ez_s)

#-----------PP Block-------------

def x_x(l,m,n,V1,V2):
    Ex_x=l**2*V1+(1-l**2)*V2
    return(Ex_x)
def y_y(l,m,n,V1,V2):
    Ey_y=m**2*V1+(1-m**2)*V2
    return(Ey_y)
def z_z(l,m,n,V1,V2):
    Ez_z=n**2*V1+(1-n**2)*V2
    return(Ez_z)
def x_y(l,m,n,V1,V2):
    Ex_y=l*m*V1-l*m*V2
    return(Ex_y)
def x_z(l,m,n,V1,V2):
    Ex_z=l*n*V1-l*n*V2
    return(Ex_z)
def y_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ey_x=x_y(l,m,n,V1,V2)
    return(Ey_x)
def y_z(l,m,n,V1,V2):
    Ey_z=m*n*V1-m*n*V2
    return(Ey_z)
def z_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ez_x=x_z(l,m,n,V1,V2)
    return(Ez_x)
def z_y(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ez_y=y_z(l,m,n,V1,V2)
    return(Ez_y)

#-----------SD Block-------------

def s_xy(l,m,n,V1):
    Es_xy=(3**0.5)*l*m*V1
    return(Es_xy)
def s_yz(l,m,n,V1):
    Es_yz=(3**0.5)*m*n*V1
    return(Es_yz)
def s_zx(l,m,n,V1):
    Es_zx=(3**0.5)*n*l*V1
    return(Es_zx)
def s_x2y2(l,m,n,V1):
    Es_x2y2=((3**0.5)/2.0)*(l**2-m**2)*V1
    return(Es_x2y2)
def s_z2(l,m,n,V1):
    Es_z2=(n**2-(l**2+m**2)/2.0)*V1
    return(Es_z2)
def xy_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Exy_s=s_xy(l,m,n,V1)
    return(Exy_s)
def yz_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Eyz_s=s_yz(l,m,n,V1)
    return(Eyz_s)
def zx_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Ezx_s=s_zx(l,m,n,V1)
    return(Ezx_s)
def x2y2_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Ex2y2_s=s_x2y2(l,m,n,V1)
    return(Ex2y2_s)
def z2_s(l,m,n,V1):
    l=-l
    m=-m
    n=-n
    Ez2_s=s_z2(l,m,n,V1)
    return(Ez2_s)

#-----------PD Block-------------

def x_xy(l,m,n,V1,V2):
    Ex_xy=3**0.5*l**2*m*V1+m*(1-2*l**2)*V2
    return(Ex_xy)
def x_yz(l,m,n,V1,V2):
    Ex_yz=3**0.5*l*m*n*V1-2*l*m*n*V2
    return(Ex_yz)
def x_zx(l,m,n,V1,V2):
    Ex_zx=3**0.5*l**2*n*V1+n*(1-2*l**2)*V2
    return(Ex_zx)
def x_x2y2(l,m,n,V1,V2):
    Ex_x2y2=(3**0.5/2)*l*(l**2-m**2)*V1+l*(1-l**2+m**2)*V2
    return(Ex_x2y2)
def x_z2(l,m,n,V1,V2):
    Ex_z2=l*(n**2-(l**2+m**2)/2)*V1-3**0.5*l*n**2*V2
    return(Ex_z2)
def y_xy(l,m,n,V1,V2):
    Ey_xy=3**0.5*m**2*l*V1+l*(1-2*m**2)*V2
    return(Ey_xy)
def y_yz(l,m,n,V1,V2):
    Ey_yz=3**0.5*m**2*n*V1+n*(1-2*m**2)*V2
    return(Ey_yz)
def y_zx(l,m,n,V1,V2):
    Ey_zx=3**0.5*l*m*n*V1-2*l*m*n*V2
    return(Ey_zx)
def y_x2y2(l,m,n,V1,V2):
    Ey_x2y2=(3**0.5/2)*m*(l**2-m**2)*V1-m*(1+l**2-m**2)*V2
    return(Ey_x2y2)
def y_z2(l,m,n,V1,V2):
    Ey_z2=m*(n**2-(l**2+m**2)/2)*V1-3**0.5*m*n**2*V2
    return(Ey_z2)
def z_xy(l,m,n,V1,V2):
    Ez_xy=3**0.5*l*m*n*V1-2*l*m*n*V2
    return(Ez_xy)
def z_yz(l,m,n,V1,V2):
    Ez_yz=3**0.5*n**2*m*V1+m*(1-2*n**2)*V2
    return(Ez_yz)
def z_zx(l,m,n,V1,V2):
    Ez_zx=3**0.5*n**2*l*V1+l*(1-2*n**2)*V2
    return(Ez_zx)
def z_x2y2(l,m,n,V1,V2):
    Ez_x2y2=(3**0.5/2)*n*(l**2-m**2)*V1-n*(l**2-m**2)*V2
    return(Ez_x2y2)
def z_z2(l,m,n,V1,V2):
    Ez_z2=n*(n**2-(l**2+m**2)/2)*V1+3**0.5*n*(l**2+m**2)*V2
    return(Ez_z2)
def xy_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Exy_x=x_xy(l,m,n,V1,V2)
    return(Exy_x)
def xy_y(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Exy_y=y_xy(l,m,n,V1,V2)
    return(Exy_y)
def xy_z(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Exy_z=z_xy(l,m,n,V1,V2)
    return(Exy_z)
def yz_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Eyz_x=x_yz(l,m,n,V1,V2)
    return(Eyz_x)
def yz_y(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Eyz_y=y_yz(l,m,n,V1,V2)
    return(Eyz_y)
def yz_z(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Eyz_z=z_yz(l,m,n,V1,V2)
    return(Eyz_z)
def zx_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ezx_x=x_zx(l,m,n,V1,V2)
    return(Ezx_x)
def zx_y(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ezx_y=y_zx(l,m,n,V1,V2)
    return(Ezx_y)
def zx_z(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ezx_z=z_zx(l,m,n,V1,V2)
    return(Ezx_z)
def x2y2_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ex2y2_x=x_x2y2(l,m,n,V1,V2)
    return(Ex2y2_x)
def x2y2_y(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ex2y2_y=y_x2y2(l,m,n,V1,V2)
    return(Ex2y2_y)
def x2y2_z(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ex2y2_z=z_x2y2(l,m,n,V1,V2)
    return(Ex2y2_z)
def z2_x(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ez2_x=x_z2(l,m,n,V1,V2)
    return(Ez2_x)
def z2_y(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ez2_y=y_z2(l,m,n,V1,V2)
    return(Ez2_y)
def z2_z(l,m,n,V1,V2):
    l=-l
    m=-m
    n=-n
    Ez2_z=z_z2(l,m,n,V1,V2)
    return(Ez2_z)

#-----------DD Block-------------

def xy_xy(l,m,n,V1,V2,V3):
    Exy_xy=3*l**2*m**2*V1+(l**2+m**2-4*l**2*m**2)*V2+(n**2+l**2*m**2)*V3
    return(Exy_xy)
def xy_yz(l,m,n,V1,V2,V3):
    Exy_yz=3*l*m**2*n*V1+l*n*(1-4*m**2)*V2+l*n*(m**2-1)*V3
    return(Exy_yz)
def xy_zx(l,m,n,V1,V2,V3):
    Exy_zx=3*l**2*m*n*V1+m*n*(1-4*l**2)*V2+m*n*(l**2-1)*V3
    return(Exy_zx)
def xy_x2y2(l,m,n,V1,V2,V3):
    Exy_x2y2=(3/2)*l*m*(l**2-m**2)*V1+2*l*m*(m**2-l**2)*V2+((l*m*(l**2-m**2))/2)*V3
    return(Exy_x2y2)
def xy_z2(l,m,n,V1,V2,V3):
    Exy_z2=(3**0.5)*(l*m*(n**2-(l**2+m**2)/2)*V1-2*l*m*n**2*V2+((l*m*(1+n**2))/2)*V3)
    return(Exy_z2)
def yz_xy(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Eyz_xy=xy_yz(l,m,n,V1,V2,V3)
    return(Eyz_xy)
def yz_yz(l,m,n,V1,V2,V3):
    Eyz_yz=3*m**2*n**2*V1+(m**2+n**2-4*m**2*n**2)*V2+(l**2+m**2*n**2)*V3
    return(Eyz_yz)
def yz_zx(l,m,n,V1,V2,V3):
    Eyz_zx=3*m*n**2*l*V1+m*l*(1-4*n**2)*V2+m*l*(n**2-1)*V3
    return(Eyz_zx)
def yz_x2y2(l,m,n,V1,V2,V3):
    Eyz_x2y2=(3/2)*m*n*(l**2-m**2)*V1-m*n*(1+2*(l**2-m**2))*V2+(m*n*(1+(l**2-m**2)/2))*V3
    return(Eyz_x2y2)
def yz_z2(l,m,n,V1,V2,V3):
    Eyz_z2=(3**0.5)*(m*n*(n**2-(l**2+m**2)/2)*V1+m*n*(l**2+m**2-n**2)*V2-(m*n*(l**2+m**2)/2)*V3)
    return(Eyz_z2)
def zx_xy(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ezx_xy=xy_zx(l,m,n,V1,V2,V3)
    return(Ezx_xy)
def zx_yz(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ezx_yz=yz_zx(l,m,n,V1,V2,V3)
    return(Ezx_yz)
def zx_zx(l,m,n,V1,V2,V3):
    Ezx_zx=3*l**2*n**2*V1+(l**2+n**2-4*l**2*n**2)*V2+(m**2+l**2*n**2)*V3
    return(Ezx_zx)
def zx_x2y2(l,m,n,V1,V2,V3):
    Ezx_x2y2=(3/2)*n*l*(l**2-m**2)*V1+n*l*(1-2*(l**2-m**2))*V2-(n*l*(1-(l**2-m**2)/2))*V3
    return(Ezx_x2y2)
def zx_z2(l,m,n,V1,V2,V3):
    Ezx_z2=(3**0.5)*(l*n*(n**2-(l**2+m**2)/2)*V1+l*n*(l**2+m**2-n**2)*V2-(l*n*(l**2+m**2)/2)*V3)
    return(Ezx_z2)
def x2y2_xy(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ex2y2_xy=xy_x2y2(l,m,n,V1,V2,V3)
    return(Ex2y2_xy)
def x2y2_yz(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ex2y2_yz=yz_x2y2(l,m,n,V1,V2,V3)
    return(Ex2y2_yz)
def x2y2_zx(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ex2y2_zx=zx_x2y2(l,m,n,V1,V2,V3)
    return(Ex2y2_zx)
def x2y2_x2y2(l,m,n,V1,V2,V3):
    Ex2y2_x2y2=(3/4)*(l**2-m**2)**2*V1+(l**2+m**2-(l**2-m**2)**2)*V2+(n**2+((l**2-m**2)**2)/4)*V3
    return(Ex2y2_x2y2)
def x2y2_z2(l,m,n,V1,V2,V3):
    Ex2y2_z2=(3**0.5)*((l**2-m**2)*(n**2-(l**2+m**2)/2)*(V1/2)+n**2*(m**2-l**2)*V2+((1+n**2)*(l**2-m**2)/4)*V3)
    return(Ex2y2_z2)
def z2_xy(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ez2_xy=xy_z2(l,m,n,V1,V2,V3)
    return(Ez2_xy)
def z2_yz(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ez2_yz=yz_z2(l,m,n,V1,V2,V3)
    return(Ez2_yz)
def z2_zx(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ez2_zx=zx_z2(l,m,n,V1,V2,V3)
    return(Ez2_zx)
def z2_x2y2(l,m,n,V1,V2,V3):
    l=-l
    m=-m
    n=-n
    Ez2_x2y2=x2y2_z2(l,m,n,V1,V2,V3)
    return(Ez2_x2y2)
def z2_z2(l,m,n,V1,V2,V3):
    Ez2_z2=(n**2-(l**2+m**2)/2)**2*V1+3*n**2*(l**2+m**2)*V2+(3/4)*(l**2+m**2)**2*V3
    return(Ez2_z2)


def dd(l,m,n,V1,V2,V3):
    Exy_xy=xy_xy(l,m,n,V1,V2,V3)
    Exy_yz=xy_yz(l,m,n,V1,V2,V3)
    Exy_zx=xy_zx(l,m,n,V1,V2,V3)
    Exy_x2y2=xy_x2y2(l,m,n,V1,V2,V3)
    Exy_z2=xy_z2(l,m,n,V1,V2,V3)
    Eyz_xy=yz_xy(l,m,n,V1,V2,V3)
    Eyz_yz=yz_yz(l,m,n,V1,V2,V3)
    Eyz_zx=yz_zx(l,m,n,V1,V2,V3)
    Eyz_x2y2=yz_x2y2(l,m,n,V1,V2,V3)
    Eyz_z2=yz_z2(l,m,n,V1,V2,V3)
    Ezx_xy=zx_xy(l,m,n,V1,V2,V3)
    Ezx_yz=zx_yz(l,m,n,V1,V2,V3)
    Ezx_zx=zx_zx(l,m,n,V1,V2,V3)
    Ezx_x2y2=zx_x2y2(l,m,n,V1,V2,V3)
    Ezx_z2=zx_z2(l,m,n,V1,V2,V3)
    Ex2y2_xy=x2y2_xy(l,m,n,V1,V2,V3)
    Ex2y2_yz=x2y2_yz(l,m,n,V1,V2,V3)
    Ex2y2_zx=x2y2_zx(l,m,n,V1,V2,V3)
    Ex2y2_x2y2=x2y2_x2y2(l,m,n,V1,V2,V3)
    Ex2y2_z2=x2y2_z2(l,m,n,V1,V2,V3)
    Ez2_xy=z2_xy(l,m,n,V1,V2,V3)
    Ez2_yz=z2_yz(l,m,n,V1,V2,V3)
    Ez2_zx=z2_zx(l,m,n,V1,V2,V3)
    Ez2_x2y2=z2_x2y2(l,m,n,V1,V2,V3)
    Ez2_z2=z2_z2(l,m,n,V1,V2,V3)

    H=[[Exy_xy,Exy_yz,Exy_zx,Exy_x2y2,Exy_z2],\
            [Eyz_xy,Eyz_yz,Eyz_zx,Eyz_x2y2,Eyz_z2],\
            [Ezx_xy,Ezx_yz,Ezx_zx,Ezx_x2y2,Ezx_z2],\
            [Ex2y2_xy,Ex2y2_yz,Ex2y2_zx,Ex2y2_x2y2,Ex2y2_z2],\
            [Ez2_xy,Ez2_yz,Ez2_zx,Ez2_x2y2,Ez2_z2]]

    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def pp(l,m,n,V1,V2):
    Ex_x=x_x(l,m,n,V1,V2)
    Ex_y=x_y(l,m,n,V1,V2)
    Ex_z=x_z(l,m,n,V1,V2)
    Ey_x=y_x(l,m,n,V1,V2)
    Ey_y=y_y(l,m,n,V1,V2)
    Ey_z=y_z(l,m,n,V1,V2)
    Ez_x=z_x(l,m,n,V1,V2)
    Ez_y=z_y(l,m,n,V1,V2)
    Ez_z=z_z(l,m,n,V1,V2)

    H=[[Ex_x,Ex_y,Ex_z],\
            [Ey_x,Ey_y,Ey_z],\
            [Ez_x,Ez_y,Ez_z]]
    
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def pd(l,m,n,V1,V2):
    Ex_xy=x_xy(l,m,n,V1,V2)
    Ex_yz=x_yz(l,m,n,V1,V2)
    Ex_zx=x_zx(l,m,n,V1,V2)
    Ex_x2y2=x_x2y2(l,m,n,V1,V2)
    Ex_z2=x_z2(l,m,n,V1,V2)
    Ey_xy=y_xy(l,m,n,V1,V2)
    Ey_yz=y_yz(l,m,n,V1,V2)
    Ey_zx=y_zx(l,m,n,V1,V2)
    Ey_x2y2=y_x2y2(l,m,n,V1,V2)
    Ey_z2=y_z2(l,m,n,V1,V2)
    Ez_xy=z_xy(l,m,n,V1,V2)
    Ez_yz=z_yz(l,m,n,V1,V2)
    Ez_zx=z_zx(l,m,n,V1,V2)
    Ez_x2y2=z_x2y2(l,m,n,V1,V2)
    Ez_z2=z_z2(l,m,n,V1,V2)

    H=[[Ex_xy,Ex_yz,Ex_zx,Ex_x2y2,Ex_z2],\
            [Ey_xy,Ey_yz,Ey_zx,Ey_x2y2,Ey_z2],\
            [Ez_xy,Ez_yz,Ez_zx,Ez_x2y2,Ez_z2]]
    
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_11_self():
    E_xy=-3.82
    E_yz=-3.00
    E_zx=-3.00
    E_x2y2=-3.82
    E_z2=-5.10
    E=[E_xy,E_yz,E_zx,E_x2y2,E_z2]

    H=[[E[0],0.0,0.0,0.0,0.0],\
            [0.0,E[1],0.0,0.0,0.0],\
            [0.0,0.0,E[2],0.0,0.0],\
            [0.0,0.0,0.0,E[3],0.0],\
            [0.0,0.0,0.0,0.0,E[4]]]
    
    #H=[[E[0],0.0,0.0],\
    #        [0.0,E[3],0.0],\
    #        [0.0,0.0,E[4]]]

    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_22_self():
    E_xy=-12.82
    E_yz=-12.00
    E_zx=-12.00
    E_x2y2=-12.82
    E_z2=-13.10
    E=[E_xy,E_yz,E_zx,E_x2y2,E_z2]

    H=[[E[0],0.0,0.0,0.0,0.0],\
            [0.0,E[1],0.0,0.0,0.0],\
            [0.0,0.0,E[2],0.0,0.0],\
            [0.0,0.0,0.0,E[3],0.0],\
            [0.0,0.0,0.0,0.0,E[4]]]
    
    #H=[[E[0],0.0,0.0],\
    #        [0.0,E[3],0.0],\
    #        [0.0,0.0,E[4]]]

    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_33_self():
    E_x=-5.45
    E_y=-5.45
    E_z=-7.13

    E=[E_x,E_y,E_z]
    
    H=[[E[0],0.0,0.0],\
            [0.0,E[1],0.0],\
            [0.0,0.0,E[2]]]
    
    #H=[[E[0],0.0],\
    #        [0.0,E[1]]]

    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_11_self_passivation():
    E_xy=-3.82
    E_yz=-3.00
    E_zx=-3.00
    E_x2y2=-3.82
    E_z2=-4.30
    E=[E_xy,E_yz,E_zx,E_x2y2,E_z2]

    H=[[E[0],0.0,0.0,0.0,0.0],\
            [0.0,E[1],0.0,0.0,0.0],\
            [0.0,0.0,E[2],0.0,0.0],\
            [0.0,0.0,0.0,E[3],0.0],\
            [0.0,0.0,0.0,0.0,E[4]]]
    
    #H=[[E[0],0.0,0.0],\
    #        [0.0,E[3],0.0],\
    #        [0.0,0.0,E[4]]]

    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_33_self_passivation():
    E_x=-7.45
    E_y=-7.45
    E_z=-9.13

    E=[E_x,E_y,E_z]
    
    H=[[E[0],0.0,0.0],\
            [0.0,E[1],0.0],\
            [0.0,0.0,E[2]]]
    
    #H=[[E[0],0.0],\
    #        [0.0,E[1]]]

    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)


def E_11_1(l,m,n):
    V_dd_sigma=-0.67
    V_dd_pi=0.83
    V_dd_del=-0.31
    
    H,rows,columns=dd(l,m,n,V_dd_sigma,V_dd_pi,V_dd_del)

    return(H,rows,columns)
    
def E_22_1(l,m,n):
    V_dd_sigma=-0.0
    V_dd_pi=0.0
    V_dd_del=-0.0
    
    H,rows,columns=dd(l,m,n,V_dd_sigma,V_dd_pi,V_dd_del)

    return(H,rows,columns)
    
def E_33_1(l,m,n):
    V_pp_sigma=1.32
    V_pp_pi=-0.0

    H,rows,columns=pp(l,m,n,V_pp_sigma,V_pp_pi)
    
    return(H,rows,columns)
    
def E_12_1(l,m,n):
    V_dd_sigma=-2.9
    V_dd_pi=2.3
    V_dd_del=-0.45
    
    H,rows,columns=dd(l,m,n,V_dd_sigma,V_dd_pi,V_dd_del)
    
    return(H,rows,columns)
    
def E_21_1(l,m,n):
    l=-l
    m=-m
    n=-n
    H1,rows,columns=E_12_1(l,m,n)
    H=np.zeros((columns,rows))
    for i in range(columns):
        for j in range(rows):
            H[i][j]=H1[j][i]
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_31_1(l,m,n):
    V_pd_sigma=-3.01
    V_pd_pi=1.41
    
    H,rows,columns=pd(l,m,n,V_pd_sigma,V_pd_pi)

    return(H,rows,columns)

def E_13_1(l,m,n):
    l=-l
    m=-m
    n=-n
    H1,rows,columns=E_31_1(l,m,n)
    H=np.zeros((columns,rows))
    for i in range(columns):
        for j in range(rows):
            H[i][j]=H1[j][i]
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_31_2(l,m,n):
    V_pd_sigma=-0.90
    V_pd_pi=0.37
    
    H,rows,columns=pd(l,m,n,V_pd_sigma,V_pd_pi)
    
    return(H,rows,columns)

def E_13_2(l,m,n):
    l=-l
    m=-m
    n=-n
    H1,rows,columns=E_31_2(l,m,n)
    H=np.zeros((columns,rows))
    for i in range(columns):
        for j in range(rows):
            H[i][j]=H1[j][i]
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)

def E_32_1(l,m,n):
    V_pd_sigma=-3.5
    V_pd_pi=2.5
    
    H,rows,columns=pd(l,m,n,V_pd_sigma,V_pd_pi)
    
    return(H,rows,columns)

def E_23_1(l,m,n):
    l=-l
    m=-m
    n=-n
    H1,rows,columns=E_32_1(l,m,n)
    H=np.zeros((columns,rows))
    for i in range(columns):
        for j in range(rows):
            H[i][j]=H1[j][i]
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)
    
def E_32_2(l,m,n):
    V_pd_sigma=-0.5
    V_pd_pi=0.5
    
    H,rows,columns=pd(l,m,n,V_pd_sigma,V_pd_pi)
    
    return(H,rows,columns)

def E_23_2(l,m,n):
    l=-l
    m=-m
    n=-n
    H1,rows,columns=E_32_2(l,m,n)
    H=np.zeros((columns,rows))
    for i in range(columns):
        for j in range(rows):
            H[i][j]=H1[j][i]
    rows=len(H)
    columns=len(H[0])

    return(H,rows,columns)


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
    
#----------------------------------------------------------INPUT START--------------------------------------------------------------

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

#----------------------------------------------------------INPUT END---------------------------------------------------------------#

total=atom1+atom2+atom3
dim=atom1*orbital1+atom2*orbital2+atom3*orbital3

limit1=atom1
limit2=atom1+atom2
limit3=atom1+atom2+atom3

# Here we read the input poscar data file.

poscar=[]
file_name="poscar.txt"
with open(file_name,'r') as file:
    for line in file:
        columns=list(map(float,line.strip().split()))
        poscar.append(columns)

hamiltonian=np.zeros((dim,dim))
row_array=np.zeros(total,dtype=int)
k1=0
k2=0
test=1
for i in range(total):
    for j in range(total):
        H=[[]]
        k2=row_array[j]
        if(i==j):
            if(i<limit1 and j<limit1):
                check=0
                for l in range(limit1,limit2):
                    d=((poscar[i][0]-poscar[l][0])**2+(poscar[i][1]-poscar[l][1])**2+(poscar[i][2]-poscar[l][2])**2)**0.5
                    if(d<=3.2):
                        check=check+1
                if(check==0):
                    H,rows,columns=E_11_self()
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                else:
                    H,rows,columns=E_11_self_passivation()
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
            if(i>=limit1 and i<limit2 and j>=limit1 and j<limit2):
                H,rows,columns=E_22_self()
                for p in range(rows):
                    for q in range(columns):
                        hamiltonian[k2+p][k1+q]=H[p][q]
            if(i>=limit2 and i<limit3 and j>=limit2 and j<limit3):
                check=0
                for l in range(limit1,limit2):
                    d=((poscar[i][0]-poscar[l][0])**2+(poscar[i][1]-poscar[l][1])**2+(poscar[i][2]-poscar[l][2])**2)**0.5
                    if(d<=2.5):
                        check=check+1
                if(check==0):
                    H,rows,columns=E_33_self()
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                else:
                    H,rows,columns=E_33_self_passivation()
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]

            #print(H)
        elif(i!=j):
            d=((poscar[j][0]-poscar[i][0])**2+(poscar[j][1]-poscar[i][1])**2+(poscar[j][2]-poscar[i][2])**2)**0.5
            #print(d)
            if(d<=r1st):
                l=(poscar[j][0]-poscar[i][0])/d
                m=(poscar[j][1]-poscar[i][1])/d
                n=(poscar[j][2]-poscar[i][2])/d
                #print(l,m,n)
                if(i<limit1 and j<limit1):
                    H,rows,columns=E_11_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i<limit1 and j>=limit1 and j<limit2):
                    H,rows,columns=E_12_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j<limit1):
                    H,rows,columns=E_21_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i<limit1 and j>=limit2 and j<limit3):
                    H,rows,columns=E_13_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j<limit1):
                    H,rows,columns=E_31_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j>=limit2 and j<limit3):
                    H,rows,columns=E_33_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j>=limit1 and j<limit2):
                    H,rows,columns=E_32_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j>=limit2 and j<limit3):
                    H,rows,columns=E_23_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j>=limit1 and j<limit2):
                    H,rows,columns=E_22_1(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                #print(H,i,j)
            elif(d>r1st and d<r2nd):
                l=(poscar[j][0]-poscar[i][0])/d
                m=(poscar[j][1]-poscar[i][1])/d
                n=(poscar[j][2]-poscar[i][2])/d
                #print(l,m,n)
                #if(i<limit1 and j<limit1):
                #    H,rows,columns=E_11_2(l,m,n)
                #    for p in range(rows):
                #        for q in range(columns):
                #            hamiltonian[k2+p][k1+q]=H[p][q]
                #elif(i<limit1 and j>=limit1 and j<limit2):
                #    H,rows,columns=E_12_2(l,m,n)
                #    for p in range(rows):
                #        for q in range(columns):
                #            hamiltonian[k2+p][k1+q]=H[p][q]
                #elif(i>=limit1 and i<limit2 and j<limit1):
                #    H,rows,columns=E_21_2(l,m,n)
                #    for p in range(rows):
                #        for q in range(columns):
                #            hamiltonian[k2+p][k1+q]=H[p][q]
                if(i<limit1 and j>=limit2 and j<limit3):
                    H,rows,columns=E_13_2(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j<limit1):
                    H,rows,columns=E_31_2(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                #elif(i>=limit2 and i<limit3 and j>=limit2 and j<limit3):
                #    H,rows,columns=E_33_2(l,m,n)
                #    for p in range(rows):
                #        for q in range(columns):
                #            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j>=limit1 and j<limit2):
                    H,rows,columns=E_32_2(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j>=limit2 and j<limit3):
                    H,rows,columns=E_23_2(l,m,n)
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                #elif(i>=limit1 and i<limit2 and j>=limit1 and j<limit2):
                #    H,rows,columns=E_22_2(l,m,n)
                #   for p in range(rows):
                #        for q in range(columns):
                #            hamiltonian[k2+p][k1+q]=H[p][q]
            else:
                l=0.0
                m=0.0
                n=0.0
                if(i<limit1 and j<limit1):
                    H,rows,columns=E_11_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i<limit1 and j>=limit1 and j<limit2):
                    H,rows,columns=E_12_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j<limit1):
                    H,rows,columns=E_21_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i<limit1 and j>=limit2 and j<limit3):
                    H,rows,columns=E_13_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j<limit1):
                    H,rows,columns=E_31_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j>=limit2 and j<limit3):
                    H,rows,columns=E_33_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit2 and i<limit3 and j>=limit1 and j<limit2):
                    H,rows,columns=E_32_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j>=limit2 and j<limit3):
                    H,rows,columns=E_23_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
                elif(i>=limit1 and i<limit2 and j>=limit1 and j<limit2):
                    H,rows,columns=E_22_1(l,m,n)
                    H=np.zeros((rows,columns))
                    for p in range(rows):
                        for q in range(columns):
                            hamiltonian[k2+p][k1+q]=H[p][q]
        row_array[j]=row_array[j]+rows
        k1=k1+columns
    k1=0

# Here we check if the matrix is symmetric or not.

flag=0
for i in range(dim):
    for j in range(dim):
        if(hamiltonian[i][j]!=hamiltonian[j][i]):
            #print(i,j)
            flag=flag+1
if(flag==0):
    print("The matrix is symmetric")
else:
    print("The matrix is antisymnmetric")

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


# Here we calculate the eigenvalues.

eigen_value,eigen_vector=linalg.eigh(hamiltonian)
del hamiltonian
gc.collect()
#eigen_value,eigen_vector=eigsh(hamiltonian,k=20,sigma=-2.71,which='LM')

print("Diagonalized")

sorted_indices = np.argsort(eigen_value)  # Ascending order
eigen_value_sorted = eigen_value[sorted_indices]
eigen_vector_sorted = eigen_vector[:, sorted_indices]

eigen_value=eigen_value_sorted
eigen_vector=eigen_vector_sorted

print("Sorting Completed")

file=open("Eigen.dat","w")

for i in range(len(eigen_value)):
    #if(eigen_value[i]<=2 and eigen_value[i]>=-5):
    file.write(f"{i}    {eigen_value[i]} \n")
    #file.write("{}    {} \n".format(i, eigen_value[i]))
    for j in range(dim):
        if(abs(eigen_vector[j][i])>=0.1):
            file.write(f"{e_vec_atom[j]}    {e_vec_orbital[j]}    {eigen_vector[j][i]} \n")
            #file.write("{}      {}      {} \n".format(e_vec_atom[j], e_vec_orbital[j], eigen_vector[j][i]))
print("Eigen.dat writing completed.")

np.save("eigval_all.npy",eigen_value)
np.save("eigvec_all.npy",eigen_vector)

######## Band Numbers #########
c1=1432
v1=1431
Nc=12
Nv=12
c_band=[]
v_band=[]
for i in range(Nc):
    c_band.append(c1+i)
for i in range(Nv):
    v_band.append(v1-i)

cv_bands=np.concatenate((c_band,v_band))

####### Delete other bands information which are not required #########

eigen_value = eigen_value[cv_bands]
eigen_vector = eigen_vector[:,cv_bands]

np.save("eigval.npy",eigen_value)
np.save("eigvec.npy",eigen_vector)

