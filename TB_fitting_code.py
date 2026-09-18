import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from matplotlib import cm
import math
import cmath
from numpy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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

################### SK matrix will be created here ############################

def SK(R,params,q,l,m,n):
    Vss_s=params[0]
    Vsp_s=params[1]
    Vsd_s=params[2]
    Vpp_s=params[3]
    Vpp_p=params[4]
    Vpd_s=params[5]
    Vpd_p=params[6]
    Vdd_s=params[7]
    Vdd_p=params[8]
    Vdd_d=params[9]
    Vps_s=params[10]
    Vds_s=params[11]
    Vdp_s=params[12]
    Vdp_p=params[13]
    # =====================================================
    # s ROW
    # =====================================================

    Es_s    = s_s(l,m,n,Vss_s)

    Es_x    = s_x(l,m,n,Vsp_s)
    Es_y    = s_y(l,m,n,Vsp_s)
    Es_z    = s_z(l,m,n,Vsp_s)

    Es_xy   = s_xy(l,m,n,Vsd_s)
    Es_yz   = s_yz(l,m,n,Vsd_s)
    Es_zx   = s_zx(l,m,n,Vsd_s)
    Es_x2y2 = s_x2y2(l,m,n,Vsd_s)
    Es_z2   = s_z2(l,m,n,Vsd_s)

    # =====================================================
    # px ROW
    # =====================================================

    Ex_s    = x_s(l,m,n,Vps_s)

    Ex_x    = x_x(l,m,n,Vpp_s,Vpp_p)
    Ex_y    = x_y(l,m,n,Vpp_s,Vpp_p)
    Ex_z    = x_z(l,m,n,Vpp_s,Vpp_p)

    Ex_xy   = x_xy(l,m,n,Vpd_s,Vpd_p)
    Ex_yz   = x_yz(l,m,n,Vpd_s,Vpd_p)
    Ex_zx   = x_zx(l,m,n,Vpd_s,Vpd_p)
    Ex_x2y2 = x_x2y2(l,m,n,Vpd_s,Vpd_p)
    Ex_z2   = x_z2(l,m,n,Vpd_s,Vpd_p)

    # =====================================================
    # py ROW
    # =====================================================

    Ey_s    = y_s(l,m,n,Vps_s)

    Ey_x    = y_x(l,m,n,Vpp_s,Vpp_p)
    Ey_y    = y_y(l,m,n,Vpp_s,Vpp_p)
    Ey_z    = y_z(l,m,n,Vpp_s,Vpp_p)

    Ey_xy   = y_xy(l,m,n,Vpd_s,Vpd_p)
    Ey_yz   = y_yz(l,m,n,Vpd_s,Vpd_p)
    Ey_zx   = y_zx(l,m,n,Vpd_s,Vpd_p)
    Ey_x2y2 = y_x2y2(l,m,n,Vpd_s,Vpd_p)
    Ey_z2   = y_z2(l,m,n,Vpd_s,Vpd_p)

    # =====================================================
    # pz ROW
    # =====================================================

    Ez_s    = z_s(l,m,n,Vps_s)

    Ez_x    = z_x(l,m,n,Vpp_s,Vpp_p)
    Ez_y    = z_y(l,m,n,Vpp_s,Vpp_p)
    Ez_z    = z_z(l,m,n,Vpp_s,Vpp_p)

    Ez_xy   = z_xy(l,m,n,Vpd_s,Vpd_p)
    Ez_yz   = z_yz(l,m,n,Vpd_s,Vpd_p)
    Ez_zx   = z_zx(l,m,n,Vpd_s,Vpd_p)
    Ez_x2y2 = z_x2y2(l,m,n,Vpd_s,Vpd_p)
    Ez_z2   = z_z2(l,m,n,Vpd_s,Vpd_p)

    # =====================================================
    # dxy ROW
    # =====================================================

    Exy_s    = xy_s(l,m,n,Vds_s)

    Exy_x    = xy_x(l,m,n,Vdp_s,Vdp_p)
    Exy_y    = xy_y(l,m,n,Vdp_s,Vdp_p)
    Exy_z    = xy_z(l,m,n,Vdp_s,Vdp_p)

    Exy_xy   = xy_xy(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Exy_yz   = xy_yz(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Exy_zx   = xy_zx(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Exy_x2y2 = xy_x2y2(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Exy_z2   = xy_z2(l,m,n,Vdd_s,Vdd_p,Vdd_d)

    # =====================================================
    # dyz ROW
    # =====================================================

    Eyz_s    = yz_s(l,m,n,Vds_s)

    Eyz_x    = yz_x(l,m,n,Vdp_s,Vdp_p)
    Eyz_y    = yz_y(l,m,n,Vdp_s,Vdp_p)
    Eyz_z    = yz_z(l,m,n,Vdp_s,Vdp_p)

    Eyz_xy   = yz_xy(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Eyz_yz   = yz_yz(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Eyz_zx   = yz_zx(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Eyz_x2y2 = yz_x2y2(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Eyz_z2   = yz_z2(l,m,n,Vdd_s,Vdd_p,Vdd_d)

    # =====================================================
    # dzx ROW
    # =====================================================

    Ezx_s    = zx_s(l,m,n,Vds_s)

    Ezx_x    = zx_x(l,m,n,Vdp_s,Vdp_p)
    Ezx_y    = zx_y(l,m,n,Vdp_s,Vdp_p)
    Ezx_z    = zx_z(l,m,n,Vdp_s,Vdp_p)

    Ezx_xy   = zx_xy(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ezx_yz   = zx_yz(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ezx_zx   = zx_zx(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ezx_x2y2 = zx_x2y2(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ezx_z2   = zx_z2(l,m,n,Vdd_s,Vdd_p,Vdd_d)

    # =====================================================
    # dx2-y2 ROW
    # =====================================================

    Ex2y2_s    = x2y2_s(l,m,n,Vds_s)

    Ex2y2_x    = x2y2_x(l,m,n,Vdp_s,Vdp_p)
    Ex2y2_y    = x2y2_y(l,m,n,Vdp_s,Vdp_p)
    Ex2y2_z    = x2y2_z(l,m,n,Vdp_s,Vdp_p)

    Ex2y2_xy   = x2y2_xy(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ex2y2_yz   = x2y2_yz(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ex2y2_zx   = x2y2_zx(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ex2y2_x2y2 = x2y2_x2y2(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ex2y2_z2   = x2y2_z2(l,m,n,Vdd_s,Vdd_p,Vdd_d)

    # =====================================================
    # dz2 ROW
    # =====================================================

    Ez2_s    = z2_s(l,m,n,Vds_s)

    Ez2_x    = z2_x(l,m,n,Vdp_s,Vdp_p)
    Ez2_y    = z2_y(l,m,n,Vdp_s,Vdp_p)
    Ez2_z    = z2_z(l,m,n,Vdp_s,Vdp_p)

    Ez2_xy   = z2_xy(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ez2_yz   = z2_yz(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ez2_zx   = z2_zx(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ez2_x2y2 = z2_x2y2(l,m,n,Vdd_s,Vdd_p,Vdd_d)
    Ez2_z2   = z2_z2(l,m,n,Vdd_s,Vdd_p,Vdd_d)

    H = [[Es_s   ,Es_x   ,Es_y   ,Es_z   ,Es_xy   ,Es_yz   ,Es_zx   ,Es_x2y2   ,Es_z2],\
            [Ex_s   ,Ex_x   ,Ex_y   ,Ex_z   ,Ex_xy   ,Ex_yz   ,Ex_zx   ,Ex_x2y2   ,Ex_z2],\
            [Ey_s   ,Ey_x   ,Ey_y   ,Ey_z   ,Ey_xy   ,Ey_yz   ,Ey_zx   ,Ey_x2y2   ,Ey_z2],\
            [Ez_s   ,Ez_x   ,Ez_y   ,Ez_z   ,Ez_xy   ,Ez_yz   ,Ez_zx   ,Ez_x2y2   ,Ez_z2],\
            [Exy_s  ,Exy_x  ,Exy_y  ,Exy_z  ,Exy_xy  ,Exy_yz  ,Exy_zx  ,Exy_x2y2  ,Exy_z2],\
            [Eyz_s  ,Eyz_x  ,Eyz_y  ,Eyz_z  ,Eyz_xy  ,Eyz_yz  ,Eyz_zx  ,Eyz_x2y2  ,Eyz_z2],\
            [Ezx_s  ,Ezx_x  ,Ezx_y  ,Ezx_z  ,Ezx_xy  ,Ezx_yz  ,Ezx_zx  ,Ezx_x2y2  ,Ezx_z2],\
            [Ex2y2_s,Ex2y2_x,Ex2y2_y,Ex2y2_z,Ex2y2_xy,Ex2y2_yz,Ex2y2_zx,Ex2y2_x2y2,Ex2y2_z2],\
            [Ez2_s  ,Ez2_x  ,Ez2_y  ,Ez2_z  ,Ez2_xy  ,Ez2_yz  ,Ez2_zx  ,Ez2_x2y2  ,Ez2_z2]]

    phase=np.exp(1j*np.dot(q,R))
    H=np.array(H,dtype=complex)
    H=H*phase
    rows=len(H)
    columns=len(H[0])

    return(H)

file1="POSCAR"
file2="PROCAR"
Ef=-3.3396
NBANDS=192
near_1st=3.19
near_2nd=4.00
E1=-2
E2=2
total_type=2
atom_no=np.array([1,2])

N_vbm=1
N_cbm=2

near=np.array([0.01,near_1st,near_2nd])

gamma=[0.0,0.0,0.0]
X=[0.33,0.33,0.0]
M=[0.5,0.0000000000,0.0000000000]
points=40
s_points=[gamma,X,M,gamma]


####### This is the Left side of tb matrix slater koster parameters. But you can see as there are 2 S so many rows are same. But we have written only the independent rows
# we will make this full matrix in TB function.

'''
SK_params_1st=np.array([[-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.76,0.84,-0.14,-0.0,-0.0,0.0,-0.0],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,2.51,-1.37],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,2.51,-1.37],\
        [-0.0,0.0,-0.0,1.27,-0.28,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0],\
        [-0.0,0.0,-0.0,1.27,-0.28,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0],\
        [-0.0,0.0,-0.0,1.27,-0.28,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0]])

SK_params_2nd=np.array([[-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.37,-0.36],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.37,-0.36],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0],\
        [-0.0,0.0,-0.0,0.0,-0.0,-0.0,0.0,-0.0,0.0,-0.0,-0.0,-0.0,0.0,-0.0]])
'''

SK_params_1st=np.array([[-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-0.67,0.83,-0.31,-100.0,-100.0,100.0,-100.0],\
        [-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,3.01,-1.41],\
        [-100.0,100.0,-100.0,1.32,-0.00,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,100.0,-100.0]])

SK_params_2nd=np.array([[-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,100.0,-100.0],\
        [-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,0.90,-0.37],\
        [-100.0,100.0,-100.0,100.0,-100.0,-100.0,100.0,-100.0,100.0,-100.0,-100.0,-100.0,100.0,-100.0]])
        
hop_no=np.array([1,2,3])

for i in range(len(SK_params_2nd)):
    var1=SK_params_2nd[i]
    var2=SK_params_1st[i]
    for j in range(len(var1)):
        if(abs(var1[j])!=100 and abs(var2[j])!=100):
            var1[j]=var1[j]/var2[j]
    SK_params_2nd[i]=var1

SK_params=np.array([SK_params_1st,SK_params_2nd])

########## Give Only independent self energies  ########


E0_1=np.array([100.0,100.0,100.0,100.0,-3.82,-3.00,100.0,100.0,-5.10])
E0_2=np.array([100.0,-5.45,100.0,-7.13,100.0,100.0,100.0,100.0,100.0])
#E0_3=np.array([100.0,-6.28,100.0,-6.95,100.0,100.0,100.0,100.0,100.0])

#--------- This is the indices which bands are equal ---------

E0_1_dp=np.array([[4,7],[5,6]])
E0_2_dp=np.array([[1,2]])
#E0_3_dp=np.array([[1,2]])

temp1=[E0_1,E0_2]
temp2=[E0_1_dp,E0_2_dp]

shape1 = SK_params.shape

E0=np.concatenate((E0_1,E0_2))
H0=np.zeros((9*total_type,9*total_type),dtype=complex)

############## Bounds for hopping parameters are created #######################

bounds_t=[(-5,0),(0,5),(-5,0),(0,5),(-5,0),(-5,0),(0,5),(-5,0),(0,5),(-5,0),(-5,0),(-5,0),(0,5),(-5,0)]

bounds_hop=[]
for i in range(len(SK_params)):
    for j in range(len(SK_params[i])):
        bounds_hop.extend(bounds_t)

################ Onsite energies in terms of delta ############################ 

E0=E0.reshape(total_type,9)

idx=[]
for i in range(len(E0)):
    idx.append(np.argsort(E0[i]))
for i in range(len(idx)):
    var1=E0[i].copy()
    for j in range(1,len(idx[i])):
        ind0=idx[i][j-1]
        ind=idx[i][j]
        if(E0[i][ind]!=100.0):
            var1[ind]=E0[i][ind]-E0[i][ind0]
    E0[i]=var1
#E0=E0.flatten()

bounds_E0=[]

bounds_on=[(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100),(100,100)]

for i in range(len(idx)):
    var1=bounds_on.copy()
    for j in range(1,len(idx[i])):
        ind0=idx[i][0]
        ind=idx[i][j]
        if(E0[i][ind]!=100.0):
            var1[ind]=(0,5)
    var1[ind0]=(-10,0)
    bounds_E0.extend(var1)
E0=E0.flatten()

#print(E0)
#print(bounds_E0)

params_full=np.concatenate((E0,SK_params.flatten()))
mask=(params_full!=100) & (params_full!=-100.0)
params_free=params_full[mask]

bounds_full=np.concatenate((bounds_E0,bounds_hop))
bounds_free=bounds_full[mask]

N1=E0.size
N2=SK_params.size


########### Reading part of POSCAR and PROCAR files ###########

with open(file1,"r") as file:
    lines=file.readlines()
    #print(lines)
    comp_name=lines[0][:-1]
    atomic_species=lines[5].strip().split()
    atomic_number=list(map(int,lines[6].strip().split()))
    a1=list(map(float,lines[2].strip().split()))
    a2=list(map(float,lines[3].strip().split()))
    a3=list(map(float,lines[4].strip().split()))
    a1=np.array(a1)
    a2=np.array(a2)
    a3=np.array(a3)
    total_atoms=sum(atomic_number)
    c_lines=lines[8:8+total_atoms]
    poscar=np.array([[float(x) for x in line.strip().split()] for line in c_lines])
    #print(poscar_c)

with open(file2,"r") as file:
    lines=file.readlines()
    dft_energy_global=[]
    dft_char_global=[]
    char=[]
    i=3
    while(i<=len(lines)):
        content=lines[i].strip().split()
        if(content[0]=="k-point"):
            orb_k=[]
            en=[]
            for j in range(NBANDS):
                i=i+2
                band=lines[i].strip().split()
                e=float(band[4])
                en.append(e-Ef)
                i=i+3
                orb_k1=[]
                for j in range(total_atoms):
                    orb=lines[i].strip().split()
                    orb_k1.append(np.array([orb[1],orb[4],orb[2],orb[3],orb[5],orb[6],orb[8],orb[9],orb[7]],dtype=float))
                    i=i+1
                orb_k1=np.array(orb_k1)
                x=orb_k1.flatten()
                x=x/np.sum(x)
                orb_k.append(x)
            i=i+3
            dft_energy_global.append(en)
            dft_char_global.append(orb_k)
            
with open("dft_band.dat","w") as fout:

    nk = len(dft_energy_global)
    nb = len(dft_energy_global[0])

    for k in range(nk):

        # write k-point index
        fout.write(f"{k+1:7d}")

        # write all band energies of this k-point
        for b in range(nb):
            fout.write(f"{dft_energy_global[k][b]:10.4f}")

        fout.write("\n")


def reconstruction(params_free):
    params=params_full.copy()
    params[mask]=params_free
    onsite=params[:N1]
    hop=params[N1:N1+N2].reshape(shape1)

    onsite=onsite.reshape(total_type,9)

    #print(hop)

    for i in range(len(idx)):
        var1=onsite[i].copy()
        for j in range(1,len(idx[i])):
            ind0=idx[i][j-1]
            ind=idx[i][j]
            if(onsite[i][ind]!=100.0):
                var1[ind]=onsite[i][ind]+var1[ind0]
        onsite[i]=var1
    var4=[]

    for i in range(len(onsite)):
        var1=onsite[i].copy()
        var2=temp2[i].copy()
        for j in range(len(var2)):
            var3=var2[j]
            var1[var3[1]]=var1[var3[0]]
        var4.extend(var1)
    onsite=var4
    #print(onsite)

    var1=hop[1]
    var2=hop[0]
    for i in range(len(var1)):
        for j in range(len(var1[i])):
            if(abs(var1[i][j])!=100 and abs(var2[i][j])!=100):
                var1[i][j]=var1[i][j]*var2[i][j]
    hop[1]=var1

    return(onsite,hop)

###################### Nearest neightbours will be calculated here ############################

tol=0.02
G_c = []
for i in range(-2,3):
    for j in range(-2,3):
        for k in range(-2,3):
            G = i*a1 + j*a2 + k*a3
            G_c.append(G)

G_c = np.array(G_c)

poscar_super=[]

for i in range(total_atoms):
    var=[]
    for j in range(len(G_c)):
        var.append(poscar[i]+G_c[j])
    poscar_super.append(var)

near_atoms=[]
for p in range(len(near)-1):
    near_atoms_1st=[]
    for i in range(total_atoms):
        for j in range(i,total_atoms):
            atoms=poscar_super[j]
            d_min=near[p]+tol
            d_max=near[p+1]+tol
            #print(d_max,i,j)
            var=[]
            for k in range(len(atoms)):
                d=((atoms[k][0]-poscar[i][0])**2+(atoms[k][1]-poscar[i][1])**2+(atoms[k][2]-poscar[i][2])**2)**0.5
                if(d>d_min and d<=d_max):
                    R=np.array([atoms[k][0]-poscar[i][0],atoms[k][1]-poscar[i][1],atoms[k][2]-poscar[i][2]])
                    var.append(R)
            #print(var)
            near_atoms_1st.append(var)
    near_atoms.append(near_atoms_1st)

#################### Reciprocal Lattice Vectors ###################

V=np.dot(a1,np.cross(a2,a3))
b1=(2.0*np.pi*np.cross(a2,a3))/V
b2=(2.0*np.pi*np.cross(a3,a1))/V
b3=(2.0*np.pi*np.cross(a1,a2))/V

################### Tight Binding Hamiltonian #####################

def tb_hamil(q,onsite,hop):
    onsite=np.array(onsite)
    onsite=onsite.reshape(total_type,9)
    onsite=np.repeat(onsite,atom_no,axis=0)
    onsite=onsite.flatten()

    #print(onsite)

    H0=np.zeros((9*total_atoms,9*total_atoms),dtype=complex)
    for i in range(len(onsite)):
        H0[i][i]=onsite[i]
    qx=b1[0]*q[0]+b2[0]*q[1]+b3[0]*q[2]
    qy=b1[1]*q[0]+b2[1]*q[1]+b3[1]*q[2]
    qz=b1[2]*q[0]+b2[2]*q[1]+b3[2]*q[2]
    q=np.array([qx,qy,qz])
    H_tb=np.zeros((9*total_atoms,9*total_atoms),dtype=complex)
    for p in range(len(near)-1):
        rows=0
        columns=0
        flag=0
        near_atoms_1st=near_atoms[p]
        #print(near_atoms_1st)
        hop1=hop[p]
        hop1=np.repeat(hop1,hop_no,axis=0)
        # restore forbidden hoppings to zero
        hop1[(hop1 == 100.0) | (hop1 == -100.0)] = 0.0
        #print(hop1)
        for i in range(total_atoms):
            rows=i*9
            for j in range(i,total_atoms):
                columns=j*9
                H_t=np.zeros((9,9),dtype=complex)
                atoms=near_atoms_1st[flag]
                params=hop1[flag]
                for k in range(len(atoms)):
                    d=(atoms[k][0]**2+atoms[k][1]**2+atoms[k][2]**2)**0.5
                    l=atoms[k][0]/d
                    m=atoms[k][1]/d
                    n=atoms[k][2]/d
                    H_t=H_t+SK(atoms[k],params,q,l,m,n)
                for w in range(len(H_t)):
                    for x in range(len(H_t[0])):
                        H_tb[rows+w][columns+x]=H_tb[rows+w][columns+x]+H_t[w][x]
                flag=flag+1

    for i in range(len(H_tb)):
        for j in range(i,len(H_tb[0])):
            H_tb[j][i]=np.conj(H_tb[i][j])

    H=H_tb+H0

    return(H)

def plot_tb(onsite,hop,plot):
    s_points=[gamma,X,M,gamma]

    k_min=((s_points[0][0])**2+(s_points[0][1])**2+(s_points[0][2])**2)**0.5
    k_max=k_min
    k_grid=[]
    x_axis=[]
    k_grid.append(k_max)

    k_nom=['$\Gamma$','R','X','$\Gamma$']
    colour=['blue','blue','blue']

    dim=9*total_atoms
    points=40

    tb_char_global=[]
    tb_energy_global=[]

    #print(onsite)
    #print(hop)
    for i in range(len(s_points)-1):
        #Here 8 is the dimension of the H matrix & 100 is the number of points of k 
        #	in any path of K points so you can change.
	
        energy=np.zeros((dim,points))		
        k_min=k_max
        kx=np.linspace(s_points[i][0],s_points[i+1][0],points)
        ky=np.linspace(s_points[i][1],s_points[i+1][1],points)
        kz=np.linspace(s_points[i][2],s_points[i+1][2],points)
        k_mod=((s_points[i][0]-s_points[i+1][0])**2+(s_points[i][1]-s_points[i+1][1])**2+(s_points[i][2]-s_points[i+1][2])**2)**0.5
        k_max=k_max+k_mod
        plt.axvline(x=k_max,linestyle="--")
        k_grid.append(k_max)
        x=np.linspace(k_min,k_max,points)

        #tb_char=[]
        #tb_energy=[]

        for j in range(points):
            eigen_value=np.zeros(dim)
            q=np.array([kx[j],ky[j],kz[j]])
            eigen_value,eigen_vector=linalg.eigh(tb_hamil(q,onsite,hop))
            eigen_vector=eigen_vector.T
            tb_char_global.append(np.array(np.abs(eigen_vector)**2))
            tb_energy_global.append(np.array(eigen_value-Ef))
            for l in range(dim):
                energy[l][j]=eigen_value[l]-Ef
        if(plot=='TRUE'):
            for l in range(dim):
                plt.plot(x,energy[l],c=colour[i])
    
    if(plot=='TRUE'):
        plt.xticks(k_grid,k_nom)
        plt.ylabel("E")
        plt.title("Band Structure")
        plt.show()

    return(tb_energy_global,tb_char_global)

onsite,hop=reconstruction(params_free)
plot='TRUE'
tb_energy_global,tb_char_global=plot_tb(onsite,hop,plot)

K_vbm=39
K_cbm=39

var5_vbm=[]
var5_cbm=[]
var6_vbm=[]
var6_cbm=[]

for i in range(len(dft_energy_global)):

    var1=dft_energy_global[i]
    var3=tb_energy_global[i]

    # DFT indices
    for j in range(len(var1)):
        if(var1[j] > 1e-5):
            if(i == K_vbm):
                for p in range(N_vbm):
                    var5_vbm.append(j-p-1)
                var5_vbm=var5_vbm[::-1]
            if(i == K_cbm):
                for p in range(N_cbm):
                    var5_cbm.append(j+p)
            break

    # TB indices
    for j in range(len(var3)):
        if(var3[j] > 1e-5):
            if(i == K_vbm):
                for p in range(N_vbm):
                    var6_vbm.append(j-p-1)
                var6_vbm=var6_vbm[::-1]
            if(i == K_cbm):
                for p in range(N_cbm):
                    var6_cbm.append(j+p)
            break
dft_index=np.concatenate((var5_vbm,var5_cbm))
tb_index=np.concatenate((var6_vbm,var6_cbm))

print(dft_index)
print(tb_index)

wk_array=[]
sigma_k=5
t_points=points*(len(s_points)-1)
'''
for i in range(t_points):
    t=i/(t_points-1)
    wk_base=1.0+9.0*(np.cos(np.pi*t))**2
    if(K_vbm==K_cbm):
        wk_vbm=5.0*np.exp(-(i-K_vbm)**2/(2*sigma_k**2))
        wk_cbm=5.0*np.exp(-(i-K_cbm)**2/(2*sigma_k**2))
    else:
        wk_vbm=10.0*np.exp(-(i-K_vbm)**2/(2*sigma_k**2))
        wk_cbm=10.0*np.exp(-(i-K_cbm)**2/(2*sigma_k**2))
    # total weight
    wk=wk_base+wk_vbm+wk_cbm
    print(wk)
    wk_array.append(wk)
#print(wk_array)
'''
# symmetry-point indices
symmetry_points = [0,39,79,119]

for i in range(t_points):
    # base weight everywhere
    wk = 1.0
    # add smooth peaks at symmetry points
    for ks in symmetry_points:
        wk += 9.0*np.exp(-(i-ks)**2/(2*sigma_k**2))
    #print(wk)
    wk_array.append(wk)

def objective(params_free,alpha):
    onsite,hop=reconstruction(params_free)
    plot='FALSE'
    tb_energy,tb_char=plot_tb(onsite,hop,plot)
    dft_energy=dft_energy_global.copy()
    dft_char=dft_char_global.copy()
    
    #####   Band index matching   ########

    dft_energy2=[]
    dft_char2=[]
    tb_energy2=[]
    tb_char2=[]
    for i in range(len(dft_energy)):
        var1=dft_energy[i]
        var2=dft_char[i]
        var3=tb_energy[i]
        var4=tb_char[i]
        var5=[]
        var6=[]
        for j in range(len(dft_index)):
            var5.append(var1[dft_index[j]])
            var6.append(var2[dft_index[j]])
        var7=[]
        var8=[]
        for j in range(len(tb_index)):
            var7.append(var3[tb_index[j]])
            var8.append(var4[tb_index[j]])
        dft_energy2.append(var5)
        dft_char2.append(var6)
        tb_energy2.append(var7)
        tb_char2.append(var8)
  
    ##### LOSS FUNCTION #####

    L_en = 0
    L_orb = 0
    sigma = 0.5
    # symmetry-point indices
    symmetry_points = [0,39,79,119]
    # width of Gaussian in k-point index space
    sigma_k = 3
    total_weight = 0
    for i in range(len(dft_energy2)):
        var1 = dft_energy2[i]
        var2 = dft_char2[i]
        var3 = tb_energy2[i]
        var4 = tb_char2[i]
        
        for j in range(len(var1)):
            if(var1[j]>1e-5):
                Ecbm=var1[j]
                Evbm=var1[j-1]
                break
        
        x = 0.0
        y = 0.0
        '''
        ##### Smooth k-point weight #####
        wk = 1.0
        for ks in symmetry_points:            
            if(ks==39):
                wk += 18.0*np.exp(-(i-ks)**2/(2*sigma_k**2))
                #wk=wk+1000
            else:
                wk += 9.0*np.exp(-(i-ks)**2/(2*sigma_k**2))
            
            #wk = wk+10.0*np.exp(-(i-ks)**2/(2*sigma_k**2))
        '''
        wk=wk_array[i]
        ##### Band loss #####
        for j in range(len(var1)):
            # prioritize near-Fermi bands
            omega_v = np.exp(-(var1[j]-Evbm)**2/(2*sigma**2))
            omega_c = np.exp(-(var1[j]-Ecbm)**2/(2*sigma**2))
            omega = omega_v + omega_c
            #omega = np.exp(-(var1[j])**2/(2*sigma**2))
            # energy loss
            x += omega*(abs(var1[j]-var3[j])**2)
            # orbital overlap loss
            overlap = np.clip(np.dot(var2[j],var4[j]),0.0,1.0)
            y += alpha*omega*(1-overlap)**2
        ##### Weighted accumulation #####
        L_en = L_en+wk*x
        L_orb = L_orb+wk*y
        total_weight = total_weight+wk
    ##### Final normalized loss #####
    
    L = (L_en + L_orb)/total_weight
    #print(L)
    return(L)

def min_routine(params_free, bounds_free, alpha):
    x=np.array(params_free.copy(), dtype=float)
    # Initial step size
    dp=0.02*np.maximum(np.abs(x), 1.0)
    print(dp)
    err_old=objective(x, alpha)
    print("Initial error =", err_old)
    for iteration in range(200):
        found = False
        # scan parameters
        for i in range(len(x)):
            accepted = False
            # try both directions
            for direction in [-1, 1]:
                x_trial = x.copy()
                # trial move
                x_trial[i]=x_trial[i]+direction*dp[i]
                # enforce bounds
                x_trial[i]=np.clip(x_trial[i],bounds_free[i][0],bounds_free[i][1])
                err_new=objective(x_trial, alpha)
                # accept only if improved
                if err_new < err_old:
                    x=x_trial
                    err_old = err_new
                    dp[i]=dp[i]*1.2
                    found = True
                    accepted = True
                    print("iter =", iteration,"param =", i,"dir =", direction,"err =", err_old)
                    break

        # no improvement anywhere
        if not found:
            dp=dp*0.2
            print("Reducing step size :",np.max(dp))
        # convergence
        if np.max(dp) < 1e-4 :
            print("Converged")
            break
    print("Final error =", err_old)
    return x, err_old

###### L-BFGS-B ######

#result=minimize(objective,params_free,method='L-BFGS-B',bounds=bounds_free)

alpha_array=[0.0,0.001,0.005,0.01,0.05,0.1]
x0=params_free
'''
for i in range(len(alpha_array)):
    x0,err=min_routine(x0,bounds_free,alpha_array[i])
    #result=minimize(objective,x0,args=(alpha_array[i],),method='L-BFGS-B',bounds=bounds_free)
    print(err,alpha_array[i])
    
    ########  PLOT FINAL BANDS  ###########
    onsite,hop=reconstruction(x0)
    print(onsite)
    print(hop)
    
    plot='FALSE'
    tb_energy,tb_char=plot_tb(onsite,hop,plot)
'''    
best_free=x0
print(best_free)

########  PLOT FINAL BANDS  ###########
onsite,hop=reconstruction(best_free)
plot='TRUE'
tb_energy,tb_char=plot_tb(onsite,hop,plot)

tb_energy_final=[]
tb_char_final=[]

for i in range(len(tb_energy)):
    var1=tb_energy[i]
    var2=tb_char[i]
    var3=[]
    var4=[]
    for j in range(len(tb_index)):
        var3.append(var1[tb_index[j]])
        var4.append(var2[tb_index[j]])
    tb_energy_final.append(var3)
    tb_char_final.append(var4)

file3=open("tb_bands.dat","w")
for i in range(len(tb_energy_final)):
    file3.write("{}    ".format(i+1))
    for j in range(len(tb_energy_final[i])):
        file3.write("{:.16f}    ".format(tb_energy_final[i][j]))
    file3.write("\n")
    
file4=open("tb_char.dat","w")
for i in range(len(tb_char_final)):
    file4.write("{}\n".format(i+1))
    for j in range(len(tb_char_final[i])):
        flag=0
        for k in range(len(tb_char_final[i][j])):
            flag=flag+1
            file4.write("{:.2f}    ".format(tb_char_final[i][j][k]))
            if(flag==9):
                file4.write("\n")
                flag=0
        file4.write("\n")
    file4.write("\n")
    
file3.close()
file4.close()



