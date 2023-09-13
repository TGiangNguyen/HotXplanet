# LAVA PLANET 1D HYDRODYNAMICAL MODEL - GIANG NGUYEN SEPT 7, 2023
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import sys
import time
import os
from operator import mul
from fractions import Fraction
import functools
import re
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>> DIRECTORIES WHERE DATA AND INPUTS ARE LOCATED <<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
directories = {
    'input_name'    :   'hd20329b.txt', # name of input file
    'path_xsect':   '/data/xsect',  # absorption cross-section data
    'path_planet'  :   '/data/planetparams', # stellar spectrum data
    'path_input'    :   '/input', # where the input file is
    'path_output'   :   '/output'
    }
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> RUN PARAMETERS <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
runparams = {
    'Tas'   :   50, # surface temperature temperature as if there's no star
    'sigfig_cap':    14,
    'current_stage' :   2,
    'final_stage'   : 99
    }
sysparams = {}
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>CONSTANTS<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
const = {
    'k' : 1.3799999999999998e-23, # Boltzmann constant [m^2*kg*s^-2*K^-1]
    'sigma' : 5.67e-8, # stefan-boltzmann constant [Watts.m/K^4]
    'm' : 7.3223e-26, # mass per molecule of SiO [kg/molecule]
    'c' : 2.99e8, # speed of light
    'h' : 6.626e-34, # planck constant
    'Cp'    : 851.2844, # heat capacity of SiO [J mol^-1 K^-1]/[kg*molecule/mol] = J / kg / K
    'u' : 0.7091, # limb darkening coefficient (linear), for instellation calculations
    'La' :   1e6 # latent heat of vaporization of SiO [J / kg]
}
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>SATURATED VAPOUR CALCULATION<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def press_vap_calc(Tf):
    return 6.16e13*mt.exp(-6.94e4/Tf)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>> INSTELLATION CALCULATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def nCk(n,k):
    return int(functools.reduce(mul, (Fraction(n-i, i+1) for i in range(k)),1))
def LegendrePoly(x,n):
    Pn = 0
    k = 0
    while k <= n:
        Pn = Pn + nCk(n,k)*nCk(n+k,k)*((x-1)/2)**k
        k = k + 1
    return Pn
def I_u(x):
    r = sysparams['r'] #
    r2 = sysparams['r2'] #
    dist = sysparams['dist'] #
    illum = sysparams['illum'] #
    C0 = 2/3 - (mt.pi/2)*(r/r2)+(r/r2)**2-(1/12)*(r/r2)**4
    C1 = (mt.pi/2)-2*(r/r2)+1/3*(r/r2)**2
    C2 = 1 - 0.5*(r/r2)**2
    C3 = 1/3*(r/r2)
    C4 = (-1/12)
    return ((r2*illum)/(mt.pi*dist**3))*((C0/mt.pi)+(C1/mt.pi)*(dist*x/r2)+(C2/mt.pi)*(dist*x/r2)**2+(C3/mt.pi)*(dist*x/r2)**3+(C4/mt.pi)*(dist*x/r2)**4)
def I_d(x):
    r = sysparams['r'] #
    r2 = sysparams['r2'] #
    dist = sysparams['dist'] #
    illum = sysparams['illum'] #
    C0 = 3/16-0.5*(r/r2)+(3/8)*(r/r2)**2-(1/16)**4
    C1 = 0.5-(3/4)*(r/r2)**2+(1/4)*(r/r2)**3
    C2 = (3/8)-(3/8)*(r/r2)**2
    C3 = (1/4)*(r/r2)
    C4 = -1/16
    return ((r2*illum)/(mt.pi*dist**3))*((C0)+(C1)*(dist*x/r2)+(C2)*(dist*x/r2)**2+(C3)*(dist*x/r2)**3+(C4)*(dist*x/r2)**4)
def illum_calc(Tas):
    sigma = const['sigma'] #
    Cp = const['Cp'] #
    R = sysparams['R'] #
    r = sysparams['r'] #
    dist = sysparams['dist'] #
    u = const['u']
    x = 0
    geof = sigma*Tas**4
    illum = (sysparams['r2']/sysparams['dist'])**2*const['sigma']*sysparams['T_stellar']**4
    sysparams['illum'] = locals()['illum']
    Tf0 = (illum/const['sigma'])**(1/4)
    illum = (sigma*Tf0**4*mt.pi*dist**2)/(LegendrePoly(mt.cos(x),1)+2*(r/dist)*LegendrePoly(mt.cos(x),2)+3*(r/dist)**2*LegendrePoly(mt.cos(x),3)*(r/dist)**3)
    sysparams['illum'] = locals()['illum']
    sysparams['geof'] = locals()['geof']
    theta1 = mt.acos((sysparams['r']+sysparams['r2'])/sysparams['dist'])
    theta2 = mt.acos((sysparams['r']-sysparams['r2'])/sysparams['dist'])
    sysparams['theta1'] = locals()['theta1']
    sysparams['theta2'] = locals()['theta2']
    x = sysparams['theta1']
    LHS = (illum/mt.pi/dist**2)*(LegendrePoly(mt.cos(x),1)+2*(r/dist)*LegendrePoly(mt.cos(x),2)+3*(r/dist)**2*LegendrePoly(mt.cos(x),3)*(r/dist)**3)
    RHS = ((3*(1-u)/(3-u))*I_u(mt.cos(x))+(2*u/(3-u))*I_d(mt.cos(x)))
    factor = LHS/RHS
    sysparams['factor'] = locals()['factor']
    return illum,geof
def L_calc(x):
    theta1 = sysparams['theta1']
    theta2 = sysparams['theta2']
    r = sysparams['r']
    r2 = sysparams['r2']
    dist = sysparams['dist']
    illum = sysparams['illum']
    geof = sysparams['geof']
    u = const['u']
    sigma = const['sigma']
    dist = sysparams['dist']
    if x<=theta1:
        L = (illum/mt.pi/dist**2)*(LegendrePoly(mt.cos(x),1)+2*(r/dist)*LegendrePoly(mt.cos(x),2)+3*(r/dist)**2*LegendrePoly(mt.cos(x),3)*(r/dist)**3)
    if x>theta1 and x<=theta2:
        L = sysparams['factor']*((3*(1-u)/(3-u))*I_u(mt.cos(x))+(2*u/(3-u))*I_d(mt.cos(x)))
    if x>theta2:
        L = 0
    if L <= 0:
        L = 0
    return L + geof
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>> RADIATIVE TRANSFER PART <<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def Planck_int1(T,mylambda): # Using series from lambda to infinity
    k,h,c = const['k'],const['h'],const['c']
    myconst = 2*mt.pi*k**4*T**4/(h**3*c**2)
    x = h*c/(mylambda*k*T)
    n=1
    planck_int = 0
    while n <= 100:
        a = x**3/n + 3*x**2/n**2 + 6*x/n**3 + 6/n**4
        b = mt.exp(-n*x)
        planck_int = planck_int + a*b
        n = n + 1
    return myconst*planck_int
def epsilon_calc(P): # calculate emissivity
    myxsect_band = sysparams['xsect_band']
    epsilon = np.array(myxsect_band)
    tau = np.array(myxsect_band)
    for i in range(len(xsect_band)):
        tau[i] = P*xsect_band[i]/(const['m']*sysparams['g'])
        epsilon[i] = 1 - mt.exp(-tau[i])
    return epsilon,tau
def RC_cooling(P,T): # calculate radiative cooling for the bands
    wavelength_band = np.array(sysparams['wavelength_band'])
    myxsect_band = np.array(sysparams['xsect_band'])
    epsilon,tau = epsilon_calc(P)
    bby = np.array(epsilon)
    for i in range(len(xsect_band)):
        bby[i] = np.abs(Planck_int1(T,wavelength_band[i])-Planck_int1(T,wavelength_band[i+1]))
    return np.dot(epsilon,bby),np.array(bby)*np.array(epsilon)
def stellar_abs(P): # calculate radiation absorbed by the
    wavelength_band = np.array(sysparams['wavelength_band'])
    myxsect_band = np.array(sysparams['xsect_band'])
    stellar_flux_band = np.array(sysparams['stellar_flux_band'])
    absorbed_band = np.array(0.*xsect_band)
    epsilon,tau = epsilon_calc(P)
    absorbed_band = np.array(epsilon)*np.array(stellar_flux_band)
    total_flux = np.dot(epsilon,stellar_flux_band)
    if total_flux<0:
        print('total_flux < 0 in stellar_abs')
        print('P={P}'.format(P=P))
    return total_flux,absorbed_band,epsilon #return as a fraction of total stellar flux
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>> SURFACE TEMPERATURE CALCULATION <<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def surf_temp_calc(theta,T,P,E):
    QRC,dummy = RC_cooling(P,T)
    Qstellar,dummy,dummy = stellar_abs(P)
    Qlatent = 0
    if E > 0:
        Qlatent = const['La']*const['m']*E
    totalflux = (1-runparams['RT_factor']*Qstellar)*L_calc(theta) + runparams['RT_factor']*QRC-runparams['La_factor']*Qlatent
    if totalflux < 0:
        print('total flux = {x} in surf_temp_calc'.format(x=totalflux))
        print('Q stellar fraction = {Qstellar}'.format(Qstellar=Qstellar))
    return (totalflux/const['sigma'])**(1/4)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>> ENERGY FLUX CALCULATION <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def w_calc(V,P,Tf,m,E,R,H): # find transfer coefficients
    Pv = press_vap_calc(Tf)
    rho = Pv/(R*Tf)
    Ve = m*E/rho
    V_fric2 = 1E-6
    eta = 1.8E-5*((Tf/291)**(1.5))*((291+120)/(Tf+120))
    dummycondition = 0
    dummycounter = 0
    while dummycondition == 0:
        V_fric = V/(2.5*(mt.log10(abs(9*V_fric2*(H/2)*rho/eta))))
        if abs(V_fric-V_fric2)<=1E-6:
            dummycondition = 1
        elif dummycounter > 1000:
            dummycondition = 1
            # print('warning: non-convergent friction velocity')
            # print('V,Vfric,Vfric2 =',V,V_fric,V_fric2)
        else:
            V_fric2 = V_fric
            dummycounter = dummycounter + 1
    if V == 0:
        Vd = 0
    else:
        Vd = V_fric**2/V
    if Ve <= 0:
        ws = 2.*Vd**2./(-Ve+2.*Vd)
    else:
        ws = (Ve**2+2*Vd*Ve+2*Vd**2)/(Ve+2*Vd)

    if Ve < 0:
        wa = (Ve**2-2*Vd*Ve+2*Vd**2)/(-Ve+2*Vd)
    else:
        wa = 2.*Vd**2./(Ve+2.*Vd)
    return ws,wa,Ve,Vd,V_fric
def Qsens_calc(y,x,Tf): # sensible heating calculation
    R = sysparams['R']
    g = sysparams['g']
    m = const['m']
    Cp = const['Cp']
    Beta = sysparams['Beta']
    k = const['k']
    V = y[0]
    P = y[1]
    T = y[2]
    Pv = press_vap_calc(Tf)
    rho_s = Pv/(R*Tf)
    mu_s = mt.sqrt(k*Tf/m)
    E = 1*(Pv-P)/(mt.sqrt(2*mt.pi*R*Tf)*m)
    H = k*T/(m*g)
    ws,wa,Ve,Vd,V_fric = w_calc(V,P,Tf,m,E,R,H)
    qs = Cp*Tf
    qa = V**2/2 + Cp*T
    tau = rho_s*(-V*wa)
    Q1 = rho_s*(ws*qs-wa*qa)
    return Q1,tau
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>> HYDRODYNAMICAL PART <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def VPTsolve(f1,f2,f3,Q,theta): # solve for state variables
    Beta = sysparams['Beta']
    Cp = const['Cp']
    radical = f2**2 - 2.*Beta*(2.-Beta)*f1*f3
    status = 1
    if radical <= 0:
        status = 0
        return 0,0,0,0,status
    V = (f2-mt.sqrt(radical))/(f1*(2-Beta)) # only difference between supersonic and subsonic
    if V<=0:
        status = 2
        V = 0
        return 0,0,0,0,status
    P = f1/V
    if P < 0:
        status = 2
        return 0,0,0,0,status
    T = (f2/P-V**2)/(Beta*Cp)
    dfmass,dfmom,dfenergy,Q,Tf,E = dy_calc([V,P,T,Q],theta)
    return V,P,T,Q,status
def y_calc(y,x): # solve for LHS of ODE
    g = sysparams['g']
    r = sysparams['r']
    m = const['m']
    k = const['k']
    Cp = const['Cp']
    R = sysparams['R']
    Beta = sysparams['Beta']
    V = y[0]
    P = y[1]
    T = y[2]
    fmass = V*P*mt.sin(x)
    fmom = (V**2 + Beta*Cp*T)*P*mt.sin(x)
    fenergy = ((V**2)/2+Cp*T)*V*P*mt.sin(x)
    return fmass,fmom,fenergy
def dy_calc(y,x): # solve for RHS of ODE
    g = sysparams['g']
    r = sysparams['r']
    m = const['m']
    k = const['k']
    Cp = const['Cp']
    R = sysparams['R']
    Beta = sysparams['Beta']
    sigma = const['sigma']
    illum = sysparams['illum']
    V = y[0]
    P = y[1]
    T = y[2]
    Q = y[3]
    E = y[0]
    Tf = surf_temp_calc(x,T,P,E)
    Qsens,tau = Qsens_calc([V,P,T],x,Tf)
    Pv = press_vap_calc(Tf)
    E = (Pv-P)/(mt.sqrt(2*mt.pi*R*Tf)*m)
    Qstellar2,Qstellar_band,epsilon = stellar_abs(P)
    Qstellar = L_calc(x)*Qstellar2
    QRC,QRC_band = RC_cooling(P,T)
    Qsurf,Qsurf_band = RC_cooling(P,Tf)
    Q = Qsens + runparams['RT_factor']*(Qstellar + Qsurf - 2*QRC)
    dfmass = m*E*g*r*mt.sin(x)
    dfmom = (Beta*Cp*T*P*mt.cos(x))+tau*g*r*mt.sin(x)
    dfenergy = Q*g*r*mt.sin(x)
    return dfmass,dfmom,dfenergy,Q,Tf,E
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>> SINGLE RUN WITH BOUNDARY CONDITIONS <<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def systemtest(y0,dthetadeg,xstart): # single run with starting boundary condition
    g = sysparams['g']
    r = sysparams['r']
    m = const['m']
    k = const['k']
    Cp = const['Cp']
    R = sysparams['R']
    Beta = sysparams['Beta']
#..........initial values calculations for the run..............................
    xstartrad = mt.radians(xstart)
    V = y0[0]
    P = y0[1]
    T = y0[2]
    Q = y0[3]
    E = y0[4]
    Tf = surf_temp_calc(xstartrad,T,P,E)
    Qsens = Qsens_calc([V,P,T],xstartrad,Tf)
    Pv = press_vap_calc(Tf)
    M = V*((1-Beta)/(Beta*Cp*T))**(1/2)
    E = (Pv-P)/(mt.sqrt(2*mt.pi*R*Tf)*m)
    Qstellar2,Qstellar_band,epsilon = stellar_abs(P)
    Qstellar = L_calc(xstartrad)*Qstellar2
    QRC,QRC_band = RC_cooling(P,T)
    Qsurf,Qsurf_band = RC_cooling(P,Tf)
    Q = Qsens + runparams['RT_factor']*(Qstellar + Qsurf - 2*QRC)
    Varray = [V]
    Parray = [P]
    Tarray = [T]
    Pvarray = [Pv]
    Marray = [M]
    Earray = [E]
    Qarray = [Q]
    Tfarray = [Tf]
    thetadeg = xstart + dthetadeg
    angledeg = [thetadeg]
    dthetarad = mt.radians(dthetadeg)
    Vnew = V
    Pnew = P
    Tnew = T
    criticalangle = 0
    while thetadeg<=180:
        Vcurrent = Vnew
        Pcurrent = Pnew
        Tcurrent = Tnew
        theta = mt.radians(thetadeg)
        Pv = press_vap_calc(Tf)
        fmass,fmom,fenergy = y_calc([Vcurrent,Pcurrent,Tcurrent],theta)
        dfmass,dfmom,dfenergy,Q,Tf,E = dy_calc([Vcurrent,Pcurrent,Tcurrent,Q],theta)

        fmass2 = ((1/2)*dfmass*dthetarad+fmass)/mt.sin(theta+dthetarad/2)
        fmom2 = ((1/2)*dfmom*dthetarad+fmom)/mt.sin(theta+dthetarad/2)
        fene2 = ((1/2)*dfenergy*dthetarad+fenergy)/mt.sin(theta+dthetarad/2)
        V,P,T,Q,status = VPTsolve(fmass2,fmom2,fene2,Q,theta+dthetarad/2)
        if status != 1:
            break

        #compute ftemp1 = f(xtemp1,tn+1/2*dt)
        dfmass2,dfmom2,dfene2,Q2,Tf2,E2 = dy_calc([V,P,T,Q],theta+dthetarad/2)

        #find xn = xn + dt*dftemp1
        f1 = (dfmass2*dthetarad+fmass)/mt.sin(theta+dthetarad)
        f2 = (dfmom2*dthetarad+fmom)/mt.sin(theta+dthetarad)
        f3 = (dfene2*dthetarad+fenergy)/mt.sin(theta+dthetarad)
        Vnew,Pnew,Tnew,Q,status = VPTsolve(f1,f2,f3,Q2,theta+dthetarad)
        if status != 1:
            break

        M = Vnew*((1-Beta)/(Beta*Cp*Tnew))**(1/2)
        if M<Marray[-1]:
            status = 2
            break
        Tf = surf_temp_calc(theta+dthetarad,Tnew,Pnew,E)
        Pv = press_vap_calc(Tf)

        thetadeg = thetadeg + dthetadeg
        Varray.extend([Vnew])
        Parray.extend([Pnew])
        Tarray.extend([Tnew])
        Pvarray.extend([Pv])
        Marray.extend([M])
        Earray.extend([E])
        Qarray.extend([Q])
        Tfarray.extend([Tf])
        angledeg.extend([thetadeg])

    return Varray,Parray,Tarray,Pvarray,Marray,Earray,angledeg,status,criticalangle,Qarray,Tfarray

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>> ACTUAL CODE STARTS HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
time0 = time.localtime()
time_start = time.time()

# set directories
home = os.getcwd()
path_output = home+directories['path_output']
path_input = home+directories['path_input']
path_xsect = home+directories['path_xsect']
path_planet = home+directories['path_planet']

# read the banded - xsection
os.chdir(path_xsect)
wavelength_band = np.genfromtxt('wavelength_bands.txt',delimiter=',')
xsect_band = np.genfromtxt('xsect_bands.txt',delimiter='\n')

# read input files to get planetary parameters
os.chdir(path_input)
regexp1 = re.compile(r'dtheta =.*?([0-9.-]+)')
regexp2 = re.compile(r'RT factor =.*?([0-9.-]+)')
regexp3 = re.compile(r'La factor =.*?([0-9.-]+)')
with open(directories['input_name']) as input_txt:
    for line in input_txt:
        match = regexp1.match(line)
        if match:
            dtheta = float(match.group(1))
        match = regexp2.match(line)
        if match:
            RT_factor = float(match.group(1))
        match = regexp3.match(line)
        if match:
            La_factor = float(match.group(1))
        if 'RUN NAME' in line:
            dummyline = line
            dummystr = ': '
            run_name = dummyline.split(dummystr)[1]
            run_name = run_name[0:-1]
        if 'planet' in line:
            dummyline = line
            dummystr = ': '
            planet_name = dummyline.split(dummystr)[1]
            planet_name = planet_name[0:-1]

runparams['dthetadeg'] = locals()['dtheta']
runparams['La_factor'] = locals()['La_factor']
runparams['RT_factor'] = locals()['RT_factor']

# read planetary parameter
os.chdir(path_planet)

regexp1 = re.compile(r'effective stellar temperature =.*?([0-9.-]+)')
regexp2 = re.compile(r'planetary radius =.*?([0-9.-]+)')
regexp3 = re.compile(r'stellar radius =.*?([0-9.-]+)')
regexp4 = re.compile(r'gravity at the surface =.*?([0-9.-]+)')
regexp5 = re.compile(r'distance between planet and star =.*?([0-9.-]+)')

with open(planet_name+'.txt') as input_txt:
    for line in input_txt:
        match = regexp1.match(line)
        if match:
            T_stellar = float(match.group(1))
        match = regexp2.match(line)
        if match:
            r = float(match.group(1))
        match = regexp3.match(line)
        if match:
            r2 = float(match.group(1))
        match = regexp4.match(line)
        if match:
            g = float(match.group(1))
        match = regexp5.match(line)
        if match:
            dist = float(match.group(1))
        if 'stellar directory' in line:
            dummyline = line
            dummystr = ': '
            path_stellar = dummyline.split(dummystr)[1]
            # path_stellar = path_stellar[0:-1]
            path_stellar = home+path_stellar

sysparams['run_name'] = locals()['run_name']
sysparams['r'] = locals()['r']
sysparams['r2'] = locals()['r2']
sysparams['T_stellar'] = locals()['T_stellar']
sysparams['dist'] = locals()['dist']
sysparams['g'] = locals()['g']

# read RT data
os.chdir(path_stellar)
stellar_flux_band = np.genfromtxt('banded_cumflux.txt',delimiter='\n')
sysparams['wavelength_band'] = locals()['wavelength_band']
sysparams['xsect_band'] = locals()['xsect_band']
sysparams['stellar_flux_band'] = locals()['stellar_flux_band']

# initiate R and Beta variables
R = const['k']/const['m']
sysparams['R'] = locals()['R']
Beta = sysparams['R']/(sysparams['R']+const['Cp'])
sysparams['Beta'] = locals()['Beta']

# initiate stellar variables for instellation calculation
illum_calc(runparams['Tas'])
Tf0 = surf_temp_calc(0,1,1,0) # substellar surface temperature under complete transparent sky

# initiate starting Pmax and Pmin to find the right initial pressure
dthetadeg0 = runparams['dthetadeg']
sigfig_cap = runparams['sigfig_cap']
Pdiff_cap = press_vap_calc(Tf0)/(10**sigfig_cap)

os.chdir(path_output)
runstr = 'output_'+run_name
if os.path.exists(runstr):
    os.chdir(runstr)
else:
    os.mkdir(runstr)
    os.chdir(runstr)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>> ACTUAL RUN STARTS HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

n = runparams['current_stage']

while n <= runparams['final_stage']:
# FINDING THE RIGHT INITIAL PRESSURE
    loop_start = time.time()

    lowerV,lowerP,lowerT,lowerPv,lowerM,lowerE,lowerangle,lowerQ,lowerTf = [],[],[],[],[],[],[],[],[]
    upperV,upperP,upperT,upperPv,upperM,upperE,upperangle,upperQ,upperTf = [],[],[],[],[],[],[],[],[]

    os.chdir('STAGE{n}'.format(n=n-1))
    regexp1 = re.compile(r'angle split at.*?([0-9.-]+)')
    regexp2 = re.compile(r'V0 = .*?([0-9.-]+)')
    regexp3 = re.compile(r'T0 =.*?([0-9.-]+)')
    regexp4 = re.compile(r'P0 = .*?([0-9.-]+)')
    regexp5 = re.compile(r'Tf0 = .*?([0-9.-]+)')
    regexp6 = re.compile(r'E0 = .*?([0-9,-]+)')
    regexp7 = re.compile(r'Q0 = .*?([0-9,-]+)')

    with open('output.txt') as input_txt:
        for line in input_txt:
            match = regexp1.match(line)
            if match:
                xstart = float(match.group(1))
            match = regexp2.match(line)
            if match:
                V0 = float(match.group(1))
            match = regexp3.match(line)
            if match:
                T0 = float(match.group(1))
            match = regexp4.match(line)
            if match:
                P0 = float(match.group(1))
            match = regexp5.match(line)
            if match:
                Tf0 = float(match.group(1))
            match = regexp6.match(line)
            if match:
                E0 = float(match.group(1))
            match = regexp7.match(line)
            if match:
                Q0 = float(match.group(1))
    os.chdir('..')

    Pmax = 1.5*P0
    Pmin = 0.5*P0
    statusarray,anglestopped,Pmidarray = [],[],[]
    loop_start = time.time()
    while Pmax-Pmin>=Pdiff_cap:
        Pmid = (Pmax+Pmin)/2
        Pv0 = press_vap_calc(Tf0)
        E0 = (Pv0-Pmid)/(mt.sqrt(2*mt.pi*sysparams['R']*Tf0)*const['m'])

        Varray,Parray,Tarray,Pvarray,Marray,Earray,angledeg,status,criticalangle,Qarray,Tfarray = systemtest([V0,Pmid,T0,Q0,E0],dthetadeg0,xstart)
        statusarray.extend([status])
        anglestopped.extend([angledeg[-1]])
        Pmidarray.extend([Pmid])
        if status == 0:
            Pmin = Pmid
        elif status == 2:
            Pmax = Pmid
    if status == 2:
        Varray2,Parray2,Tarray2,Pvarray2,Marray2,Earray2,angledeg2,status2,criticalangle2,Qarray2,Tfarray2 = systemtest([V0,Pmin,T0,Q0,E0],dthetadeg0,xstart)

    elif status == 0:
        Varray2,Parray2,Tarray2,Pvarray2,Marray2,Earray2,angledeg2,status2,criticalangle2,Qarray2,Tfarray2 = systemtest([V0,Pmax,T0,Q0,E0],dthetadeg0,xstart)


    if np.size(Qarray[0])>1:
        Qarray[0] = Qarray[0][0]
    if np.size(Qarray2[0])>1:
        Qarray2[0] = Qarray2[0][0]

    xstop = 0
    div_cap = Parray[0]*1e-5
    if len(angledeg)>len(angledeg2):
        Pdiff = abs(Parray[0]-Parray2[0])
        i = 1
        while i<len(angledeg2):
            Pdiff = abs(Parray[i]-Parray2[i])
            if Pdiff>=div_cap:
                xstop = angledeg2[i]
                index = i
                break
            i = i + 1
    if len(angledeg2)>len(angledeg):
        Pdiff = abs(Parray[0]-Parray2[0])
        i = 1
        while i<len(angledeg):
            Pdiff = abs(Parray[i]-Parray2[i])
            if Pdiff>=div_cap:
                xstop = angledeg[i]
                index = i
                break
            i = i + 1

    V0 = (Varray[index]+Varray2[index])/2
    T0 = (Tarray[index]+Tarray2[index])/2
    Tf0 = (Tfarray[index]+Tfarray2[index])/2
    Pv0 = press_vap_calc(Tf0)
    Q0 = (Qarray[index]+Qarray2[index])/2
    E0 = (Earray[index]+Earray2[index])/2
    P0 = (Parray[index]+Parray2[index])/2

    if status == 0:
        lowerV = Varray
        lowerP = Parray
        lowerT = Tarray
        lowerTf = Tfarray
        lowerPv = Pvarray
        lowerM = Marray
        lowerE = Earray
        lowerangle = angledeg
        lowerQ = Qarray2
        lowerTf = Tfarray2
        upperV = Varray2
        upperP = Parray2
        upperT = Tarray2
        upperTf = Tfarray2
        upperPv = Pvarray2
        upperM = Marray2
        upperE = Earray2
        upperangle = angledeg2
        upperQ = Qarray2
        upperTf = Tfarray2
    else:
        lowerV = Varray2
        lowerP = Parray2
        lowerT = Tarray2
        lowerTf = Tfarray2
        lowerPv = Pvarray2
        lowerM = Marray2
        lowerE = Earray2
        lowerangle = angledeg2
        lowerQ = Qarray
        lowerTf = Tfarray
        upperV = Varray
        upperP = Parray
        upperT = Tarray
        upperTf = Tfarray
        upperPv = Pvarray
        upperM = Marray
        upperE = Earray
        upperangle = angledeg
        upperQ = Qarray
        upperTf = Tfarray

    outfoldername = 'STAGE{n}'.format(n=n)
    if os.path.exists(outfoldername):
        os.chdir(outfoldername)
    else:
        os.mkdir(outfoldername)
        os.chdir(outfoldername)

    np.savetxt('lowerangle.txt',lowerangle,delimiter=',')
    np.savetxt('lowerV.txt',lowerV,delimiter=',')
    np.savetxt('lowerP.txt',lowerP,delimiter=',')
    np.savetxt('lowerT.txt',lowerT,delimiter=',')
    np.savetxt('lowerPv.txt',lowerPv,delimiter=',')
    np.savetxt('lowerTf.txt',lowerTf,delimiter=',')
    np.savetxt('lowerQ.txt',lowerQ,delimiter=',')
    np.savetxt('lowerM.txt',lowerM,delimiter=',')
    np.savetxt('lowerE.txt',lowerE,delimiter=',')
    np.savetxt('upperangle.txt',upperangle,delimiter=',')
    np.savetxt('upperV.txt',upperV,delimiter=',')
    np.savetxt('upperP.txt',upperP,delimiter=',')
    np.savetxt('upperT.txt',upperT,delimiter=',')
    np.savetxt('upperPv.txt',upperPv,delimiter=',')
    np.savetxt('upperTf.txt',upperTf,delimiter=',')
    np.savetxt('upperQ.txt',upperQ,delimiter=',')
    np.savetxt('upperM.txt',upperM,delimiter=',')
    np.savetxt('upperE.txt',upperE,delimiter=',')

    loop_end = time.time()
    looptstr = 'time for stage {n} is {t} min'.format(n=n,t=(loop_end-loop_start)/60)
    print(looptstr)
    print('max M is {M}'.format(M=min(Marray[-1],Marray2[-1])))

    outfile = open('output.txt','a')
    outfile.write('\n'+'\n'+ 'STAGE {st} RESULTS'.format(st = n))
    outfile.write('\n' + 'angle split at {x}'.format(x = xstop))
    outfile.write('\n' + 'V0 = {V} m/s'.format(V=V0))
    outfile.write('\n' + 'T0 = {T} K'.format(T=T0))
    outfile.write('\n' + 'P0 = {P0}'.format(P0=P0))
    outfile.write('\n' + 'Tf0 = {Tf}'.format(Tf=Tf0))
    outfile.write('\n' + 'Mmax = {M}'.format(M=min(Marray[-1],Marray2[-1])))
    outfile.write('\n'+'E0 = {E} molecules/m^2'.format(E=E0))
    outfile.write('\n'+'Q0 = {Q} W/m^2'.format(Q=Q0))
    outfile.close()

    os.chdir('..')
    if min(Marray[-1],Marray2[-1])>0.99:
        outfile = open('output.txt','a')
        outfile.write('\n'+'Mach 1 achieved stage {n}'.format(n=n))
        outfile.close()
        break
    n = n + 1

time_end = time.time()
timestr = 'total time taken for code is {t} min'.format(t = (time_end-time_start)/60)
outfile = open('output.txt','a')
outfile.write('\n'+timestr)
outfile.close()
print('this is the end, my only friend')