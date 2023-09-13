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
    'path_stellar'  :   '/data/solarspectrum', # stellar spectrum data
    'path_input'    :   '/input', # where the input file is
    'path_output'   :   '/output'
    }
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
# >>>>>>>>>>>>>>>>>>>>>>>> ACTUAL CODE STARTS HERE <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
time0 = time.localtime()
time_start = time.time()

home = os.getcwd()
path_output = home+directories['path_output']
path_input = home+directories['path_input']
path_xsect = home+directories['path_xsect']
path_stellar = home+directories['path_stellar']

os.chdir(path_input)
with open(directories['input_name']) as input_txt:
    for line in input_txt:
        if 'RUN NAME' in line:
            dummyline = line
            dummystr = ': '
            run_name = dummyline.split(dummystr)[1]
            run_name = run_name[0:-1]

os.chdir(path_output)
os.chdir('output_'+run_name)
regexp = re.compile(r'Mach 1 achieved stage.*?([0-9.-]+)')
with open('output.txt') as input_txt:
    for line in input_txt:
        match = regexp.match(line)
        if match:
            current_stage = int(match.group(1))

lowerV,lowerP,lowerT,lowerPv,lowerM,lowerE,lowerangle,lowerQ,lowerTf = [],[],[],[],[],[],[],[],[]
upperV,upperP,upperT,upperPv,upperM,upperE,upperangle,upperQ,upperTf = [],[],[],[],[],[],[],[],[]
n = 1
while n <= current_stage:

    os.chdir('STAGE{n}'.format(n=n))

    angle1 = np.genfromtxt('lowerangle.txt')
    P1 = np.genfromtxt('lowerP.txt')
    T1 = np.genfromtxt('lowerT.txt')
    E1 = np.genfromtxt('lowerE.txt')
    M1 = np.genfromtxt('lowerM.txt')
    V1 = np.genfromtxt('lowerV.txt')
    Tf1 = np.genfromtxt('lowerTf.txt')
    Q1 = np.genfromtxt('lowerQ.txt')
    Pv1 = np.genfromtxt('lowerPv.txt')

    angle2 = np.genfromtxt('upperangle.txt')
    P2 = np.genfromtxt('upperP.txt')
    T2 = np.genfromtxt('upperT.txt')
    E2 = np.genfromtxt('upperE.txt')
    M2 = np.genfromtxt('upperM.txt')
    V2 = np.genfromtxt('upperV.txt')
    Tf2 = np.genfromtxt('upperTf.txt')
    Q2 = np.genfromtxt('upperQ.txt')
    Pv2 = np.genfromtxt('upperPv.txt')

    regexp1 = re.compile(r'angle split at.*?([0-9.-]+)')
    with open('output.txt') as input_txt:
        for line in input_txt:
            match = regexp1.match(line)
            if match:
                xstart = float(match.group(1))

    os.chdir('..')
    index = np.abs(np.array(angle1)-xstart).argmin()

    lowerangle = np.append(lowerangle,angle1[0:index])
    lowerV = np.append(lowerV,V1[0:index])
    lowerP = np.append(lowerP,P1[0:index])
    lowerT = np.append(lowerT,T1[0:index])
    lowerE = np.append(lowerE,E1[0:index])
    lowerTf = np.append(lowerTf,Tf1[0:index])
    lowerQ = np.append(lowerQ,Q1[0:index])
    lowerPv = np.append(lowerPv,Pv1[0:index])
    lowerM = np.append(lowerM,M1[0:index])

    upperangle = np.append(upperangle,angle2[0:index])
    upperV = np.append(upperV,V2[0:index])
    upperP = np.append(upperP,P2[0:index])
    upperT = np.append(upperT,T2[0:index])
    upperE = np.append(upperE,E2[0:index])
    upperTf = np.append(upperTf,Tf2[0:index])
    upperQ = np.append(upperQ,Q2[0:index])
    upperPv = np.append(upperPv,Pv2[0:index])
    upperM = np.append(upperM,M2[0:index])

    n+=1

angle = lowerangle
P = 0.5*np.add(lowerP,upperP)
T = 0.5*np.add(lowerT,upperT)
E = 0.5*np.add(lowerE,upperE)
M = 0.5*np.add(lowerM,upperM)
V = 0.5*np.add(lowerV,upperV)
Tf = 0.5*np.add(lowerTf,upperTf)
Q = 0.5*np.add(lowerQ,upperQ)
Pv = 0.5*np.add(lowerPv,upperPv)
M = 0.5*np.add(lowerM,upperM)

os.chdir('STAGE{n}'.format(n=current_stage+1))

superangle = np.genfromtxt('angle.txt')
superV = np.genfromtxt('V.txt')
superP = np.genfromtxt('P.txt')
superT = np.genfromtxt('T.txt')
superE = np.genfromtxt('E.txt')
superTf = np.genfromtxt('Tf.txt')
superQ = np.genfromtxt('Q.txt')
superPv = np.genfromtxt('Pv.txt')
superM = np.genfromtxt('M.txt')

angle = np.append(angle,superangle)
V = np.append(V,superV)
P = np.append(P,superP)
T = np.append(T,superT)
E = np.append(E,superE)
Tf = np.append(Tf,superTf)
Q = np.append(Q,superQ)
Pv = np.append(Pv,superPv)
M = np.append(M,superM)

os.chdir('..')
if os.path.exists('output_final'):
    os.chdir('output_final')
else:
    os.mkdir('output_final')
    os.chdir('output_final')

np.savetxt('angle.txt',angle,delimiter=',')
np.savetxt('T.txt',T,delimiter=',')
np.savetxt('P.txt',P,delimiter=',')
np.savetxt('V.txt',V,delimiter=',')
np.savetxt('E.txt',E,delimiter=',')
np.savetxt('Tf.txt',Tf,delimiter=',')
np.savetxt('Q.txt',Q,delimiter=',')
np.savetxt('M.txt',M,delimiter=',')

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(angle,P,linewidth=2,label='actual P')
plt.plot(angle,Pv,linewidth=2,linestyle='dashed',label='saturated P')
plt.ylabel('pressure [Pa]')
plt.legend()

plt.subplot(3,1,2)
plt.plot(angle,V,linewidth=2)
plt.ylabel('wind speed [m/s]')

plt.subplot(3,1,3)
plt.plot(angle,T,linewidth=2,label='atm T')
plt.plot(angle,Tf,linewidth=2,label='surf T')
plt.legend()
plt.ylabel('temperature [K]')
plt.xlabel('angular distace from substellar point')
plt.savefig('PVT.pdf',dpi=600)
plt.close()

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(angle,const['m']*np.array(E),linewidth=2)
plt.ylabel('evap rate [kg/m^2/s]')

plt.subplot(3,1,2)
plt.plot(angle,M,linewidth=2)
plt.ylabel('mach speed')

plt.subplot(3,1,3)
plt.plot(angle,Q,linewidth=2)
plt.ylabel('energy flux [W/m^2]')
plt.xlabel('angular distace from substellar point')
plt.savefig('EMQ.pdf',dpi=600)
plt.close()








