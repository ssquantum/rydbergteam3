import numpy as np
import math

#Constants
h = 6.62607004 * 10**(-34)
hbar = h / (2*math.pi)
c = 299792458
kb = 1.38064852 * 10**(-23)
e = 1.60217662 * 10**(-19)
a0 = 5.29177 * 10**(-11)

#Energies of SPD states
S12State = np.array([6,7,8,9]) #N for the J=1/2 S State
S12WaveNo = np.array([0,18535.5286,24317.149400, 26910.6627]) #Wavenumber of state in cm^-1

P12State = np.array([6,7,8]) #N for the J=1/2 P State
P12WaveNo = np.array([11178.26815870,21765.348,25708.85473]) #Wavenumber of state in cm^-1

P32State = np.array([6,7,8]) #N for the J=3/2 P State
P32WaveNo = np.array([11732.3071041,21946.397,25791.508]) #Wavenumber of state in cm^-1

S12Energy = h * c * S12WaveNo * 100 #Energy of the states
P12Energy = h * c * P12WaveNo * 100 #Energy of the states
P32Energy = h * c * P32WaveNo * 100 #Energy of the states

#Finding polarisability of 6S1/2 State
N = 6
L = 'S'
J = 1/2

#Dipole Matrix Elements from ARC
Dipole6S12toP12 = np.array([3.174,0.197,0.057]) #in units of ea0
Dipole6S12toP32 = np.array([4.472,0.407,0.154]) #in units of ea0

#Polarisability
alphaelementP12 = np.divide((-(Dipole6S12toP12*e*a0)**2),(hbar*(P12Energy-S12Energy[N-6])))
alphaelementP32 = np.divide((-(Dipole6S12toP32*e*a0)**2),(hbar*(P32Energy-S12Energy[N-6])))
alpha = np.sum(alphaelementP12+alphaelementP32)
print('alpha = {}'.format(alpha))