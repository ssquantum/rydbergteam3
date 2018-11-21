"""Stefan Spence 19.11.18
Grad Course: Atomic and Molecular Interactions
Calculate the spectrum of noe and two Rydberg atoms in an optical tweezer.

Calculate the polarizability of the 133 Caesium groundstate as a function of wavelength
Include the D1 and D2 transitions in calculations
"""
import numpy as np
import matplotlib.pyplot as plt
import DipoleInt as DI
from DipoleInt import c, eps0, hbar, a0, e

# laser beam properties
wavelength = np.logspace(-6.3, -5.5, 100000)   # wavelength of laser in m
omega = 2*np.pi * c / wavelength          # angular frequency in rad/s
power = 1                                 # power in watts
beamwaist = 1e-6                          # beam waist in m
bprop = [wavelength, power, beamwaist]    # collect beam properties


# dipole matrix elements for: 6S1/2 -> 6P1/2, 6P3/2, 7P1/2, 7P3/2
# data for the 7P states was found using atomcalc and linewidths from
# Vasilyev, AA & Savukov, I & S. Safronova, M & Berry, H. (2001). Measurement of the 6s - 7p transition probabilities in atomic cesium and a revised value for the weak charge Q_W. Physical Review A. 66. 10.1103/PhysRevA.66.020101. 
CsD0s = np.array([3.19, 4.48, 0.197, 0.407]) * e * a0 

# resonances
Cswavelengths = np.array([894.6, 852.3, 459.4, 455.7]) * 1e-9 # wavelength in m
Csomega0 = 2*np.pi*c /Cswavelengths             # angular frequency in rad/s
Csgamma = 2*np.pi*np.array([4.575, 5.234, 0.1263, 0.2922])*1e6      # natural linewidth in rad/s

# create the dipole object and calculate the polarizability
# apart from the resonance at 456nm, including the 7p transitions is indistinguishable
# instead we will compare the use of RWA with D1 and D2 transitions
Cs_d = DI.dipole(133, 7/2., CsD0s[:2], bprop, Csomega0[:2], Csgamma[:2])
Csalpha = Cs_d.RWApolarizability()            # polarizability with RWA
Csalpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)

# compare the effect of including the D1, D2 transition:
Cs_Dboth = DI.dipole(133, 7/2., CsD0s[:2], bprop, Csomega0[:2], Csgamma[:2])
D1alpha = Cs_Dboth.polarizability()
D1alpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)

# only the D2 transition
Cs_D2 = DI.dipole(133, 7/2., CsD0s[1], bprop, Csomega0[1], Csgamma[1])
D2alpha = Cs_D2.polarizability()
D2alpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)
    
plt.figure()
plt.title("Polarizability for ground state $^{133}$Cs")
# Cs_d and Cs_Dboth are indistinguishable except for the resonances at 456nm
# so four our laser working at 980nm, we can just use the D1 and D2 transitions.
plt.semilogx(wavelength, Csalpha, label="D1 and D2 transitions with RWA")
plt.semilogx(wavelength, D1alpha, label="D1 and D2 transitions")
plt.semilogx(wavelength, D2alpha, label="Just D2 transition")
ymax = 2e3
plt.plot([980e-9]*2, [-ymax, ymax], 'k--', label="Laser at 980nm")
plt.xlabel("Wavelength (m)")
plt.ylabel("Polarizability (a$_0$$^3$)")
plt.ylim((-ymax,ymax))
plt.tight_layout()
plt.legend()
plt.show()