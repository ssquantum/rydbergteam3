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
wavelength = np.logspace(-7, -5, 10000)   # wavelength of laser in m
omega = 2*np.pi * c / wavelength          # angular frequency in rad/s
power = 1                                 # power in watts
beamwaist = 1e-6                          # beam waist in m
bprop = [wavelength, power, beamwaist]    # collect beam properties


# dipole matrix elements for: 6S1/2 -> 6P1/2, 6S1/2 -> 6P3/2
CsD0s = np.array([3.19, 4.48]) * e * a0 

# resonances
Cswavelengths = np.array([894.6, 852.3]) * 1e-9 # wavelength in m
Csomega0 = 2*np.pi*c /Cswavelengths             # angular frequency in rad/s
Csgamma = 2*np.pi*np.array([4.575, 5.234])      # natural linewidth in rad/s

# create the dipole object and calculate the polarizability
Cs_d = DI.dipole(133, 7/2., CsD0s, bprop, Csomega0, Csgamma)
Csalpha = Cs_d.polarizability()            # polarizability without RWA
Csalpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)

# compare the effect of including the D1, D2 transition:
Cs_D1 = DI.dipole(133, 7/2., CsD0s[0], bprop, Csomega0[0], Csgamma[0])
D1alpha = Cs_D1.polarizability()
D1alpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)

Cs_D2 = DI.dipole(133, 7/2., CsD0s[1], bprop, Csomega0[1], Csgamma[1])
D2alpha = Cs_D2.polarizability()
D2alpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)
    
plt.figure()
plt.title("Polarizability for ground state $^{133}$Cs")
plt.semilogx(wavelength, Csalpha, label="Both D1 and D2 transitions")
plt.semilogx(wavelength, D1alpha, label="2-level model with D1")
plt.semilogx(wavelength, D2alpha, label="2-level model with D2")
plt.xlabel("Wavelength (m)")
plt.ylabel("Polarizability (a$_0$$^3$)")
plt.ylim((-1e3,1e3))
plt.tight_layout()
plt.legend()
plt.show()