"""Stefan Spence 13.11.18
Grad Course: Atomic and Molecular Interactions

Calculate the spectrum of noe and two Rydberg atoms in an optical tweezer.

1) Formulate the equations for Gaussian beam propagation.
2) Look at the dipole interaction and the induced dipole moment
"""
import numpy as np
import matplotlib.pyplot as plt

# global constants:
c = 299792458        # speed of light in m/s
eps0 = 8.85419e-12   # permittivity of free space in m^-3 kg^-1 s^4 A^2
hbar = 1.0545718e-34 # in m^2 kg / s


class Gauss:
    """Properties and associated equations of a Gaussian beam"""
    def __init__(self, wavelength, power, beam_waist, polarization=(0,0,1)):
        self.lam = wavelength    # wavelength of the laser light (in metres)
        self.P   = power         # total power of the beam (in Watts)
        self.w0  = beam_waist    # the beam waist defines the laser mode (in metres)
        self.ehat= polarization  # the direction of polarization (assume linear)
        # note: we will mostly ignore polarization since the induced dipole 
        # moment will be proportional to the direction of the field
        
        # assume that the beam waist is positioned at z0 = 0
        
        # from these properties we can deduce:
        self.zR = np.pi * beam_waist**2 / wavelength # the Rayleigh range
        self.E0 = 2/beam_waist * np.sqrt(power / eps0 / c / np.pi) # field amplitude at the origin
        self.k  = 2 * np.pi / wavelength             # the wave vector
        
    def amplitude(self, x, y, z):
        """Calculate the amplitude of the Gaussian beam at a given position"""
        rhosq = x**2 + y**2       # radial coordinate squared
        q     = z - 1.j * self.zR # complex beam parameter
        
        # Gaussian beam equation (see Optics f2f Eqn 11.7)
        return self.zR /1.j /q * self.E0 * np.exp(1j * self.k * z) * np.exp(1j * self.k * rhosq / 2. / q)
        
        
class dipole:
    """Properties and equations of the dipole interaction between atom and field"""
    def __init__(self, mass, nuclear_spin, dipole_matrix_element, 
                    field_properties, resonant_frequency, decay_rate):
        self.m     = mass * 1.67e-26          # mass of the atom in kg
        self.I     = nuclear_spin
        self.field = Gauss(*field_properties) # combines all properties of the field
        self.Delta = 2*np.pi*c/self.field.lam - resonant_frequency # detuning (in rad/s)
        self.gam   = decay_rate               # spontaneous decay rate
        self.D0    = dipole_matrix_element    # D0 = -e <a|r|b> for displacement r along the polarization direction
        
        # from these properties we can deduce:
        self.rabi  = dipole_matrix_element * self.field.E0 / hbar        # Rabi frequency in rad/s
        self.s     = 0.5 * self.rabi**2 / (self.Delta**2 + 0.25*self.gam**2)
        self.ust   = self.Delta * self.s / self.rabi / (1. + self.s)     # in phase dipole moment
        self.vst   = self.gam * 0.5 / self.rabi * self.s / (1. + self.s) # quadrature dipole moment
        

class potential:

    def __init__(self, beam, dipole_param):
        self.G = Gauss(beam)
        self.D = dipole(dipole_param)

    def U(self, x, y, z):
        """Return the potential from the dipole interaction U = -<d>E = -1/2 Re[alpha] E^2
        where in the FORT limit we can take alpha = -D0^2/hbar /Delta for detuning Delta >> than the natural linewidth"""
        return self.D.D0**2 / 2 / hbar / self.D.Delta *np.abs( self.G.amplitude(x,y,z) )**2
        
if __name__ == "__main__":
    # plot the dipole moment components as a function of detuning
    # in order to make the units nice, normalise the detuning to the spontaneous decay rate
    # and take the rabi frequency to be equal to the spontaneous decay rate
    x = np.linspace()
    y = x
    z = np.linsapce()

    wavelength = 980e-9  # wavelength of laser in m
    power = 1 # power in watts
    beamwaist = 1e-6 # beam waist in m

    beam1 = Gauss(wavelength, power, beamwaist)
    d1 = dipole(133, 
