"""Stefan Spence 13.11.18
Grad Course: Atomic and Molecular Interactions

Calculate the spectrum of noe and two Rydberg atoms in an optical tweezer.

1) Formulate the equations for Gaussian beam propagation.
2) Look at the dipole interaction and the induced dipole moment

14.11.18 add in dipole potential
calculate the polarizability and compare the 2-level model to including other transitions
"""
import numpy as np
import matplotlib.pyplot as plt

# global constants:
c = 299792458        # speed of light in m/s
eps0 = 8.85419e-12   # permittivity of free space in m^-3 kg^-1 s^4 A^2
hbar = 1.0545718e-34 # in m^2 kg / s
a0 = 5.29177e-11     # Bohr radius in m
e = 1.6021766208e-19 # magnitude of the charge on an electron in C


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
        self.omega0 = resonant_frequency      # resonant frequency
        self.Delta = 2*np.pi*c/self.field.lam - resonant_frequency # detuning (in rad/s)
        self.gam   = decay_rate               # spontaneous decay rate
        self.D0    = dipole_matrix_element    # D0 = -e <a|r|b> for displacement r along the polarization direction
        
        # from these properties we can deduce:
        self.rabi  = dipole_matrix_element * self.field.E0 / hbar        # Rabi frequency in rad/s
        self.s     = 0.5 * self.rabi**2 / (self.Delta**2 + 0.25*self.gam**2)
        self.ust   = self.Delta * self.s / self.rabi / (1. + self.s)     # in phase dipole moment
        self.vst   = self.gam * 0.5 / self.rabi * self.s / (1. + self.s) # quadrature dipole moment

    def U(self, x, y, z):
        """Return the potential from the dipole interaction U = -<d>E = -1/2 Re[alpha] E^2
        where in the FORT limit we can take alpha = -D0^2/hbar /Delta for detuning Delta >> than the natural linewidth"""
        return self.D0**2 / 2 / hbar / self.Delta *np.abs( self.field.amplitude(x,y,z) )**2
    
    def RWApolarizability(self):
        """Return the real part of the polarizability with the RWA
        Note the factor of 1/3 is from averaging over spatial directions"""
        return -self.D0**2 /3. /hbar * self.Delta / (self.Delta**2 - self.gam**2/4.)
    
    def polarizability(self):
        """Return the real part of the polarizability with the RWA
        Note the factor of 1/3 is from averaging over spatial directions"""
        omega = 2*np.pi * c/ self.field.lam # laser angular frequency
        return -self.D0**2 /3. /hbar * (self.Delta / (self.Delta**2 - self.gam**2/4.) + (self.omega0 + omega) / ((self.omega0 + omega)**2 - self.gam**2/4.))

        
if __name__ == "__main__":
    # plot the dipole moment components as a function of detuning
    # in order to make the units nice, normalise the detuning to the spontaneous decay rate
    # and take the rabi frequency to be equal to the spontaneous decay rate

    # need a large number of points in the array to resolve resonances
    wavelength = np.logspace(-7, -5, 10000)   # wavelength of laser in m
    omega = 2*np.pi * c / wavelength  # angular frequency in rad/s
    power = 1 # power in watts
    beamwaist = 1e-6 # beam waist in m
    fprop = [wavelength, power, beamwaist]
    
    D0guess = 3.584e-29 # dipole matrix element for 87Rb D2 transition in Cm
    omega0 = 2*np.pi*c / 780.2e-9 # estimate of resonant frequeny in rad/s for 87Rb D2 transition
    gamma = 2*np.pi*6.07e6 # estimate of natural linewidth in rad/s
    
    beam1 = Gauss(wavelength, power, beamwaist)
    d1 = dipole(87, 3/2., D0guess, fprop, omega0, gamma)

    # get a graph of polarizability with RWA
    alpha1 = d1.RWApolarizability()
    alpha1 *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)

    # get a graph of polarizability without RWA
    alpha2 = d1.polarizability()
    alpha2 *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)
    
    # compare K. Weatherill's polarizability
    # the values are obtained from K J Weatherill's thesis, Fig 2.5, available from Durham etheses
    As = np.array([1.37e5, 8.91e4, 3.96e5, 2.89e5, 1.77e6, 1.5e6, 3.81e7, 3.61e7])
    omegas = 2*np.pi*c / np.array([334.87, 335.08, 358.71, 359.16, 420.18, 421.55, 780.27, 794.76]) / 1e-9
    kwalphas = 3*np.pi * eps0 * c**3 * np.array([sum(As/omegas**3 * (1/(omegas-x) + 1/(omegas+x))) for x in omega])
    kwalphas *= 1 / 4. / np.pi / eps0 /(a0)**3 # convert to units of Bohr radius cubed (cgs)
    
    plt.figure()
    plt.title("Polarizability for ground state $^{87}$Rb")
    plt.semilogx(wavelength, alpha1, label="2-level model with RWA")
    plt.semilogx(wavelength, alpha2, label="2-level model without RWA")
    plt.semilogx(wavelength, kwalphas, label="Added Transitions without RWA")
    plt.xlabel("Wavelength (m)")
    plt.ylabel("Polarizability (a$_0$$^3$)")
    plt.ylim((-1e3,1e3))
    plt.tight_layout()
    plt.legend()
    plt.show()

    # ----------------------------
    # now try and calculate the polarizability for Cs133
    CsD0 = 3.174 * e * a0 # dipole matrix element for Cs 6S1/2 -> 6P1/2
    Csomega0 = 2*np.pi*c /894.6e-9 # estimate of resonant frequeny in rad/s for Cs D1 transition
    Csgamma = 2*np.pi*4.5612 # estimate of natural linewidth in rad/s
    
    Cs_d = dipole(133, 7/2., D0guess, fprop, omega0, gamma)

    Csalpha = Cs_d.polarizability()       # polarizability without RWA
    Csalpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)
    CsRWAalpha = Cs_d.RWApolarizability() # polarizability with RWA
    CsRWAalpha *= 1 / 4. / np.pi / eps0  /(a0)**3 # convert to units of Bohr radius cubed (cgs)
    
    plt.figure()
    plt.title("Polarizability for ground state $^{133}$Cs")
    plt.semilogx(wavelength, CsRWAalpha, label="2-level model with RWA")
    plt.semilogx(wavelength, Csalpha, label="2-level model without RWA")
    plt.xlabel("Wavelength (m)")
    plt.ylabel("Polarizability (a$_0$$^3$)")
    plt.ylim((-1e3,1e3))
    plt.tight_layout()
    plt.legend()
    plt.show()
