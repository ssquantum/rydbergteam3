""" Adapted Code from Stefan's dipoleint.py, to look at the potential of the guassian laser """ 

import numpy as np
import matplotlib.pyplot as plt


# global constants:

c = 299792458        # speed of light in m/s
eps0 = 8.85419e-12   # permittivity of free space in m^-3 kg^-1 s^4 A^2
hbar = 1.0545718e-34 # in m^2 kg / s
a0 = 5.29177e-11     # Bohr radius in m
kB = 1.38064852e-23  # Boltzmann's constant in m^2 kg s^-2 K^-1
hz2kel = 2.0836612e-10 # Kelvin-Hertz relationship


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
        
    def U(self, x, y, z):
        """Return the potential from the dipole interaction U = -<d>E = -1/2 Re[alpha] E^2
        where in the FORT limit we can take alpha = -D0^2/hbar /Delta for detuning Delta >> than the natural linewidth"""
        return self.D0**2 / 2 / hbar / self.Delta *np.abs( self.field.amplitude(x,y,z) )**2


    def reala(self):
        """Return the real part of the polarizability """
        return -self.D0**2 / hbar * self.Delta / (self.Delta**2 - self.gam**2/4.)



class Potential:
    """Laser Potential test-ground"""
    def __init__(self, beam_waist, power, wavelength, dipole_matrix_element, resonant_frequency, field_properties,decay_rate):
        self.w0 = beam_waist    # the beam waist defines the laser mode (in metres)
        self.field = Gauss(*field_properties) # combines all properties of the field
        self.D0    = dipole_matrix_element    # D0 = -e <a|r|b> for displacement r along the polarization direction
        self.Delta = 2*np.pi*c/self.field.lam - resonant_frequency # detuning (in rad/s)
        self.zR = np.pi * beam_waist**2 / wavelength # the Rayleigh range
        self.gam = decay_rate # spont. decay rate
        self.alpha = -self.D0**2 / hbar * self.Delta / (self.Delta**2 - self.gam**2/4.) # alpha
        self.u0 = - (self.alpha * power) / (eps0 * c * np.pi * (self.w0 ** 2)) * (1/kB)# Kev's thesis (2.11)
        

    def laserpot(self, x, y, z):        
        rhosq = x**2 + y**2       # radial coordinate squared
        omeg = self.w0 * np.sqrt(1 + z / self.zR)    # beam width as a function of z
        return  (1 / kB) * (self.D0 ** 2 / (2 * hbar)) * (1 / self.Delta) * ((4 * power) / (np.pi * self.w0 * c * eps0)) * ((self.w0 / omeg) ** 2) * np.exp(-2 * rhosq / (omeg ** 2))
        


if __name__ == "__main__":
    # Plot a slice of the laser's potential in the XZ plane and look for trap depth.    
    
    # Parameters to use in laser
    wavelength = 9.4e-8 # wavelength of laser in m
    omega = 2*np.pi * c / wavelength  # angular frequency in rad/s   
    power = 1 # power in watts
    beamwaist = 1e-6 # beam waist in m
    fprop = [wavelength, power, beamwaist]
    D0guess = 3.584e-29 # dipole matrix element for 87Rb D2 transition in Cm
    omega0 = 2*np.pi*c / 780.2e-9 # estimate of resonant frequeny in rad/s for 87Rb D2 transition
    gamma = 2*np.pi*6.07e6 # estimate of natural linewidth in rad/s
    

    # Potential of Laser
    beam2 = Potential(beamwaist,power,wavelength,D0guess,omega0,fprop,gamma)
    zmax = 1e-5 # Length in z direction of laser
    npts = 400 # Number of grid points
    X = Y = np.linspace(-2*beamwaist, 2*beamwaist, npts) # X,Y vectors for calc
    Z = np.linspace(0, zmax, npts) # Z vector for calc
    x, y, z = np.meshgrid(X, Y, Z) # Stitching into mesh
    test1 = beam2.laserpot(x,y,z) # Run laser
    beamXZ = test1[:,1,:] # Extract x-z components of 3D array
    beamYZ = test1[1,:,:] # Extract y-z components of 3D array
    beamXY = test1[:,:,1] # Extract x-y components of 3D array
    
    
    # Fitting a Gaussian
    def Gaussfit(xdat, mean, sigma):
        return np.exp(-(xdat-mean)**2 / (2 * sigma**2))

    for j in np.linspace(1,npts-1,10):
        xdat = beamXZ[:,j]
        ydat = beamXZ[j,:]
        
        mean = sum(xdat * ydat) / sum(ydat)
        sigma = np.sqrt(sum(ydat * (xdat - mean)**2) / sum(ydat))
    
        plt.plot(xdat, Gaussfit(xdat, mean, sigma), color = [0, 0 ,j/npts], label=(np.round(j)))
        #plt.legend()
        plt.title('Fig. 1 - Guassian fit of potential in the XZ plane')
        plt.xlabel('Potential')
        plt.ylabel('PDF')
  
    plt.show()  
    
    
    # Trap depth
    tdepth_calc = np.max(beamXZ) - np.min(beamXZ)
    tdepth = beam2.u0 * hz2kel
    print('Difference between min and max =', tdepth_calc)
    print('Kevs thesis formula predicts trap depth =', tdepth)


    # Plotting
    plt.figure()
    plt.pcolor(Z,X,beamXZ)
    plt.colorbar()
    plt.title("Fig. 2 - Potential of Gaussian laser in x-z plane (K)",y=1.08)
    plt.xlabel("z (m)")
    plt.ylabel("x (m)")
    plt.ylim(-2*beamwaist,2*beamwaist)
    plt.xlim(0,zmax)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xticks(np.linspace(0,max(Z),6))
    plt.yticks(np.linspace(-max(X),max(X),6))
    plt.tight_layout()
    plt.show()
