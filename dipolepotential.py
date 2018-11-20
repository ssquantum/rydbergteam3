""" Adapted Code from Stefan's dipoleint.py, to look at the potential of the guassian laser 
20/11/18 Stefan: plot the potential in the x-z plane with contours

TO DO: fit a Gaussian to the potential in the x and z directions
use this to get a trap width in the x and z directions respectively
then see how these widths vary with beamwaist and wavelength
Also:
    make the units more explicit - at the moment they're hard-coded in several places
""" 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import DipoleInt as DI
from DipoleInt import c, eps0, hbar, a0, e, kB
from scipy.optimize import curve_fit


def fitgauss(x, A, x0, sig, y0):
        """Gaussian function centred at x0 with amplitude A, 1/e^2 width sig
        and offset y0.
        Used to estimate the width of the trap."""
        return A * np.exp(-(x-x0)**2 /2. /sig**2) /sig /2. /np.pi + y0
        
        
def plotcontour(x, z, u):
    """make a contour plot of the dipole potential in the x-z plane"""
    xmax = np.amax(x)
    zmax = np.amax(z)
    
    xaxis, xmax, zaxis, zmax = np.array([x, xmax, z, zmax])*1e6                 # convert from m to microns
    
    plt.figure()
    plt.title("Fig. 2 - Potential of Gaussian laser in x-z plane (K)",y=1.08)
    plt.contour(zaxis, xaxis, u)
    plt.imshow(u, 
               extent = (-zmax, zmax, -xmax, xmax),
               origin = 'lower',
               cmap = 'Blues',
               aspect = 'auto')
    plt.colorbar()
    plt.xlabel("z ($\mu$m)")
    plt.ylabel("x ($\mu$m)")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xticks(np.linspace(-zmax,zmax,7))
    plt.yticks(np.linspace(-xmax,xmax,7))
    plt.tight_layout()
    plt.show()

def applyfit(x, u, pguess, verb=0):
    """Use scipy curve_fit to get parameters from a Gaussian fit
    to the dipole potential"""
    popt, pcov = curve_fit(fitgauss, x, u,
                    p0 = pguess,
                    bounds = ((-np.inf, -np.inf, 0, -np.inf),np.inf))

    wid = popt[2]           # width estimate from Gaussian 1/e^2 width
    errwid = pcov[2,2]**0.5 # error in the width

    if verb > 0:
        # check the fit by plotting a graph
        plt.figure()
        x *= 1e6    # convert units to microns
        plt.plot(x, u, '.')
        plt.plot(x, fitgauss(x, *popt), '--')
        plt.ylabel("Dipole Potential (K)")
        plt.xlabel("Position ($\mu$m)")
        plt.show()

        print("The width is %.3g +/- %.1g microns"%(wid*1e6,errwid*1e6))

    return (popt, pcov)


if __name__ == "__main__":
    # Plot a slice of the laser's potential in the XZ plane and look for trap depth.    
    
    # Parameters to use in laser
    wavelength = 980e-9                 # wavelength of laser in m
    omega = 2*np.pi * c / wavelength    # angular frequency in rad/s   
    power = 1                           # power in watts
    beamwaist = 1e-6                    # beam waist in m
    bprop = [wavelength, power, beamwaist]
    
    # dipole matrix elements for: 6S1/2 -> 6P1/2, 6P3/2
    CsD0s = np.array([3.19, 4.48]) * e * a0 

    # resonances
    Cswavelengths = np.array([894.6, 852.3]) * 1e-9     # wavelength in m
    Csomega0 = 2*np.pi*c /Cswavelengths                 # angular frequency in rad/s
    Csgamma = 2*np.pi*np.array([4.575, 5.234])*1e6      # natural linewidth in rad/s

    # create the dipole object 
    Cs_d = DI.dipole(133, 7/2., CsD0s, bprop, Csomega0, Csgamma)

    npts = 200                       # number of sample points on one axis
    xlim = 2*beamwaist               # x-axis limits in m
    xaxis = np.linspace(-xlim, xlim, npts)
    zlim = 5*beamwaist               # z-axis limits in m
    zaxis = np.linspace(-zlim, zlim, npts)
    ypos = 0                         # position on y axis in m
    
    # calculate the potential in the x-z plane
    u = np.zeros((len(xaxis), len(zaxis)))
    for ix, xpos in enumerate(xaxis):
        u[ix,:] = Cs_d.U(xpos, ypos, zaxis)
        
    u /= kB                          # convert from Joules to Kelvin
    
    # plot the potential
    plotcontour(xaxis, zaxis, u)
    
    # calculate the trap depth
    tdepth = np.amax(u) - np.amin(u)
    
    # theoretical trap depth from K J Weatherill's thesis:
    U0 = Cs_d.polarizability() * power / eps0 / c / np.pi / beamwaist**2
    
    print('Difference between min and max = %.3g K'%tdepth)
    print('Kevs thesis formula predicts trap depth = %.3g K'%(U0/kB))
    
    # fit Gaussian to get estimate of trap widths
    poptx, pcovx = applyfit(xaxis, Cs_d.U(xaxis,0,0)/kB,
                            (-tdepth, 0, beamwaist, 0), 1)
    poptz, pcovz = applyfit(zaxis, Cs_d.U(0,0,zaxis)/kB,
                            (-tdepth, 0, beamwaist, 0), 1)

    
    # see how trap widths vary with wavelength and beamwaist
    
    
