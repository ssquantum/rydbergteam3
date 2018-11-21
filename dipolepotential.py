""" Adapted Code from Stefan's dipoleint.py, to look at the potential of the guassian laser 
20/11/18 Stefan: plot the potential in the x-z plane with contours

21/11/18 fit a Gaussian to the potential in the x and z directions
use this to get a trap width in the x and z directions respectively
then see how these widths vary with beamwaist and wavelength

TO DO:
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
        
        
def plotcontour(xlim, zlim, npts, w0, verb=0):
    """make a contour plot of the dipole potential in the x-z plane
    xlim is the maximum distance covered in the x direction, as for zlim
    npts is the number of points to take along each axis
    w0 is the beam waist of the laser"""
    
    xaxis = np.linspace(-xlim, xlim, npts) # positions along x in m
    zaxis = np.linspace(-zlim, zlim, npts) # positions along z in m
    ypos = 0                         # position on y axis in m
    
    # calculate the potential in the x-z plane
    u = np.zeros((len(xaxis), len(zaxis)))
    for ix, xpos in enumerate(xaxis):
        u[ix,:] = Cs_d.U(xpos, ypos, zaxis)
        
    u /= kB                          # convert from Joules to Kelvin
    
    # convert from m to microns:
    xaxis, xmax, zaxis, zmax = np.array([xaxis, xlim, zaxis, zlim])*1e6                 
    
    if verb > 0:
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
        
    return u

def applyfit(x, u, pguess, verb=0):
    """Use scipy curve_fit to get parameters from a Gaussian fit
    to the dipole potential"""
    # fit is only valid around the centre point
    popt, pcov = curve_fit(fitgauss, x, u,
                    p0 = pguess,
                    bounds = ((-np.inf, -np.inf, 0, -np.inf),np.inf))

    wid = popt[2]           # width estimate from Gaussian 1/e^2 width
    errwid = pcov[2,2]**0.5 # error in the width

    if verb > 0:
        # check the fit by plotting a graph
        plt.figure()
        x_inum = x  *1e6          # convert units 
        plt.plot(x_inum, u, '.')
        plt.plot(x_inum, fitgauss(x, *popt), '--')
        plt.ylabel("Dipole Potential (K)")
        plt.xlabel("Position ($\mu$m)")
        plt.text(min(x_inum), min(u), "$w_x = %.3g \pm %.1g \mu$m"%(wid*1e6, errwid*1e6))
        plt.show()

    return (popt, pcov)
    
    
    
def AspectRatio(lams, w0s, P, D0s, omega0s, gammas, npts=200, verb=0):
    """Get the aspect ratio for a given range of wavelengths, lams,
    and beamwaists, w0s.
    npts defines the number of positions sampled along each axis.
    The beam power is P, D0s are dipole matrix elements,
    omega0s are resonant frequencies
    gammas are the natural linewidths"""
    ar = np.zeros((np.size(lams),np.size(w0s)))  # aspect ratios
    
    for i in range(np.size(lams)):
        for j in range(np.size(w0s)):
            # create dipole object with new properties
            beamprop = [lams[i], P, w0s[j]]
            newdipole = DI.dipole(133, 7/2, D0s, beamprop, omega0s, gammas)
            
            # make sensible axes
            xlim = 2*w0s[j]               # x-axis limits in m
            xaxis = np.linspace(-xlim, xlim, npts)
            zlim = 5*w0s[j]               # z-axis limits in m
            zaxis = np.linspace(-zlim, zlim, npts)
            
            # fit Gaussian to get estimate of trap widths
            U0guess = min(newdipole.U(xaxis,0,0)/kB)
            poptx, pcovx = applyfit(xaxis, newdipole.U(xaxis,0,0)/kB,
                                    (U0guess, 0, w0s[j], 0), 
                                    verb=verb-1)
            poptz, pcovz = applyfit(zaxis, newdipole.U(0,0,zaxis)/kB,
                                    (U0guess, 0, w0s[j], 0), 
                                    verb=verb-1)
            # aspect ratio:
            ar[j,i] = poptz[2] / poptx[2]
    
    if verb > 0:
        plt.figure()
        plt.imshow(ar, extent=(min(lams)*1e6,max(lams)*1e6,min(w0s)*1e6,max(w0s)*1e6),
                        cmap='Blues', origin='lower', aspect='auto')
        plt.colorbar()
        plt.xlabel("Wavelength ($\mu$m)")
        plt.ylabel("Beam Waist ($\mu$m)")
        plt.title("Aspect Ratio of the Dipole Trap: $w_z / w_x$")
        plt.show()
    
    return ar
    
    
    
def getTrapDepth(u, alpha, P, w0, verb=0):
    """Calculate the trap depth from comparing the min and max of the potential
    compare to the analytical calculation from KJ Weatherill's thesis"""
    trapdepth = np.amax(u) - np.amin(u)
    
    if verb > 0:
        print('Difference between min and max of potential = %.3g K'%trapdepth)
        
        # theoretical trap depth from K J Weatherill's thesis:
        U0 = alpha * P / eps0 / c / np.pi / w0**2    
        print('Kevs thesis formula predicts trap depth = %.3g K'%(U0/kB))
    return trapdepth


if __name__ == "__main__":
    # Plot a slice of the laser's potential in the XZ plane and look for trap depth.    
    
    # Parameters to use in laser
    wavelength = 980e-9                 # wavelength of laser in m
    omega = 2*np.pi * c / wavelength    # angular frequency in rad/s   
    power = 1e-3                        # power in watts
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

    
    # plot the potential
    u = plotcontour(xlim = 2*beamwaist,       # x-axis limits in m
                    zlim = 5*beamwaist,       # z-axis limits in m
                    npts = 200,               # number of points to take on each axis
                    w0 = beamwaist,           # beam waist in m
                    verb=1)
    
    # get the trap depth
    tdepth = getTrapDepth(u, Cs_d.polarizability(), power, beamwaist, verb=1)
    
    """
    # see how the aspect ratio varies with wavelength and beamwaist
    # keep the wavelength above 895nm otherwise polarizability changes sign
    # note: this takes a long time to run because of the many times a fit has to be made
    wavelengths = np.linspace(9e-7, 5e-6, 50) # wavelengths in m
    beamwaists = np.linspace(5e-7, 5e-6, 50)  # beam waists in m
    aspectratios = AspectRatio(wavelengths, beamwaists, power, CsD0s, Csomega0, Csgamma, verb=1)
    """