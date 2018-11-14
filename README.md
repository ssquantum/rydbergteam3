# rydbergteam3
Team 3 code for Charles Adam's Atomic and Molecular Interactions Grad course 2018

Calculate real trap depths for the Cs 133 6s groundstate and Rydberg states.

The dipole potential is U = - <d> E = -1/2 Re[a] |E|^2
  for expectation of the dipole operator <d>
  applied electric field E
  and polarizability a
  
We must find the atomic properties that give the polarizability for a given atomic state as a function of wavelength.
This will then be used to plot the potential as a function of position.

in the far off resonance limit the detuning is much greater than the natural linewidth and we can approximate
a = - D0^2 / hbar / Delta
for detuning Delta and dipole matrix element D0.
in the case of the Rydberg state there must be a sum over the contributions from all relevant transitions.
