"""
Required functions for 'HyperfineSolvers' Module.

Author: James Vandeleur
Contact: jamesvandeleur@proton.me
UQ Physics Honours Project 2023
"""

### Constants ###
global const c = 137.0359895
global const α = 1/c # fine structure constant

## Masses in atomic units (me = 1 a.u.)
me = 9.1093837015e-31 # kg
global const mmu = 1.883531627e-28 / me # muon mass a.u.
global const mp = 1.67262192369e-27/ me  # proton mass a.u.

### UNIT Conversions ###
# Energy
global const Eh_to_Hz = 6.579683920502e15
global const Eh_to_eV = 27.211386245988
global const Hz_to_eV = 4.13566769692386e-15

# Length
global const fm_to_au = 1/(5.29177210903e4)