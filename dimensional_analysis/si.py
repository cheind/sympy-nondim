import numpy as np
from .quantities import create_dimensional_system

"""SI base units length (L), mass (M) and time (T)"""
SIQuantity, (L,M,T) = create_dimensional_system('SI', 'L','M','T')

"""Dimensionless"""
one = SIQuantity(np.zeros(SIQuantity.NBASEDIMS))
# Derived quantities
"""Force"""
F = L*M/T**2
"""Pressure"""
pressure = F/L**2
density = M/L**3
torque = L**2*M*T**-2
angle = one
