import numpy as np
from .quantities import create_dimensional_system

"""SI base units length (L), mass (M) and time (T)"""
SIQuantity, (L,M,T) = create_dimensional_system('SI', 'L','M','T')

"""Dimensionless"""
ONE = SIQuantity(np.zeros(SIQuantity.NBASEDIMS))
# Derived quantities
"""Force"""
F = L*M/T**2
"""Pressure"""
P = F/L**2