import numpy as np
from .quantities import create_dimensional_system

"""SI base units length (L), mass (M) and time (T)"""
system = create_dimensional_system('SI', 'L','M','T')
L,M,T = system.base_dimensions
unity = system.unity

F = L*M/T**2
pressure = F/L**2
density = M/L**3
torque = L**2*M*T**-2
angle = unity
