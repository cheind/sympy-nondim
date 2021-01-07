from .quantities import create_dimensional_system

"""SI base units length (L), mass (M) and time (T)"""
L,M,T = create_dimensional_system('SI', 'L','M','T')
"""Force"""
F = L*M/T**2
"""Pressure"""
P = F/L**2