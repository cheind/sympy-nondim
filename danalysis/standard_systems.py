from . import quantities as qt

def _add_LMT_derived_quantities(dimsys: qt.DimensionalSystem) -> None:
    L,M,T = dimsys.L, dimsys.M, dimsys.T
    dimsys.F = L*M/T**2
    dimsys.Velocity = L/T
    dimsys.Acceleration = dimsys.Velocity/T
    dimsys.Pressure = dimsys.F/L**2
    dimsys.Density = M/L**3
    dimsys.Torque = L**2*M*T**-2
    dimsys.DynamicViscosity = M*(L*T)**-1
    dimsys.Angle = dimsys.Unity

"""International System of Units (SI)"""
SI = qt.DimensionalSystem(['L','M','T','A','K','Mol','Cd'])
"""Minimal International System of Units (SI) comprised by L,M,T"""
LMT = qt.DimensionalSystem(['L','M','T'])

_add_LMT_derived_quantities(SI)
_add_LMT_derived_quantities(LMT)

