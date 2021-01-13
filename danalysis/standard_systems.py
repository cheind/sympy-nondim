from .quantities import DimensionalSystem

def _add_LMT_derived_quantities(dimsys: DimensionalSystem) -> None:
    L,M,T = dimsys.L, dimsys.M, dimsys.T
    dimsys.F = L*M/T**2
    dimsys.Pressure = dimsys.F/L**2
    dimsys.Density = M/L**3
    dimsys.Torque = L**2*M*T**-2
    dimsys.Angle = dimsys.Unity

"""International System of Units (SI)"""
SI = DimensionalSystem(['L','M','T','A','K','Mol','Cd'])
"""Minimal International System of Units (SI) comprised by L,M,T"""
LMT = DimensionalSystem(['L','M','T'])

_add_LMT_derived_quantities(SI)
_add_LMT_derived_quantities(LMT)

