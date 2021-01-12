import numpy as np
from .quantities import Quantity

LMTQuantity = Quantity.create_type(
    'LMTQuantity', ['L','M','T']
)

L,M,T = LMTQuantity.basevars()
unity = LMTQuantity.unity()

F = L*M/T**2
pressure = F/L**2
density = M/L**3
torque = L**2*M*T**-2
angle = unity