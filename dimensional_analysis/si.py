import numpy as np
from .quantities import Quantity

SIQuantity = Quantity.create_type('SIQuantity', 'LMT')

L,M,T = SIQuantity.basevars()
unity = SIQuantity.unity()

F = L*M/T**2
pressure = F/L**2
density = M/L**3
torque = L**2*M*T**-2
angle = unity
