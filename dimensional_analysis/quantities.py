import numpy as np
from . import utils as u

class DimensionalSystem:
    def __init__(self, name, *base_dimension_names):
        N = len(base_dimension_names)
        assert N > 0, 'Need more than zero base dimensions'
        self.name = name
        self.base_dimension_names = base_dimension_names        
        e = np.eye(N)
        self.base_dimensions = [self.create(u.basis_vec(i,N)) for i in range(N)]
        
    def __len__(self):
        return len(self.base_dimension_names)

    @property
    def unity(self):
        return self.create(np.zeros(len(self)))

    def create(self, exponents):
        assert len(exponents) == len(self), 'Number of exponents does not match number of dimensions'
        return Quantity(self, exponents)

class Quantity:    
    def __init__(self, sys, exponents):        
        self.e = np.asarray(exponents)
        self.system = sys

    def __repr__(self):
        return f'<Quantity({self.system.name}) {str(self)}>'       

    def __str__(self):
        if self.dimensionless:
            return '1'
        else:
            d = [self._fmt_dim(name,e) for name,e in zip(self.system.base_dimension_names, self.e)]
            d = [dd for dd in d if dd != '']
        return '*'.join(d)

    def __pow__(self, exponent):
        return self._create(self.e*exponent)

    def __mul__(self, other):
        assert isinstance(other, Quantity)
        return self._create(self.e + other.e)

    def __truediv__(self, other):    
        assert isinstance(other, Quantity)
        return self * (other**-1)

    def __array__(self):
        return self.e

    def __iter__(self):
        return iter(self.e)

    def __len__(self):
        return len(self.e)

    def _create(self, exponents):
        return self.system.create(exponents)

    def __getitem__(self, idx):
        return self.e[idx]

    @property
    def exponents(self):
        return self.e

    @property
    def dimensionless(self):
        return u.dimensionless(self)

    @property
    def shape(self):
        return self.e.shape

    def _fmt_dim(self, name,e):
        if np.allclose(e, 1): # misuse of allclose for scalars
            return name
        elif not np.allclose(e, 0):
            return f'{name}**{format(e,".2f").rstrip("0").rstrip(".")}'
        else:
            return ''

def create_dimensional_system(name, *base_dimension_names):  
    sys = DimensionalSystem(name, *base_dimension_names)
    return sys 

