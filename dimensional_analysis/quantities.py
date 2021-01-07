import numpy as np

class Quantity:
    def __init__(self, exponents):        
        self.e = np.asarray(exponents)

    def __repr__(self):
        return f'<{self.qcls.__name__} {str(self)}>'       

    def __str__(self):
        if self.dimensionless:
            return '1'
        else:
            d = [self._fmt_dim(name,e) for name,e in zip(self.base_dimensions, self.e)]
        return ''.join(d)

    def __pow__(self, exponent):
        cls = self.__class__
        return cls(self.e*exponent)

    def __mul__(self, other):
        cls = self.__class__     
        assert isinstance(other, cls)        
        return cls(self.e + other.e)

    def __truediv__(self, other):    
        cls = self.__class__
        assert isinstance(other, cls)    
        return self * (other**-1)

    def __array__(self):
        return self.e

    def __iter__(self):
        return iter(self.e)

    def __len__(self):
        return len(self.e)

    def __getitem__(self, i):
        if i > len(self):
            raise IndexError('Index out of range')
        return self.e[n]

    @property
    def base_dimensions(self):
        raise NotImplementedError()

    @property
    def dimensionless(self):
        return np.allclose(self.e, 0.)

    @property
    def shape(self):
        return self.e.shape

    def _fmt_dim(self, name,e):
        if np.allclose(e, 1): # misuse of allclose for scalars
            return name
        elif not np.allclose(e, 0):
            return f'{name}({format(e,".2f").rstrip("0").rstrip(".")})'
        else:
            return ''

def create_dimensional_system(name, *base_dimension_names):   
    klass = type(f'{name}Quantity', (Quantity,), {'base_dimensions':property(lambda self: base_dimension_names)})
    N = len(base_dimension_names)
    e = np.eye(N)
    return [klass(e[i].copy()) for i in range(N)]


