[![Build Status](https://www.travis-ci.com/cheind/sympy-nondim.svg?branch=develop)](https://www.travis-ci.com/cheind/sympy-nondim)
# sympy-nondim

This Python package addresses physical dimensional analysis. In
particular, `sympy-nondim` calculates from an unknown relation of
(dimensional) variables, a new relation of (usually fewer) dimensionless
variables.

See [nondim-sympy.pdf](docs/nondim-sympy.pdf) for a detailed introduction. 

``` python
import sympy
from sympy.physics import units

import nondim

# Potentially relevent variables
t, m, l, g, theta = sympy.symbols('t m l g theta')
# and associated dimensions
dimmap = {
    t:units.time, 
    m:units.mass, 
    l:units.length, 
    g:units.acceleration, 
    theta:units.Dimension(1)
}

# Setup an general equation, informing dimensional analysis
# of dependent and independent variables.
eq = sympy.Eq(t, sympy.Function('f')(m,l,g,theta))

# Perform dimensional analysis which returns a new (reduced) 
# expr. of dimensionless variables
r = nondim.nondim(eq, dimmap)

print(sympy.latex(r))
# \frac{\sqrt{g} t}{\sqrt{l}} = F{\left(\theta \right)}
```

The method implemented in this library is based on the Buckingham-Pi theorem and the Rayleigh algorithm as explained in (Szirtes 2007). The method implemented here frames the problem in linear algebra terms, see [buckpi.py](nondim/buckpi.py) for details.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-lemons2017student" class="csl-entry">

Lemons, Don S. 2017. *A Student’s Guide to Dimensional Analysis*.
Cambridge University Press.

</div>

<div id="ref-santiago2019first" class="csl-entry">

Santiago, Juan G. 2019. *A First Course in Dimensional Analysis:
Simplifying Complex Phenomena Using Physical Insight*. MIT Press.

</div>

<div id="ref-schetz1999fundamentals" class="csl-entry">

Schetz, Joseph A, and Allen E Fuhs. 1999. *Fundamentals of Fluid
Mechanics*. John Wiley & Sons.

</div>

<div id="ref-sonin2001dimensional" class="csl-entry">

Sonin, Ain A. 2001. “Dimensional Analysis.” Technical report,
Massachusetts Institute of Technology.
<http://web.mit.edu/2.25/www/pdf/DA_unified.pdf>.

</div>

<div id="ref-szirtes2007applied" class="csl-entry">

Szirtes, Thomas. 2007. *Applied Dimensional Analysis and Modeling*.
Butterworth-Heinemann.

</div>

</div>
