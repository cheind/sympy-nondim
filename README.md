# py-dimensional-analysis

This Python package addresses physical dimensional analysis. In
particular, `py-dimensional-analysis` calculates from a given system of
(dimensional) variables those products that yield a desired target
dimension.

The following example illustrates, how variables corresponding to mass,
force, time, and pressure have to relate to produce dimension length.

``` python
import danalysis as da
import danalysis.standard_units as si

r = da.solve(
    {'a':si.M, 'b':si.F, 'c':si.T, 'd':si.pressure}, 
    si.L*si.T
)
print(r)
# Found 2 variable products generating dimension L*T:
#    1: [a*c**-1*d**-1] = L*T
#    2: [b**0.5*c*d**-0.5] = L*T
```

This library is based on (Szirtes 2007), and also incorporates ideas and
examples from (Santiago 2019; Sonin 2001).

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-santiago2019first" class="csl-entry">

Santiago, Juan G. 2019. *A First Course in Dimensional Analysis:
Simplifying Complex Phenomena Using Physical Insight*. MIT Press.

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
