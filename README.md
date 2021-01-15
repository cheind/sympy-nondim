# py-dimensional-analysis

This Python package addresses physical dimensional analysis. In
particular, `py-dimensional-analysis` calculates from a given system of
(dimensional) variables those products that yield a desired target
dimension.

The following example illustrates how the variables mass, force, time
and pressure must relate to each other in order to produce the dimension
length\*time.

``` python
import danalysis as da

si = da.standard_systems.SI         # predefined standard units
s = da.Solver(
    {
        'a' : si.M,                 # [a] is mass
        'b' : si.L*si.M*si.T**-2,   # [b] is force (alt. si.F)
        'c' : si.T,                 # [c] is time
        'd' : si.Pressure           # [d] is pressure
    },
    si.L*si.T                       # target dimension
)
print(s.solve())
    # Found 2 variable products of variables
    # {
    #         a:Q(M),
    #         b:Q(L*M*T**-2),
    #         c:Q(T),
    #         d:Q(L**-1*M*T**-2)
    # }, each of dimension L*T:
    #         1: [a*c**-1*d**-1] = L*T
    #         2: [b**0.5*c*d**-0.5] = L*T
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
