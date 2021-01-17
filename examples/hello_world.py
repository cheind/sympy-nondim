import logging


def main():
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
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
