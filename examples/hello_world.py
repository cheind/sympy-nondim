import logging


def main():
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

