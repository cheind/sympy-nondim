import logging


def main():
    import danalysis as da
    import danalysis.standard_units as si

    r = da.solve(
        {'a':si.M, 'b':si.F, 'c':si.T}, 
        si.L
    )
    print(r)
    # Found 1 variable products generating dimension L:
    #   1: [a**-1*b*c**2] = L

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

