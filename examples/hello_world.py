import logging
import danalysis as da
from danalysis import standard_units as si

def main():
    print([si.M, si.F, si.T], si.L)
    r = da.solve([si.M, si.F, si.T], si.L)
    print(r)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

