import logging
import dimensional_analysis as da
from dimensional_analysis import si

def main():
    r = da.solve([si.M, si.F, si.T], si.L)
    print(r)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

