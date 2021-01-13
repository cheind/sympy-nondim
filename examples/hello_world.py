import logging


def main():
    import danalysis as da

    si = da.standard_systems.SI         # predefined standard units
    with da.new_problem() as p:         # records variables and dimensions
        p.a = si.M
        p.b = si.L*si.M*si.T**-2        # or simply si.F
        p.c = si.T
        p.d = si.Pressure
        
        result = p.solve_for(si.L*si.T) # solve with target dimension
        print(result)
            # Found 2 independent variable products, each of dimension L*T:
            #   1: [a*c**-1*d**-1] = L*T
            #   2: [b**0.5*c*d**-0.5] = L*T
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
