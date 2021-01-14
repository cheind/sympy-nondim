import logging


def main():
    import danalysis as da

    si = da.standard_systems.SI         # predefined standard units
    with da.new_problem() as p:         # records variables and dimensions
        p.drag = si.F
        p.mu = si.Viscosity             # order is important it seems.
        p.b = si.L
        p.V = si.Velocity
        p.rho = si.Density
        
        result = p.solve_for(si.Unity) # solve with target dimension
        print(result)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()