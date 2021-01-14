import logging


def main():
    import danalysis as da

    si = da.standard_systems.SI         
    with da.new_problem() as p:         
        p.drag = si.F
        p.mu = si.DynamicViscosity     
        p.b = si.L
        p.V = si.Velocity
        p.rho = si.Density
        
        result = p.solve_for(si.Unity) 
        print(result)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()