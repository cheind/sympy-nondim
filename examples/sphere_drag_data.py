import numpy as np
import pandas as pd
import itertools as it

def cd_sphere(Re):
    """This function computes the drag coefficient of a sphere as a function of the Reynolds number Re. Curve fitted after fig . A -56 in Evett and Liu: "Fluid Mechanics and Hydraulics. Taken from 
    http://lrhgit.github.io/tkt4140/allfiles/digital_compendium/._main006.html
    """
    
    if Re <= 0.0:
        CD = 0.0
    elif Re > 8.0e6:
        CD = 0.2
    elif Re > 0.0 and Re <= 0.5:
        CD = 24.0/Re
    elif Re > 0.5 and Re <= 100.0:
        p = np.array([4.22, -14.05, 34.87, 0.658])
        CD = np.polyval(p, 1.0/Re) 
    elif Re > 100.0 and Re <= 1.0e4:
        p = np.array([-30.41, 43.72, -17.08, 2.41])
        CD = np.polyval(p, 1.0/np.log10(Re))
    elif Re > 1.0e4 and Re <= 3.35e5:
        p = np.array([-0.1584, 2.031, -8.472, 11.932])
        CD = np.polyval(p, np.log10(Re))
    elif Re > 3.35e5 and Re <= 5.0e5:
        x1 = np.log10(Re/4.5e5)
        CD = 91.08*x1**4 + 0.0764
    else:
        p = np.array([-0.06338, 1.1905, -7.332, 14.93])
        CD = np.polyval(p, np.log10(Re))
    return CD

def drag_force(b, v, rho, mu):
    re = rho * v * b / mu
    cd = cd_sphere(re)
    drag = 0.5 * rho * v**2 * b**2 * re
    return drag

def main():
    print(cd_sphere(10e4))
    # data = []

    # gen = it.product(
    #     [0.001, 0.01, 0.1],             # base diameter m        
    #     [(998.,1e-3), (13600.,1.5e-3), (1.23,1.5e-5)],           # density and dyn.viscosity of water,mercury,air at 20°
    #     np.logspace(-1,2,20)*2,         # velocities [0.2..200]
    # )

    # for (b,(rho,mu),v) in gen:
    #     df = drag_force(b,v,rho,mu)
    #     data.append((b,rho,mu,v,df))

    # data = np.asarray(data)
    # np.savetxt('drag.txt', data, fmt='%1.7f', header='diameter density viscosity velocity drag')
    
if __name__ == '__main__':    
    main()