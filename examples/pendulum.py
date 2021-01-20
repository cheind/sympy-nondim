
def main():
    import nondim

    import sympy
    from sympy.physics import units    

    t, m, l, g, theta = sympy.symbols('t m l g theta')
    dimmap = {
        t:units.time, 
        m:units.mass, 
        l:units.length, 
        g:units.acceleration, 
        theta:units.Dimension(1)
    }
    eq = sympy.Eq(t, sympy.Function('f')(m,l,g,theta))
    # t = f(m,l,g,theta)
    print(sympy.latex(eq))
    
    r = nondim.nondim(eq, dimmap)
    print(sympy.latex(r))
    f = sympy.Eq(t, sympy.solve(r, t)[0])
    print(sympy.latex(f))

if __name__ == '__main__':
    main()
