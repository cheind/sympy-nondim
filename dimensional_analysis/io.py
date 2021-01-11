from functools import reduce
import string


def fmt_solution(P, dvars, vnames=string.ascii_lowercase):
    if len(P) > 0:
        finaldim = reduce(
            lambda prev, z: prev*z[0]**z[1], 
            zip(dvars, P[0]), 
            dvars[0].system.unity)
        m = f'Found {len(P)} indep. variable products of dimension {finaldim}.\n'
        for i,p in enumerate(P):
            ve = [f'{v}**{e}' for v,e in zip(vnames, p) if e!=0]
            m = m + f'{i}: {"*".join(ve)}\n'
    else:
        m = 'Found 0 solutions.'
    return m