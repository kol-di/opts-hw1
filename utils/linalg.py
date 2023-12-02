import sympy as sp


def FO_der(func: sp.Expr, X1, X2) -> sp.Matrix:
    return sp.Matrix(
        [sp.diff(func, X1), 
         sp.diff(func, X2)]
    )

def SO_der_inv(func: sp.Expr, X1, X2) -> sp.Matrix:
    so_der = sp.Matrix(
        [[func.diff(x1).diff(x2) for x1 in (X1, X2)]
            for x2 in (X1, X2)]
    )
    return so_der.inv()