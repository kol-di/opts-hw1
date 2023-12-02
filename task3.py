from dataclasses import dataclass, asdict
import sympy as sp
import numpy as np

from utils.linalg import FO_der
from plotter.plot import draw_plot

X1, X2 = sp.symbols('x1 x2')


@dataclass
class TaskParams:
    cords: sp.Matrix
    optim_func: sp.Expr
    quad_form: sp.Matrix
    b: float
    stop_val: float
    stop_iter: int


def init_params() -> TaskParams:
    return TaskParams(
        cords=sp.Matrix([1, 1]),
        optim_func=(X1)**2 + 2*(X2**2),    
        quad_form=sp.Matrix([
            [1, 0],
            [0, 2]]),
        b=0,
        stop_val=1e-9, 
        stop_iter=10
    )

def _new_cord(func: sp.Expr, cords: sp.Matrix) -> np.array:
    return np.array([
        [cords[0]], 
        [cords[1]],
        [func.evalf(subs={X1: cords[0], X2: cords[1]})]
    ])

def iteration(
        fo_der: sp.Matrix, 
        d: sp.Matrix,
        A: sp.Matrix,
        b: float, # idk how to handle it
        cords: sp.Matrix, 
) -> sp.Matrix:
    # convinience wrapper
    eval = lambda expr: expr.evalf(subs={X1: cords[0], X2: cords[1]})
    
    if d is not None:
        B_new = eval((A@d).dot(fo_der) / (A@d).dot(d))
        d_new = eval((-1)*fo_der + B_new*d)
        print('B_new', B_new)
    else:
        d_new = eval((-1)*fo_der)
    print('d_new', d_new)
    
    alpha = (-1)*eval((2 * A@cords).dot(d_new) / (2*((A@d_new).dot(d_new))))
    print('alpha', alpha)
    cords_new = eval(cords + alpha * d_new)
    print('cords_new', cords_new)

    return cords_new, d_new
    


def conj_grad_optimise(visualise):
    cords, optim_func, quad_form, b, stop_val, stop_iter = asdict(init_params()).values()
    num_iter = 0
    fo_der = FO_der(optim_func, X1, X2)
    if visualise:
        steps_plt = _new_cord(optim_func, cords)

    d = None # to skip B on first iteration
    while (sum(fo_der.evalf(subs={X1: cords[0], X2: cords[1]})) > stop_val) and (num_iter < stop_iter):
        cords, d = iteration(fo_der, d, quad_form, b, cords)
        num_iter += 1
        
        if visualise:
            steps_plt = np.concatenate((steps_plt, _new_cord(optim_func, cords)), axis=1)


    print(f'Completed in {num_iter} iterations')
    print ('Local minimum at', *cords)

    if visualise:
        draw_plot(sp.lambdify((X1, X2), optim_func), steps_plt)


if __name__ == '__main__':
    conj_grad_optimise(visualise=True)