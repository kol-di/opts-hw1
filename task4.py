from dataclasses import dataclass, asdict
import sympy as sp
import numpy as np
from typing import Tuple
from itertools import product

from utils.linalg import FO_der
from plotter.plot import draw_plot

X1, X2 = sp.symbols('x1 x2')


@dataclass
class TaskParams:
    cords: sp.Matrix
    optim_func: sp.Expr
    x1_lims: Tuple[float, float]
    x2_lims: Tuple[float, float]
    stop_val: float

def init_params() -> TaskParams:
    return TaskParams(
        cords=sp.Matrix([0, 0]),
        optim_func=(X1**2) - (4*X1) + (X2**2) - (2*X2),    
        x1_lims=(0, 1),
        x2_lims=(0, 2),
        stop_val=0.1
    )

def _new_cord(func: sp.Expr, cords: sp.Matrix) -> np.array:
    return np.array([
        [cords[0]], 
        [cords[1]],
        [func.evalf(subs={X1: cords[0], X2: cords[1]})]
    ])


def is_optimal(cords_prev, cords_new, stop_val):
    cord_diff = cords_prev - cords_new
    l2_norm = sp.sqrt(sum(cord_diff.T @ cord_diff))
    return l2_norm <= stop_val


def _get_dir(
        fo_der: sp.Matrix,
        cords: sp.Matrix, 
        x1_lims: Tuple[float, float], 
        x2_lims: Tuple[float, float] 
) -> sp.Matrix:
    # evaluated dot product is reduced to linear funtion (in our case)
    prod = fo_der.evalf(subs={X1: cords[0], X2: cords[1]}).dot(sp.Matrix([X1, X2]) - cords)
    # for linear objectiove with linear constrints minimum would be in one of corners
    cords_minimizing = min(
        product(x1_lims, x2_lims), 
        key=lambda crd: prod.evalf(subs={X1: crd[0], X2: crd[1]}))
    cords_minimizing = sp.Matrix([*cords_minimizing])

    return cords_minimizing - cords


def _get_alpha(
        optim_func: sp.Expr, 
        cords: sp.Matrix,
        d: sp.Matrix
) -> float:
    alpha = sp.symbols('alpha')
    new_cords = cords + alpha * d
    min_func_wrt_alpha = optim_func.subs({X1: new_cords[0], X2: new_cords[1]})
    der_wrt_alpha = sp.diff(min_func_wrt_alpha, alpha)
    minimizing_alpha = [a for a in sp.solve(der_wrt_alpha) if a > 0][0]

    return minimizing_alpha


def iteration(
        optim_func: sp.Expr,
        fo_der: sp.Matrix, 
        cords: sp.Matrix, 
        x1_lims: Tuple[float, float], 
        x2_lims: Tuple[float, float]
) -> sp.Matrix:
    d = _get_dir(fo_der, cords, x1_lims, x2_lims)
    alpha = _get_alpha(optim_func, cords, d)
    new_cords = (cords + alpha * d).evalf()

    return new_cords
    

def cond_grad_optimise(visualise):
    cords, optim_func, x1_lims, x2_lims, stop_val = asdict(init_params()).values()
    num_iter = 0
    fo_der = FO_der(optim_func, X1, X2)
    if visualise:
        steps_plt = _new_cord(optim_func, cords)

    while True:
        new_cords = iteration(optim_func, fo_der, cords, x1_lims, x2_lims)
        num_iter += 1
        
        if visualise:
            steps_plt = np.concatenate((steps_plt, _new_cord(optim_func, cords)), axis=1)

        if is_optimal(new_cords, cords, stop_val):
            break
        cords = new_cords


    print(f'Completed in {num_iter} iterations')
    print ('Local minimum at', *cords)

    if visualise:
        draw_plot(sp.lambdify((X1, X2), optim_func), steps_plt, x1_lims, x2_lims)


if __name__ == '__main__':
    cond_grad_optimise(visualise=True)