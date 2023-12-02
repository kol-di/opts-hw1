from dataclasses import dataclass, asdict
from typing import Generator
import sympy as sp
import numpy as np

from plotter.plot import draw_plot
from utils.linalg import FO_der, SO_der_inv

X1, X2 = sp.symbols('x1 x2')


@dataclass
class TaskParams:
    cords: sp.Matrix
    optim_func: sp.Expr
    restrict_func: sp.Expr
    stop_val: float # stop val for newton optimisation
    num_iter: int   # number of times to run quadratic penalty optimisation



def _new_cord(func: sp.Expr, cords: sp.Matrix) -> np.array:
    return np.array([
        [cords[0]], 
        [cords[1]],
        [func.evalf(subs={X1: cords[0], X2: cords[1]})]
    ])

def _C_SEQ_GEN(start=1, step=0.5) -> Generator[float, None, None]:
    c = start
    while True:
        yield c
        c += step


def init_params() -> TaskParams:
    return TaskParams(
        cords=sp.Matrix([-5, 5]),
        optim_func=2*(X1**2) + (X2-1)**2, 
        restrict_func=2*X1+X2,
        stop_val=1e-9, 
        num_iter=30
    )


def approximate_optim_func(optim_func: sp.Expr, restrict_func: sp.Expr, c: float) -> sp.Expr:
    penalty = (c/2) * restrict_func**2
    return optim_func + penalty


def newton_iteration(
        fo_der: sp.Matrix, 
        so_der_inv: sp.Matrix,
        cords: sp.Matrix, 
) -> sp.Matrix:
    """Newton unconditional optimisation step"""
    eval_expr = sp.Matrix([*cords]) - (so_der_inv @ fo_der)
    return eval_expr.evalf(subs={X1: cords[0], X2: cords[1]})

def is_optimal(
        fo_der: sp.Expr, 
        cords: sp.Matrix, 
        stop_val: float
) -> bool:
    """Condition used in Newton optimisation of approximated function"""
    l2_norm = sp.sqrt(sum(fo_der.T @ fo_der))
    return l2_norm.evalf(subs={X1: cords[0], X2: cords[1]}) < stop_val

def newton_optimise(
        optim_func: sp.Expr, 
        cords: sp.Matrix, 
        stop_val: float
) -> sp.Matrix:
    """
    Runs newton optimisation procedure for approximated function.
    Returns found minimizing coordinates for approximated function.
    """
    fo_der = FO_der(optim_func, X1, X2)
    so_der_inv = SO_der_inv(optim_func, X1, X2)

    while not is_optimal(fo_der, cords, stop_val):
        cords = newton_iteration(fo_der, so_der_inv, cords)
    return cords


def quad_penalty_optimise(visualise):
    cords, optim_func, restrict_func, stop_val, num_iter = asdict(init_params()).values()
    
    # define the c series
    c_seq_start, c_seq_step = 1, 1
    c_seq = _C_SEQ_GEN(c_seq_start, c_seq_step)

    iter_cnt = 0
    if visualise:
        steps_plt = _new_cord(optim_func, cords)

    # there might be a better stop criterion than number of iterations but idk
    for _ in range(num_iter):
        c = c_seq.__next__()
        optim_func_approx = approximate_optim_func(optim_func, restrict_func, c)

        # find minimizng coordinates for approximated function using
        # newton optimisation like in task 2
        cords = newton_optimise(optim_func_approx, cords, stop_val)

        iter_cnt += 1
        
        if visualise:
            steps_plt = np.concatenate((steps_plt, _new_cord(optim_func, cords)), axis=1)


    print(f'Executed algorithm with c series defined as: \nc_0 = {c_seq_start}\nc_i+1 = c_i + {c_seq_step}')
    print('Approximate function optimisation done using Newton method')
    print(f'Completed in {iter_cnt} iterations. Number of iterations was manually defined!')
    print ('Local minimum at', *cords)

    if visualise:
        draw_plot(sp.lambdify((X1, X2), optim_func), steps_plt, x_lims=(-5, 5), y_lims=(-5, 5))


if __name__ == '__main__':
    quad_penalty_optimise(visualise=True)