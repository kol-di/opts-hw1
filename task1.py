from dataclasses import dataclass, asdict
import sympy as sp
import numpy as np
import time

from plotter.plot import draw_plot
from utils.linalg import FO_der#, SO_der_inv

X1, X2 = sp.symbols('x1 x2')


@dataclass
class TaskParams:
    cords: sp.Matrix
    optim_func: sp.Expr
    alpha: float
    eps: float
    theta: float
    stop_val: float



def _new_cord(func: sp.Expr, cords: sp.Matrix) -> np.array:
    return np.array([
        [cords[0]], 
        [cords[1]],
        [func.evalf(subs={X1: cords[0], X2: cords[1]})]
    ])


def init_params() -> TaskParams:
    return TaskParams(
        cords=sp.Matrix([-1, 1]),
        optim_func=2*(X1)**2 + X1*X2 + 3*(X2)**2, 
        # 2*(X1**2) + (X1-2)*X2 + 3*(X2-1)**2,
        alpha=1, 
        eps=0.5, 
        theta=0.5,
        stop_val=1e-9
    )


def _choose_alpha_armiho(
        cords: sp.Matrix, 
        optim_func: sp.Expr, 
        alpha: float,
        eps: float, 
        theta: float,
        fo_der: sp.Matrix
) -> float:
    """Choose alpha to step in the direction of antigradient"""
    while True:
        new_cords = sp.Matrix([*cords]) + alpha * (-1)*fo_der
        linear_decrement = eps * alpha * fo_der.dot((-1)*fo_der)
        eval_ineq = optim_func.subs({X1: new_cords[0], X2: new_cords[1]}) - \
                    optim_func.subs({X1: cords[0], X2: cords[1]}) - \
                    linear_decrement

        if eval_ineq.evalf(subs={X1: cords[0], X2: cords[1]}) <= 0:
            break
        alpha = theta * alpha
    
    return alpha


def iteration(
        fo_der: sp.Matrix, 
        cords: sp.Matrix, 
        optim_func: sp.Expr,
        alpha: float, 
        eps: float, 
        theta: float
) -> sp.Matrix:
    alpha = _choose_alpha_armiho(cords, optim_func, alpha, eps, theta, fo_der)
    eval_expr = sp.Matrix([*cords]) - alpha * fo_der
    return eval_expr.evalf(subs={X1: cords[0], X2: cords[1]})


def is_optimal(fo_der: sp.Expr, cords: sp.Matrix, stop_val: float) -> bool:
    l2_norm = sp.sqrt(sum(fo_der.T @ fo_der))
    return l2_norm.evalf(subs={X1: cords[0], X2: cords[1]}) < stop_val


def grad_optimise(visualise):
    cords, optim_func, alpha, eps, theta, stop_val = asdict(init_params()).values()
    num_iter = 0
    if visualise:
        steps_plt = _new_cord(optim_func, cords)

    fo_der = FO_der(optim_func, X1, X2)

    while not is_optimal(fo_der, cords, stop_val):
        cords = iteration(fo_der, cords, optim_func, alpha, eps, theta)
        num_iter += 1
        
        if visualise:
            steps_plt = np.concatenate((steps_plt, _new_cord(optim_func, cords)), axis=1)


    print(f'Completed in {num_iter} iterations')
    print ('Local minimum at', *cords)

    if visualise:
        draw_plot(sp.lambdify((X1, X2), optim_func), steps_plt)


if __name__ == '__main__':
    grad_optimise(visualise=True)