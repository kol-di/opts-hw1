from dataclasses import dataclass, asdict
import sympy as sp
import numpy as np

from plotter.plot import draw_plot
from utils.linalg import FO_der, SO_der_inv

X1, X2 = sp.symbols('x1 x2')


@dataclass
class TaskParams:
    cords: sp.Matrix
    optim_func: sp.Expr
    stop_val: float



def _new_cord(func: sp.Expr, cords: sp.Matrix) -> np.array:
    return np.array([
        [cords[0]], 
        [cords[1]],
        [func.evalf(subs={X1: cords[0], X2: cords[1]})]
    ])


def init_params() -> TaskParams:
    return TaskParams(
        sp.Matrix([1, 1]),
        (X1)**2 + sp.exp(X2**2), 
        # 2*(X1**2) + (X1-2)*X2 + 3*(X2-1)**2,
        1e-9
    )


def iteration(
        fo_der: sp.Matrix, 
        so_der_inv: sp.Matrix,
        cords: sp.Matrix, 
) -> sp.Matrix:
    
    eval_expr = sp.Matrix([*cords]) - (so_der_inv @ fo_der)
    return eval_expr.evalf(subs={X1: cords[0], X2: cords[1]})



def is_optimal(fo_der: sp.Expr, cords: sp.Matrix, stop_val: float) -> bool:
    l2_norm = sp.sqrt(sum(fo_der.T @ fo_der))
    return l2_norm.evalf(subs={X1: cords[0], X2: cords[1]}) < stop_val


def newton_optimise(visualise):
    cords, optim_func, stop_val = asdict(init_params()).values()
    num_iter = 0
    if visualise:
        steps_plt = _new_cord(optim_func, cords)

    fo_der = FO_der(optim_func, X1, X2)
    so_der_inv = SO_der_inv(optim_func, X1, X2)

    while not is_optimal(fo_der, cords, stop_val):
        cords = iteration(fo_der, so_der_inv, cords)
        num_iter += 1
        
        if visualise:
            steps_plt = np.concatenate((steps_plt, _new_cord(optim_func, cords)), axis=1)


    print(f'Completed in {num_iter} iterations')
    print ('Local minimum at', *cords)

    if visualise:
        draw_plot(sp.lambdify((X1, X2), optim_func), steps_plt)


if __name__ == '__main__':
    newton_optimise(visualise=True)