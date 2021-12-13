import numpy as np

def get_fn_at_x(xgrid, fn, at_x):
    """
    Returns function at x

    Args:
        xgrid np.array(n_grid): Grid where function was evaluated
        fn np.array((n_samples,n_grid)): Multiple function evaluations at xgrid 
        at_x float: Value closest to the desired function evaluation   

    Returns:
        fn_at_x np.array(n_samples): Function evaluations that where closest to x 
    """
    x_id = np.abs(xgrid - at_x).argmin().astype(int)
    fn_at_x = fn[:,x_id]
    return fn_at_x
