"""
Utils for logging
"""
import numpy as np

def convert_log_dict_to_np(logs):
    """
    Take in logs and return params
    """
    # Init params
    n_samples_after_warmup = len(logs)
    n_grid = logs[0]['u'].shape[-1] 
    u = np.zeros((n_samples_after_warmup, n_grid))
    Y = np.zeros((n_samples_after_warmup, n_grid))
    k = np.zeros((n_samples_after_warmup, n_grid))
    kl_trunc_errs = np.empty((n_samples_after_warmup,1))
    n_stoch_disc = logs[0]['coefs'].shape[-1] # e.g., n_alpha_indices for PCE, or kl_dim for KL-E
    coefs = np.empty((n_samples_after_warmup, n_grid, n_stoch_disc))
    stoch_dim = logs[0]['rand_insts'].shape[-1]
    rand_insts = np.empty((n_samples_after_warmup, stoch_dim))
    # Copy logs into params
    for n, log in enumerate(logs):
        k[n,:] = log['rand_param']
        Y[n,:] = log['Y']
        u[n,:] = log['u']
        kl_trunc_errs[n,0] = log['kl_trunc_err']
        coefs[n,:,:] = log['coefs']
        rand_insts[n,:] = log['rand_insts']
    return k, Y, u, kl_trunc_errs, coefs, rand_insts

def convert_log_dict_to_np_localAdvDiff(logs):
    """
    Convert log dictionary into np.arrays of solution, parameters, etc.

    # TODO: double check dimensionality of rand_param
    Args:
        logs n_samples*[
            dict(
                'rand_insts': np.array(LocalAdvDiffEq.gp_stoch_dim): Instances to generate random parameter
                'rand_param': np.array(LocalAdvDiffEq.n_zgrid): Random parameter, here, k
                'u': np.array((LocalAdvDiffEq.n_tgrid, LocalAdvDiffEq.n_zgrid)): Solution
            )
        ]: n samples of the model

    Returns:
        u_args dict(
            'rand_param': np.array((n_samples_after_warmup, n_tgrid, n_xgrid)): Random parameter, here, k
            )
        u np.array((n_samples_after_warmup, n_tgrid, n_xgrid))
        rand_insts np.array((n_samples_after_warmup, LocalAdvDiffEq.gp_stoch_dim)): 
    """
    # Retrieve dimensions
    n_samples_after_warmup = len(logs)
    n_tgrid = logs[0]['u'].shape[0]
    n_xgrid = logs[0]['u'].shape[1]
    stoch_dim = logs[0]['rand_insts'].shape[-1]

    # Initialize output vectors
    u = np.zeros((n_samples_after_warmup, n_tgrid, n_xgrid))
    k = np.zeros((n_samples_after_warmup, n_tgrid, n_xgrid))
    rand_insts = np.empty((n_samples_after_warmup, stoch_dim))

    # Copy logs into np.arrays
    for n, log in enumerate(logs):
        k[n,:,:] = log['rand_param']
        u[n,:,:] = log['u']
        rand_insts[n,:] = log['rand_insts']
    u_args = {'rand_param': k}
    return u_args, u, rand_insts

def convert_log_dict_to_np_lorenz96(logs):
    """
    Convert log dictionary into np.arrays of solution, parameters, etc.

    Args:
        logs n_samples*[
            dict(
                'rand_insts': np.array(1): Instances to generate random parameter
                'rand_param': np.array(1): Random parameter, here, k
                'u': (np.array(n_tgrid, K): Low-res solution
                    np.array(n_tgrid, J, K): Middle-res solution
                    np.array(n_tgrid, I, J, K): High-res solution
                ): Solution
            )
        ]: n samples of the model

    Returns:
        u_args {
            'rand_param': np.array((n_samples_after_warmup, 1)) : Random parameter, currently undefined
            'rand_ic': (np.array((n_samples_after_warmup, K))
                np.array((n_samples_after_warmup, J, K))
                np.array((n_samples_after_warmup, I, J, K))
                ): Random initial condition
        } 
        u (np.array((n_samples_after_warmup, n_tgrid, K))
            np.array((n_samples_after_warmup, n_tgrid, J, K))
            np.array((n_samples_after_warmup, n_tgrid, I, J, K))
        )
        rand_insts np.array((n_samples_after_warmup, 1))
    """
    # Retrieve dimensions
    n_samples_after_warmup = len(logs)
    stoch_dim = logs[0]['rand_insts'].shape[-1]

    # Initialize output vectors
    u_x = np.zeros((n_samples_after_warmup,) + logs[0]['u'][0].shape)
    u_y = np.zeros((n_samples_after_warmup,) + logs[0]['u'][1].shape)
    u_z = np.zeros((n_samples_after_warmup,) + logs[0]['u'][2].shape)
    
    u_x0 = np.zeros((n_samples_after_warmup,) + logs[0]['rand_ic'][0].shape)
    u_y0 = np.zeros((n_samples_after_warmup,) + logs[0]['rand_ic'][1].shape)
    u_z0 = np.zeros((n_samples_after_warmup,) + logs[0]['rand_ic'][2].shape)
    
    rand_param = np.zeros((n_samples_after_warmup, logs[0]['rand_param'].shape[0]))
    rand_insts = np.zeros((n_samples_after_warmup, stoch_dim))

    # Copy logs into np.arrays
    for n, log in enumerate(logs):
        u_x[n,:] = log['u'][0]
        u_y[n,:,:] = log['u'][1]
        u_z[n,:,:,:] = log['u'][2]
        
        u_x0[n,:] = log['rand_ic'][0]
        u_y0[n,:,:] = log['rand_ic'][1]
        u_z0[n,:,:,:] = log['rand_ic'][2]

        rand_param[n,:] = log['rand_param']
        rand_insts[n,:] = log['rand_insts']

    u = (u_x, u_y, u_z)
    u_0 = (u_x0, u_y0, u_z0)

    u_args = {
        'rand_param': rand_param, 
        'rand_ic': u_0}
    return u_args, u, rand_insts

def convert_log_dict_to_np_helmholtz(logs):
    """
    Convert log dictionary into np.arrays of solution, parameters, etc.

    Args:
        logs n_samples*[
            dict(
                'rand_ic': np.array(1): Instances to generate random parameter
                'rand_insts': np.array(1): Instances to generate random parameter
                'rand_param': np.array(n_x1grid, n_x2grid): Complex random parameter, here, k^2
                'u': np.array(n_x1grid, n_x2grid): Complex solution
            )
        ]: n samples of the model

    Returns:
        u_args {
            'rand_param': np.array((n_samples_after_warmup, n_x1grid, n_x2grid, 1)) : Random parameter
        } 
        u np.array((n_samples_after_warmup, n_x1grid, n_x2grid, 1)
        rand_insts np.array((n_samples_after_warmup, 1))
    """
    # Retrieve dimensions
    n_samples_after_warmup = len(logs)
    stoch_dim = logs[0]['rand_insts'].shape[-1]

    # Initialize output vectors
    u = np.zeros((n_samples_after_warmup,) + logs[0]['u'].shape + (1,), dtype=complex)
    u0 = np.zeros((n_samples_after_warmup, logs[0]['rand_ic'].shape[0]), dtype=complex)
    rand_param = np.zeros((n_samples_after_warmup,) + logs[0]['rand_param'].shape + (1,), dtype=complex)
    rand_insts = np.zeros((n_samples_after_warmup, stoch_dim))

    # Copy logs into np.arrays
    for n, log in enumerate(logs):
        u[n,:,:,0] = log['u']
        u0[n,:] = log['rand_ic']
        rand_param[n,:,:,0] = log['rand_param']
        rand_insts[n,:] = log['rand_insts']

    u_args = rand_param

    return u_args, u, rand_insts
