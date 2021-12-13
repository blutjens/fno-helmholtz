"""
A lot of nasty reshaping
Author: Björn Lütjens (lutjens@mit.edu)
"""
import numpy as np


def get_input_dims(grid, y_args, grid_in_dims):
    """
    Returns input dimensions

    Args:
        grid np.array((n_samples, n_grid)) or 
            (np.array(n_xgrid), np.array(n_tgrid))
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_grid)): grid for 1D, 2D, or ND input, respectively
        y_args np.array((n_samples, n_grid, dim_y_args)) or 
            (np.array(n_samples, n_tgrid, n_xgrid, dim_y_args)) or
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_y_args)): Additional inputs, e.g., forcing terms 
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all.
    Returns:
        n_grid tuple(int): Tuple of all grid lengths (n_tgrid, n_x1grid, n_x2grid, ...)
        n_tgrid int:
        n_xgrid int:
        dim_grid int:
        dim_in int:
    """
    # Calculate grid dimension for different shapes of input grids
    if len(grid) > 2: # n_samples, n_tgrid, n_x1 ..., dim_grid
        n_tgrid = grid.shape[1]
        n_xgrid = grid.shape[2]
        n_grid = grid.shape[1:-1]
    elif type(grid) is not tuple: # n_samples, n_grid
        n_tgrid = grid.shape[-1]
        n_xgrid = 0
        n_grid = (n_tgrid,)
    else:
        n_xgrid = grid[0].shape[0]
        n_tgrid = grid[1].shape[0]
        n_grid = (n_tgrid, n_xgrid)
    dim_grid = len(n_grid)

    # Calculate number of y_args inputs
    if y_args is not None:
        dim_y_args = y_args.shape[-1]
    else:
        dim_y_args = 0

    # Calculate number of grid input dimensions
    if len(grid_in_dims) == 1:
        if grid_in_dims[0] == -1:
            n_grid_in_dims = dim_grid 
        n_grid_in_dims = len(grid_in_dims)
    else:
        n_grid_in_dims = len(grid_in_dims)

    # Calculate number of total inputs
    dim_in = n_grid_in_dims + dim_y_args

    return n_grid, n_tgrid, n_xgrid, dim_grid, dim_in

def flatten_1D_to_0D(grid, y, y_args, grid_in_dims=(-1,)):
    """
    Merges n_samples and n_grid axes into one flattened array. Assumes that samples are iid. across location, x
    
    Args:
        grid np.array((n_samples, n_grid)) or 
            np.array((n_samples, n_grid, dim_grid)): grid for 1D input
        y_args np.array((n_samples, n_grid, dim_y_args))
        y np.array((n_samples, n_grid, dim_y))
        grid_in_dims tuple(int): Indices of the grid dimensions who's will be used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all grids
    Returns:
        x np.array((n_samples*n_grid, dim_in))
        y np.array((n_samples*n_grid, dim_out))
    """
    if y_args is None:
        x = grid
    else:
        if len(grid.shape) > 2: # Grid shape: n_samples, n_grid, dim_grid
            if len(grid_in_dims) > grid.shape[-1]:
                raise NotImplementedError('Too many indices used as input. Reduce #indices in grid_in_dims.')
            elif len(grid_in_dims) == 0: # No grid as input
                x = y_args
            elif grid_in_dims[0] == -1: # All grid as input
                x = np.concatenate((grid, y_args), axis=2)
            else: # Selected cells as input
                x = np.concatenate((grid[:,:,grid_in_dims], y_args), axis=2)
        else:
            if grid_in_dims[0] != -1:
                raise NotImplementedError('Change grid shape to ues selected elements of grid as input.')
            x = np.concatenate((grid[:,:,np.newaxis], y_args), axis=2)
    x = x.reshape(-1,x.shape[-1])

    if y is not None:
        dim_out = y.shape[-1]
        y = y.reshape(-1, dim_out)

    # TODO: delete this code block once flatten_1D_to_0D was tested with mu_k or k_eigvecs
    #elif target_name == 'mu_k' or target_name == 'k_eigvecs':
    """
    Merge n_samples and n_grid axes into one flattened array. Assumes that samples are iid. across location, x
    Args:
        grid np.array((n_samples, n_grid))
    Returns:
        x np.array((n_samples*n_grid, 1))
        y np.array((n_samples*n_grid, dim_out))
    """
    #x = grid.reshape(-1, 1) # dim: (n_samples*n_grid, 1)
    #dim_out = y.shape[-1] 
    #y = y.reshape(-1, dim_out)
    #if y_args is not None: return NotImplementedError("y_args not implemented in neural_net.flatten_to_1D_nn_input")

    return x, y

def flatten_2D_to_0D(grid, y, y_args, grid_in_dims=(-1,)):
    """
    Flattens out 2D array into input, x, and target, y   
    Args: 
        grid (np.array(n_xgrid), np.array(n_tgrid)) or
            np.array((n_samples, n_tgrid, n_xgrid, dim_grid)): grid for 2D input
        y_args np.array((n_samples, n_tgrid, n_xgrid, dim_y_args)) or
        y np.array((n_samples, n_tgrid, n_xgrid, dim_y))
        grid_in_dims tuple(int): Indices of the grid dimensions who's will be used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all grids
    Returns:
        x np.array(n_samples*n_tgrid*n_xgrid, dim_in)
        y np.array(n_samples*n_tgrid*n_xgrid, dim_y)
    """
    if type(grid) is not tuple: # n_samples, n_tgrid, n_xrid, dim_grid
        dim_grid = grid.shape[-1]
        x = grid
    else:
        # Create meshgrid over spatio-temporal domain t, x # TODO: remove when all grid inputs are n_samples, n_tgrid, n_xrid, dim_grid
        xgrid, tgrid = grid
        n_xgrid = xgrid.shape[0]
        n_tgrid = tgrid.shape[0]
        n_samples = y.shape[0]
        xx, tt = np.meshgrid(xgrid, tgrid)
        x = np.concatenate((xx[:,:, np.newaxis], tt[:,:, np.newaxis]), axis=2) # dim: n_tgrid, n_xgrid, dim_grid
        x = np.repeat(x[np.newaxis,:,:,:], repeats=n_samples, axis=0) # dim: n_samples, n_tgrid, n_xgrid, dim_grid
    
    # Concatenate grid and y_args, depending on availability
    x = concat_grid_y_args(grid, y_args, grid_in_dims)

    # Flatten
    x = x.reshape(-1, x.shape[-1]) # dim:  n_samples*n_tgrid*n_xgrid, dim_in
    
    # Flatten output array array to match input grid
    if y is not None:
        dim_y = y.shape[-1]
        y = y.reshape(-1, dim_y) # dim:  n_samples*n_tgrid*n_xgrid, dim_in
        # TODO: Align the random instances with their samples

    return x, y

def repeat_0D_to_flattened_2D_to_0D(rand_insts, n_grid):
    """
    Flattens out 2D array into input, x, and target, y   
    Args: 
        rand_insts np.array((n_samples, dim_rand)) or
            np.array((n_samples, n_tgrid, n_x1grid, ..., dim_rand))
        n_grid tuple(int): tuple(n_tgrid, n_x1grid, x_x2grid, ...)
    Returns:
        rand_insts np.array(n_samples*prod(n_grid), dim_rand)
    """
    if rand_insts is not None:
        if len(rand_insts.shape) <= 2:
            rand_insts = np.repeat(rand_insts, np.prod(n_grid), axis=0)
        rand_insts = rand_insts.reshape(-1, rand_insts.shape[-1])

    return rand_insts

def flatten_to_0D_data(grid, y, y_args=None, rand_insts=None, 
    target_name="u", batch_size=None, alpha_indices=None, grid_in_dims=(-1,)):
    """
    Flattens PDE solution, parameters, and grid to create a 0D neural network input and output array

    Args:
        grid np.array((n_samples, n_grid)) or 
            (np.array(n_xgrid), np.array(n_tgrid))
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_grid)): grid for 1D, 2D, or ND input, respectively
        y_args np.array((n_samples, n_grid, dim_y_args)) or 
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_y_args)): Additional inputs, e.g., forcing terms 
        y np.array((n_samples, n_grid, dim_y)) or
            np.array(n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_y: Target solution, e.g., temperature
        rand_insts np.array((n_samples, dim_rand)): Random instances, used to generated training dataset
        target_name string: Indication which target
        batch_size int: Batch size
        alpha_indices np.array((n_alpha_indices, dim_rand)): Set of alpha indices. If not None, used as prediction target and in PceLoss. See rom.pce.multi_indices()
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid.

    Returns:
        x np.array((n, dim_in)): 1D input array, see if/else statements for more information
        y np.array((n, dim_y)): 1D output array, see if/else statements for more information
        dim_out int: Number of output dimensions of the NN. Used to distinguish direct and indirect prediction of the target. 
        rand_insts np.array((n, dim_rand)): Random instances, see if/else statements for more information
        batch_size int: Batch size
    """
    # Get input dimensions
    n_grid, n_tgrid, n_xgrid, dim_grid, dim_in = get_input_dims(grid, y_args, grid_in_dims)
    if y is None: dim_out = 0

    # TODO. the current reshaping assumes samples are independent in space!!
    if target_name == 'u' or target_name == 'u_true':
        if dim_grid == 1:
            x, y = flatten_1D_to_0D(grid, y, y_args, grid_in_dims=grid_in_dims)
        if dim_grid == 2:
            x, y = flatten_2D_to_0D(grid, y, y_args, grid_in_dims=grid_in_dims)
            rand_insts = repeat_0D_to_flattened_2D_to_0D(rand_insts, n_grid)
    elif grid.shape > 2:
        raise NotImplementedError('grid shape np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_grid)) only implemented for target_name "u"')
    elif target_name == 'pce_coefs':
        # Repeat target into batches for deterministic target
        # TODO: check if this is actually necessary
        repeats = batch_size
        x = np.repeat(grid[:, np.newaxis], repeats=repeats, axis=0)
        if y is not None:
            dim_out = y.shape[-1] 
            y = np.repeat(y[:,:], repeats=repeats, axis=0)
        if y_args is not None: return NotImplementedError("y_args not implemented in neural_net.flatten_to_0D_data")
    elif target_name == 'k' or target_name == 'k_true':
        if y_args is not None or dim_grid > 1: return NotImplementedError("y_args not implemented in neural_net.flatten_to_0D_data")
        # Merge n_samples and n_grid axes into one. Assumes that samples are iid. across location, x
        x = grid.reshape(-1, 1)
        if y is not None:
            y = y.reshape(-1, 1) 
        # Align the random instances with their samples
        if rand_insts is not None:
            # TODO! This is probably wrong; it should rather be:
            # rand_insts = np.repeat(rand_insts[:,np.newaxis,:], repeats=n_xgrid,axis=1)
            # rand_insts = rand_insts.reshape(n_samples*n_xgrid, dim_rand)
            rand_insts = np.repeat(rand_insts[:,:], repeats=grid.shape[1],axis=0)
    elif target_name == 'mu_k' or target_name == 'k_eigvecs':
        x, y = flatten_1D_to_0D(grid, y, y_args, grid_in_dims=grid_in_dims)

    # Set output dimension to predict y directly or indirectly
    if alpha_indices is None:
        if y is not None:
            dim_out = y.shape[-1]
    else: 
        # Predict y target indirectly through PCE coefficients 
        dim_out = alpha_indices.shape[0]

    # Set batch_size to equal the full grid input
    if batch_size is None:
        batch_size = int(np.prod(n_grid))

    return x, y, dim_out, rand_insts, batch_size

def concat_grid_y_args(grid, y_args, grid_in_dims):
    """
    # Concatenate grid and y_args, depending on availability
    Args:
        grid np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_grid)): grid for 2D input
        y_args np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_y_args)): Additional inputs, e.g., forcing terms   
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid.
    Returns:
       x np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_in)): 
    """
    if len(grid_in_dims) > grid.shape[-1]:
        raise AssertionError('Too many indices used as input. Reduce #indices in grid_in_dims.')
    if y_args is not None:
        if len(grid_in_dims) == 0: # No grid as input
            x = y_args
        elif grid_in_dims[0] == -1: # All grids as input, default
            x = np.concatenate((grid, y_args), axis=-1)
        else: # Selected cells as input
            x = np.concatenate((grid[...,grid_in_dims], y_args), axis=-1) # dim: n_samples, n_tgrid, n_xgrid, dim_in
    else:
        if grid_in_dims[0] == -1:
            x = grid
        else:
            x = grid[...,grid_in_dims]
    return x

def shape_2D_to_2D_nn_input(grid, y, y_args=None, rand_insts=None,
    target_name=None,
    batch_size=None, alpha_indices=None, grid_in_dims=(-1,)):
    """
    Shapes PDE solution, parameters, and grid to create a 2D convolutional neural network input and output array

    Args:
        grid np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_grid)): grid for 2D input
        y_args np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_y_args)): Additional inputs, e.g., forcing terms 
        y np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_y: Target solution, e.g., temperature
        rand_insts np.array((n_samples, dim_rand)): Random instances, used to generated training dataset
        batch_size int: Batch size
        target_name string: Indication which target
        alpha_indices np.array((n_alpha_indices, dim_rand)): Set of alpha indices. If not None, used as prediction target and in PceLoss. See rom.pce.multi_indices()
        grid_in_dims tuple(int): Indices of grid dimensions that are used as input, e.g., 0, 1 for tgrid and x1grid.

    Returns:
        x np.array((n_samples*n_tgrid, n_x1grid, n_x2grid, dim_in)): 
        y np.array((n_samples*n_tgrid, n_x1grid, n_x2grid, dim_y)): 
        dim_out int: Number of output dimensions of the NN. Used to distinguish direct and indirect prediction of the target. 
        rand_insts np.array((n_samples*n_tgrid, dim_rand)): Random instances, see if/else statements for more information
        batch_size int: Batch size
    """
    n_tgrid = grid.shape[1] 

    # Concatenate grid and y_args, depending on availability
    x = concat_grid_y_args(grid, y_args, grid_in_dims)

    # Flatten time into samples
    x = x.reshape(-1, *x.shape[2:]) # dim:  n_samples*n_tgrid, n_x1grid, n_x2grid, dim_in
    y = y.reshape(-1, *y.shape[2:]) # dim:  n_samples*n_tgrid*n_xgrid, dim_in

    # Random instances
    if rand_insts is not None:
        rand_insts = np.repeat(rand_insts, n_tgrid, axis=0)

    if alpha_indices is not None:
        raise NotImplementedError('Spectral PINNs not implemented for 2D shape')
    dim_out = y.shape[-1]

    # Set batch_size to equal the full t_grid input
    if batch_size is None:
        batch_size = int(n_tgrid)

    return x, y, dim_out, rand_insts, batch_size


def get_1sample_0D_nn_input(grid, y_args, dim_in, target_name="u", grid_in_dims=(-1,), i=0):
    """
    Returns 1 flattened sample of the grid as test input
    Args:
        grid np.array((n_samples, n_grid)) or 
            (np.array(n_xgrid), np.array(n_tgrid))
            np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_grid)): grid for 1D, 2D, or ND input, respectively
        y_args np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, ..., dim_y_args)): Additional inputs, e.g., forcing terms 
        dim_in
        target_name
        grid_in_dims tuple(int): Indices of the grid dimensions who's will be used as input, e.g., 0, 1 for tgrid and x1grid. -1 for all grids
        i int: sample ID
    Returns:
        x_test np.array((np.prod(n_grid), dim_in)): One input batch.
    """
    x_test, _, _, _, _= flatten_to_0D_data(grid[i:i+1], y=None, y_args=y_args[i:i+1], 
        target_name=target_name, grid_in_dims=grid_in_dims)

    """ TODO: remove once tested with mu_k, etc. inputs
    if len(grid.shape) > 2:
        if y_args is not None:
            grid = np.concatenate((grid, y_args), axis=len(grid.shape)-1) # Concatenate along last axis
        x_test = grid[i].reshape(-1, dim_in) # flatten all first axes
    elif target_name == 'u' or target_name == 'u_true':
        if dim_in == 1:
            x_test = grid[i,:,np.newaxis] # dim: (n_grid, 1)
        elif dim_in == 2:
            #Returns:
            #    x_test np.array((n_tgrid*n_xgrid, 2))
            #x_test, _ = flatten_2D_to_0D(grid, y[i,:,:], y_args, grid_in_dims)
            xgrid, tgrid = grid
            xx, tt = np.meshgrid(xgrid, tgrid)
            x_test = np.concatenate((xx[:,:, np.newaxis], tt[:,:, np.newaxis]), axis=2) # dim: n_tgrid, n_xgrid, 2
            x_test = x_test.reshape(n_tgrid*n_xgrid, 2)
    elif target_name == 'pce_coefs':
        x_test = grid[:,np.newaxis]
    elif target_name == 'k' or target_name == 'k_true':
        x_test = grid[i,:,np.newaxis]
    elif target_name == 'mu_k' or target_name == 'k_eigvecs':
        if dim_in == 1:
            x_test = grid.reshape(-1, 1)
        elif dim_in == 2:
            xgrid, tgrid = grid
            x_test = xgrid.reshape(-1, 1)
    """
    return x_test


def downsample1d_log2(x, tgt_size):
    """
    Downsamples 1D array via average pooling. Only downsampling of factors multiple of 2.
    Args:
        x torch.tensor(N_samples, n_gridx): 1D input vector. Downsamples accros n_gridx
        tgt_size int: Target size 
    Returns
        x torch.tensor(N_samples, tgt_size)
    """
    import torch
    n_gridx = x.shape[-1] - 1
    assert n_gridx%tgt_size == 0, "Downsample requires tgt_size to be factor of 2 lower than input"
    n_downsamples = (np.log2(n_gridx)-np.log2(tgt_size)).astype(int)
    downsample1d = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=1, count_include_pad=False) 
    for i in range(n_downsamples):
        x = downsample1d(x)
    return x

def downsample2d_log2(x, tgt_size):
    """
    Downsamples 2D array via average pooling. Only downsampling of factors multiple of 2.
    Args:
        x torch.tensor(N_samples, n_gridx1, x_gridx2, n_channels): 2D input vector. Downsamples accros n_gridx*
        tgt_size int: Target size 
    Returns
        x torch.tensor(N_samples, tgt_size, tgt_size, n_channels)
    """
    import torch
    n_gridx = x.shape[1] - 1
    assert n_gridx%tgt_size == 0, "Downsample requires tgt_size to be factor of 2 lower than input"
    n_downsamples = (np.log2(n_gridx)-np.log2(tgt_size)).astype(int)
    downsample2d = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1, count_include_pad=False)
    x_coarse = torch.permute(x, (0, -1, 1, 2))
    for i in range(n_downsamples):
        x_coarse = downsample2d(x_coarse)
    x_coarse = torch.permute(x_coarse, (0, 2, 3, 1)) 
    return x_coarse
