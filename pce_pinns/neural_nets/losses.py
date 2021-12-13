"""
Loss functions
Author: Björn Lütjens (lutjens@mit.edu)
"""
import numpy as np
import torch
import pce_pinns.rom.pce as pce

class PceLoss(object):
    def __init__(self, alpha_indices, verbose=False, 
            omit_const_pce_coef=False, rand_insts=None):
        """
        MSE Loss assuming that NN predicts PCE coefficients of target  

        Args:
            alpha_indices np.array((n_alpha_indices, ndim)): Set of alpha indices see rom.pce.multi_indices()
            verbose bool: If true, created verbose prints
            omit_const_pce_coef bool: If true, removes PCE coefficients of zero-th order.
            rand_insts np.array(n_grid*n_samples, n_stoch_dim): Random instances, corresponding to the training set; (opt)
        """
        self.alpha_indices=alpha_indices
        self.verbose=verbose
        if omit_const_pce_coef:
            return NotImplementedError('omitting zero-th order PCE coefs in PceLoss not implemented.')
        self.rand_insts = rand_insts
        self.rand_inst = None # Current random instance
        self.iter = 0
        
    def __call__(self, pce_coefs, y_target):
        """
        Args:
            pce_coefs np.array((n_grid, n_alpha_indices)): Predicted PCE coefficients
            y_target np.array((n_grid, 1)): Ground truth solution
        """
        # !!!TODO!!! rand_inst need to be shape of (n_batches, batch_size, ndim) 
        y_pred = pce.sample_pce_torch(pce_coefs, self.alpha_indices, rand_inst=self.rand_inst)
        pce_loss = torch.mean((y_pred - y_target)**2)

        if self.verbose: print('y: mse, pred, target', pce_loss, y_pred, y_target)
        self.iter += 1
        return pce_loss

    def set_rand_inst(self, batch_id, batch_size):
        """
        Sets:
            rand_inst np.array(batch_size, ndim)
        """
        start_id = batch_id * batch_size
        end_id = batch_id * batch_size + batch_size
        self.rand_inst = self.rand_insts[start_id:end_id,:]

class PcePinnLoss(object):
    def __init__(self, diffeq, u_params, alpha_indices, verbose=False, 
            omit_const_pce_coef=False, rand_insts=None, a_pinn=0.1, debug=False):
        """
        Computes sum of PceLoss and physics-informed NN loss, i.e., the squared residual of the differential equation
        Args:
            diffeq DiffEquation(object): Differential equation object with function residual(). Only necessary for XPINN loss.
            u_params dict(np.array(n_samples, n_grid, dim_out)): Parameters of each differential equation sample; only used for PINN loss
            alpha_indices np.array((n_alpha_indices, ndim)): see rom.pce.multi_indices()
            verbose bool: If true, created verbose prints
            rand_insts np.array(n_grid*n_samples, n_stoch_dim): Random instances, corresponding to the training set; (opt)
            a_pinn float: Ratio of PINN loss in comparison to PCEloss
        """
        self.diffeq = diffeq
        self.u_params = u_params
        self.alpha_indices=alpha_indices
        self.verbose=verbose
        self.rand_insts = rand_insts
        self.rand_inst = None # Current random instance
        self.a_pinn = a_pinn
        self.iter = 0
        self.debug = debug

    def __call__(self, pce_coefs, y_target, inputs, sample_id):
        # Compute PCE loss
        y_pred = pce.sample_pce_torch(pce_coefs, self.alpha_indices, rand_inst=self.rand_inst)
        pce_loss = torch.mean((y_pred - y_target)**2)

        # Compute PINN loss
        y_pred_sample = y_pred.reshape(next(iter(self.u_params.values())).shape[1:])
        u_param = {key: self.u_params[key][sample_id,:,:] for key in self.u_params.keys()}
        res = self.diffeq.residual(y_pred_sample, u_params=u_param)

        pinn_loss = torch.mean(torch.square(res))

        pce_pinn_loss = (1.-self.a_pinn) * pce_loss + self.a_pinn * pinn_loss

        if self.verbose or self.debug: print('y: pce_pinn, pce_loss, pinn_loss, pred, target', pce_pinn_loss, pce_loss, pinn_loss)# y_pred, y_target)
        self.iter += 1
        return pce_pinn_loss, pce_loss, pinn_loss

    def set_rand_inst(self, batch_id, batch_size):
        """
        Sets:
            rand_inst np.array(batch_size, ndim)
        """
        start_id = batch_id * batch_size
        end_id = batch_id * batch_size + batch_size
        self.rand_inst = self.rand_insts[start_id:end_id,:]

def calculate_losses(criterion, y_pred, y_true, y_inputs=None, batch_idx=0, custom_rand_inst=False):
    """
    Calculates all losses
    """
    # Get rand instance id of test batch. TODO: make this independent of n_grid
    if custom_rand_inst:
        batch_size = y_pred.shape[0]
        criterion.set_rand_inst(batch_id=batch_idx, batch_size=batch_size)
    # Compute loss (TODO: integrate PcePinnLoss more elegantly)
    if type(criterion).__name__ == 'PcePinnLoss':
        loss, pce_loss, pinn_loss = criterion(y_pred, y_true, 
            y_inputs=y_inputs, sample_id=batch_idx)
    else:
        loss = criterion(y_pred, y_true)
        pce_loss = 0 if not type(criterion).__name__ == 'PceLoss' else loss
        pinn_loss = 0

    losses = {
        'loss': loss,
        'pce_loss': pce_loss,
        'pinn_loss': pinn_loss
    }
    return losses

def calc_infty_loss(y_pred, y_true, grid_dims=(1,2)):
    """
    Calculates infinity error 
    infty_loss = || y_pred - y_true ||_inf / || y_true ||_inf

    Args:
        y torch.tensor(n_samples, xgrid1, xgrid2, n_ydim)
        y_pred torch.tensor(n_samples, xgrid1, xgrid2, n_ydim)
        grid_dims tuple(): Indices of grid dimensions. Infinity error is taken across these dimensions
    Returns:
        infty_loss torch.tensor(n_samples, n_ydim)
    """
    ydiff_max = torch.amax((torch.abs(y_true) - torch.abs(y_pred)), dim=grid_dims)# (n_samples,n_ch)
    ynorm = torch.amax(torch.abs(y_true), dim=grid_dims) # (n_samples,n_ch)
    infty_loss = torch.div(ydiff_max, ynorm)
    return infty_loss

def calc_mse_loss(y_pred, y_true, grid_dims=(1,2)):
    """
    Calculates mean squared error 
    mse_loss = avg((y_pred)^2 - (y_true)^2)

    Args:
        y torch.tensor(n_samples, xgrid1, xgrid2, n_ydim)
        y_pred torch.tensor(n_samples, xgrid1, xgrid2, n_ydim)
        grid_dims tuple(): Indices of grid dimensions. Infinity error is taken across these dimensions
    Returns:
        mse_loss torch.tensor(n_samples, n_ydim)
    """
    n_px = 1
    for grid_dim in grid_dims:
        n_px = n_px * y_true.shape[grid_dim]
    mse_loss = 1./float(n_px)*(torch.sqrt(torch.sum(torch.pow(y_true - y_pred,2), dim=grid_dims)))# (n_samples,n_ch)
    return mse_loss