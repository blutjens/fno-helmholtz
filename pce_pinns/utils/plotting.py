"""
Accumulation of plotting functions
"""
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_msmt(x_obs, xgrid, u_obs, k_true, Y_true):
    """ Plots the measurements

    Plots the measurements of the observed solution, u_obs, 
    parameters, k_true, and log-parameters, Y_true, against 
    the grid, x_obs
    
    Args:
        x_obs 
        xgrid
        u_obs 
        k_true 
        Y_true 
    """
    fig, axs = plt.subplots(nrows=1, ncols=3, dpi=300)
    axs[0].plot(x_obs, u_obs)
    axs[0].set_xlabel(r'location, $x$')
    axs[0].set_ylabel(r'measurements, $u$')
    axs[1].plot(xgrid, k_true)
    axs[1].set_xlabel(r'location, $x$')
    axs[1].set_ylabel(r'permeability, $k$')
    axs[2].plot(xgrid, Y_true)
    axs[2].set_xlabel(r'location, $x$')
    axs[2].set_ylabel(r'log-permeability, $Y$')
    fig.tight_layout()
    fig.savefig('doc/figures/ai_msmts.png')
    plt.close(fig)
    
"""
Diff eq plots
"""
def ridge_plot_prob_dist_at_sample_x(xs, uxs):
    '''
    Creates ridge plot, plotting a probability distribution over all values in uxs 
    Source: https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    
    Args:
        xs np.array(n_curves)
        uxs np.array(n_curves, n_samples)
    '''
    import matplotlib.gridspec as grid_spec

    gs = grid_spec.GridSpec(len(xs),1)
    fig = plt.figure(figsize=(16,9))

    axs = []
    for i, x in enumerate(xs):
        x = xs[i]

        # creating new axes object
        axs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        histogram = axs[-1].hist(uxs[i,:],  density=True, color="blue", alpha=0.6)#bins='auto',#, lw=1) bins='auto', density=True, 
        #axs[-1].fill_between(range(uxs[i,:].shape[0]), uxs[i,:] , alpha=1,color=colors[i])

        # setting uniform x and y lims
        axs[-1].set_ylim(0,20)
        u_min = np.min(uxs[:,:])#.mean(axis=0))
        u_max = np.max(uxs[:,:])#.mean(axis=0))
        axs[-1].set_xlim(u_min, u_max)

        # make background transparent
        rect = axs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        axs[-1].set_yticklabels([])

        # Label x-axis
        if i == len(xs)-1:
            axs[-1].set_xlabel(r"$u(x)$", fontsize=16)
        else:
            axs[-1].set_xticklabels([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            axs[-1].spines[s].set_visible(False)

        # Label y-axis
        adj_x = x#.replace(" ","\n")
        axs[-1].set_ylabel(r'$x=$'+str(x))
        #text(-0.02,0,adj_x,fontweight="bold",fontsize=14,ha="right")


    gs.update(hspace=-0.0)

    fig.text(0.07,0.85,"prob dist at sample x",fontsize=20)
    #plt.title("prob dist at sample x",fontsize=20)

    plt.tight_layout()
    plt.savefig('doc/figures/plot_prob_dist_at_sample_x.png')

    plt.show()
    plt.close(fig)

def plot_sol(xgrid, u, ux, ux_stats, x, xgrid_true, u_true=None):
    """
    Plot solution and statistics
    
    Args:
        xgrid_true np.array(n_grid): Grid of true solution
        u_true np.array(n_grid): True solution
        ux np.array(n_samples): Samples of solution, u, at x
        ux_stats {np.array(n_samples)}: Stats about the set of solutions, u(x)
    """
    fig, axs = plt.subplots(2,3, figsize=(15,10), dpi=150)
    # Plot sample solutions
    n_plotted_samples = 4
    for i in range(n_plotted_samples):
        axs[0,0].plot(xgrid, u[-i,:])
    axs[0,0].set_xlabel(r'x')
    axs[0,0].set_ylabel(r'$u_{n=0}(x,\omega)$')
    axs[0,0].set_title('sample solutions')

    # Plot mean and std dev of solution
    print('u', u.shape)
    if u_true is not None:
        axs[0,1].plot(xgrid_true, u_true, color='black', label=r'$u(x, k_{true})$')
    u_mean = u.mean(axis=0)
    u_std = u.std(axis=0)
    axs[0,1].plot(xgrid, u_mean[:], label=r'$\mathbf{E}_w[u(x,w)]$')
    axs[0,1].set_xlabel('x')
    #axs[0,1].set_ylabel(r'$u(x{=}'+str(x)+', \omega)$')
    axs[0,1].fill_between(xgrid,#range(u_mean.shape[0]),
        u_mean + u_std, u_mean - u_std, 
        alpha=0.3, color='blue',
        label=r'$\mathbf{E}_w[u(x,w)] \pm \sigma_n$')
    axs[0,1].set_title('mean and std dev of solution')
    axs[0,1].legend()


    axs[1,0].plot(ux[:])
    axs[1,0].set_xlabel('sample id, n')
    axs[1,0].set_ylabel(r'$u_n(x='+str(x)+', \omega)$')
    axs[1,0].set_title('sol at x over iterations')

    axs[1,1].plot(ux_stats['ux_cum_mean'][:], label=r'$\bar u_n = \mathbf{E}_{n^* \in \{0,..., n\}}[u_{n^*}(x{=}'+str(x)+',\omega)]$')
    axs[1,1].fill_between(range(ux_stats['ux_cum_mean'].shape[0]), 
        ux_stats['ux_cum_mean'][:] + ux_stats['ux_cum_std'][:], 
        ux_stats['ux_cum_mean'][:] - ux_stats['ux_cum_std'][:], 
        alpha=0.4, color='blue',
        label=r'$\bar u_n \pm \sigma_n$')
    axs[1,1].fill_between(range(ux_stats['ux_cum_mean'].shape[0]), 
        ux_stats['ux_cum_mean'][:] + ux_stats['ux_cum_sem'][:], 
        ux_stats['ux_cum_mean'][:] - ux_stats['ux_cum_sem'][:], 
        alpha=0.4, color='black',
        label=r'$\bar u_n \pm$ std err$_n$')
    axs[1,1].fill_between(range(ux_stats['ux_cum_mean'].shape[0]), 
        ux_stats['ux_conf_int'][:,0], ux_stats['ux_conf_int'][:,1], 
        alpha=0.4, color='orange',
        label=r'$95\% conf. interval; P $')
    axs[1,1].set_xlabel('sample id, n')
    axs[1,1].set_ylabel(r'$u(x{=}'+str(x)+', \omega)$')
    axs[1,1].set_ylim((ux_stats['ux_cum_mean'].min()-np.nanmax(ux_stats['ux_cum_std']), ux_stats['ux_cum_mean'].max()+np.nanmax(ux_stats['ux_cum_std'])))
    axs[1,1].set_title('sample mean over iterations')
    axs[1,1].legend()

    axs[1,2].plot(ux_stats['ux_cum_var'][:], label=r'$\sigma^2_n$')
    axs[1,2].set_xlabel('sample id, n')
    axs[1,2].set_ylabel(r'$\sigma^2(u(x{=}'+str(x)+', \omega)$')
    axs[1,2].fill_between(range(ux_stats['ux_cum_var'].shape[0]), 
        ux_stats['ux_cum_var'][:] + ux_stats['ux_cum_sem'][:], 
        ux_stats['ux_cum_var'][:] - ux_stats['ux_cum_sem'][:], 
        alpha=0.4, color='black',
        label=r'$\sigma^2_n \pm$ std err$_n$')
    axs[1,2].set_title('sample std dev. at x over iterations')
    axs[1,2].legend()

    #axs[1,1].plot(ux_stats['ux_conf_int'][:])
    #axs[1,1].set_xlabel('sample id, n')
    #axs[1,1].set_ylabel(r'$V[E[u_{n\prime}(x='+str(x)+',w)]_0^n]$')
    #axs[1,1].set_title('95\% conf. int.')


    fig.tight_layout()
    plt.savefig('doc/figures/proj1_3.png')

    # Plot prob. dist. at sample locations
    xs = np.linspace(0,1, 5)#[0., 0.25, 0.5, 0.75]
    ux_samples = np.empty((len(xs), u.shape[0]))
    for i, x_sample in enumerate(xs):
        x_id = (np.abs(xgrid - x_sample)).argmin()
        ux_samples[i,:] = u[:, x_id]

    ridge_plot_prob_dist_at_sample_x(xs, ux_samples)
    plt.close(fig)
    
def plot_k(xgrid, k, k_true, y_gp_mean, y_gp_cov):
    """
    Plot k samples vs. ground-truth

    Args:
        xgrid: np.array(n_grid): Grid with equidistant grid spacing, n_grid
        k: np.array(n_samples, n_grid): Samples of permeability, k
        k_true: np.array(n_grid): Ground truth of permeability, k
    """
    kmean = k.mean(axis=0)
    kvar = k.var(axis=0)
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
    axs[0].plot(xgrid, k_true, color='black', label=r'true $k$')
    axs[0].plot(xgrid, kmean, color='blue')#, label=r'post. $\bar k_n$')
    axs[0].fill_between(xgrid, kmean+kvar, kmean-kvar,alpha=0.4,color='blue', label=r'post. $\bar k_n \pm \sigma_k^2$')
    axs[0].set_ylim((k_true.min()-5.,k_true.max()+5.))
    axs[0].set_xlabel(r'location, $x$')
    axs[0].set_ylabel(r'permeability, $k$')
    axs[0].legend()
    
    # log permeability
    Y = np.log(k)
    Y_true = np.log(k_true)
    Ymean = Y.mean(axis=0)
    Yvar = Y.var(axis=0)
    # True
    axs[1].plot(xgrid, Y_true, color='black', label=r'true $Y$')
    # Prior
    axs[1].plot(xgrid, y_gp_mean, color='green')#, label=r'prior $\bar Y_n$')
    axs[1].fill_between(xgrid, y_gp_mean+np.diag(y_gp_cov), y_gp_mean-np.diag(y_gp_cov),alpha=0.4,color='green', label=r'prior $\bar Y_n \pm \bar\sigma_Y^2$')
    # Posterior
    axs[1].plot(xgrid, Ymean, color='blue')#, label=r'post. $\bar Y_n$')
    axs[1].fill_between(xgrid, Ymean+Yvar, Ymean-Yvar,alpha=0.4,color='blue', label=r'post. $\bar Y_n \pm \bar\sigma_Y^2$')
    axs[1].set_xlabel(r'location, $x$')
    axs[1].set_ylabel(r'log-permeability, $Y$')
    axs[1].legend()
        
    fig.tight_layout()
    fig.savefig('doc/figures/k_gp_vs_msmts.png')
    plt.close(fig)

def plot_source(xgrid, source):
    """
    Plots the left boundary condition source term
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(xgrid, source)
    axs.set_xlabel(r'location, $x$')
    axs.set_ylabel(r'source, $s$')
    fig.tight_layout()
    fig.savefig('doc/figures/source_injection_wells.png')
    plt.close(fig)
    
"""
PCE and KL-expansion plots
"""
import numpy.polynomial.hermite_e as H

def plot_trunc_err(truncs, trunc_err):
    """
    Source: https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    
    Args
        truncs np.array(n_truncations)
        trunc_err np.array(n_truncations)
    """
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.plot(truncs, trunc_err)
    axs.set_xlabel(r'truncation, $r$')
    axs.set_ylabel(r'truncation err.')
    fig.savefig('doc/figures/ai_trunc_err.png')
    plt.close(fig)
    plt.close(fig)

def plot_pce(xgrids, exp_y_pce, sample_pce, pce_dim, alpha_indices, c_alphas):
    """
    Creates various plots to analyse the polynomial chaos expansion
    Args:
        xgrids: np.array(n_dim, n_grid): Gaussian samples of stochastic dimensions, n_dim, with equidistant grid spacing, n_grid
        exp_Y np.array(n1_grid,n2_grid): Exponential of approximated stochastic process, exp(Y)
        sample_pce function()->y,exp_y,trunc_err,coefs,xi: 
            Function that draws samples of log-permeability, Y, as approximated by PCE
        pce_dim np.array(p): Polynomial degree
        alpha_indices np.array(ndim,n_alpha_indices): Lattice of alpha vectors, that represent the order of polynomial basis 
        c_alphas (np.array(n_grid, n_alpha_indices)): PCE coefficients
    """
    from pce_pinns.rom.pce import Herm

    n_grid = xgrids[0,:].shape[0]

    # Plot basis polynomials
    fig = plt.figure(figsize=(9,5))
    x_poly = np.linspace(-4,4)
    for alpha in range(pce_dim):
        herm_alpha = H.hermeval(x_poly, Herm(alpha))
        plt.plot(x_poly, herm_alpha, label=r'$\alpha_i=$'+str(alpha))
    plt.xlabel(r'location, $x$')
    plt.ylabel(r'Probabilists Hermite polynomial, $He_{\alpha_i}(x)$')
    plt.ylim((-10,20))
    plt.tight_layout()
    plt.legend()
    plt.savefig('doc/figures/hermite_poly.png')
    plt.close()
    plt.close(fig)

    # Plot the stochastic process
    fig = plt.figure(figsize=(9,5))
    #for a, y_alpha in enumerate(y_alphas):
    #    plt.plot(xgrids[dim,:], y_alpha, label=r'$y(\alpha\leq$' + str(a) + ')')
    plt.plot(xgrids[0,:], exp_y_pce, label=r'$y$')
    plt.xlabel(r'location $x$')
    plt.ylabel(r'log-permeability $x$')
    plt.tight_layout()
    plt.legend()
    plt.savefig('doc/figures/polynomial_chaos.png')
    plt.close()
    plt.close(fig)

    # Plot mean and variance of PCE-estimated stochastic process 
    fig = plt.figure(figsize=(9,5))
    n_samples_plt = 100
    print('PCE dim', pce_dim)
    H_alpha_plt = np.zeros((n_samples_plt, pce_dim))
    Y_alpha_plt = np.zeros((n_samples_plt, n_grid))
    exp_Y_plt = np.zeros((n_samples_plt, n_grid))
    for n in range(100):
        #xi = np.random.normal(0,1,ndim) # one random variable per stochastic dimension
        #for a, alpha_vec in enumerate(alpha_indices):
        #    herm_alpha = np.zeros(ndim)
        #    for idx, alpha_i in enumerate(alpha_vec):
        #        herm_alpha[idx] = H.hermeval(xi[idx], Herm(alpha_i))
        #    exp_Y_plt[n,:] += c_alphas[:,a] * np.prod(herm_alpha)
        _, exp_Y_plt[n,:], _, _, _ = sample_pce()

    exp_Y_plt_mean = exp_Y_plt.mean(axis=0)
    exp_Y_plt_std = exp_Y_plt.std(axis=0)
    plt.plot(xgrids[0,:], exp_Y_plt_mean)
    plt.fill_between(xgrids[0,:], exp_Y_plt_mean+exp_Y_plt_std, exp_Y_plt_mean-exp_Y_plt_std,alpha=0.4,color='blue', label=r'$Y_{PCE} \pm \sigma$')
    plt.xlabel(r'location, $x$')
    plt.ylabel(r'PCE of permeability, $PCE(\exp(Y))$')
    plt.savefig('doc/figures/pce_exp_of_y.png')
    plt.close()
    plt.close(fig)

    # Plot PCE coefficients
    fig = plt.figure(figsize=(9,5))
    for a, alpha_vec in enumerate(alpha_indices):
        plt.plot(xgrids[0,:], c_alphas[:,a], label=r'$C_{\vec \alpha=' + str(alpha_vec) + '}$)')
    plt.xlabel(r'location $x$')
    plt.ylabel(r'PCE coefficient, $C_{\vec\alpha}(x)$')
    plt.tight_layout()
    plt.legend()
    plt.savefig('doc/figures/pce_coefs.png')
    plt.close()
    plt.close(fig)

    return 1

def plot_kl_expansion(xgrid, Y, trunc, eigvals, eigvecs, sample_kl):
    """
    Creates various plots to analyse the polynomial chaos expansion
    Args:
        xgrid np.array(n_grid): 1D grid points
        Y np.array(n_grid): Sample of the apprximated stochastic process 
        trunc (int): Number of non-truncated eigenvalues for Karhunen-Loeve expansion
        eigvals np.array((trunc)): Major eigenvalues of the cov matrix
        eigvecs np.array((n_grid, trunc)): Major eigenvectors of the cov matrix
        sample_kl fn(np.array((n_grid,trunc)): Function to quickly query KL-expansion for new z  
    """
    n_grid = xgrid.shape[0]

    # Plot eigenvalues
    fig = plt.figure(figsize=(9,5))
    plt.plot(range(eigvals.shape[0]), eigvals)
    plt.vlines(trunc,ymin=eigvals.min(),ymax=eigvals.max(), linestyles='--', label=r'truncation, $r=$'+str(trunc))
    plt.xlabel(r'mode id')
    plt.ylabel(r'eigenvalue of prior covariance')
    plt.legend()
    plt.savefig('doc/figures/kl_exp_eigvals.png')
    plt.close()
    plt.close(fig)

    # Plot eigenvectors
    fig = plt.figure(figsize=(9,5))
    for t in range(trunc):
        plt.plot(xgrid, eigvecs[:,t], label=r'$\phi_'+str(t)+'$')
    plt.xlabel(r'location, $x$')
    plt.ylabel(r'eigenvector of prior covariance')
    plt.legend()
    plt.savefig('doc/figures/kl_exp_eigvecs.png')
    plt.close()
    plt.close(fig)

    # Plot KL expansion
    fig = plt.figure(figsize=(9,5))
    plt.plot(xgrid, Y, color='gray', label='first sample')
    n_samples_plt = 1000
    Y_plt = np.zeros((n_samples_plt, n_grid))
    for n in range(n_samples_plt):
        #z_plt = np.repeat(np.random.normal(0., 1., trunc)[np.newaxis,:], repeats=n_grid, axis=0)
        #Y_plt[n,:] = kl_fn(z_plt) 
        Y_plt[n,:], _, _ = sample_kl()
    Y_plt_mean = Y_plt.mean(axis=0)
    Y_plt_var = Y_plt.var(axis=0)
    plt.plot(xgrid, Y_plt_mean,color='blue')
    plt.fill_between(xgrid, Y_plt_mean+Y_plt_var, Y_plt_mean-Y_plt_var,alpha=0.4,color='blue', label=r'$Y_{KL} \pm \sigma^2$')
    plt.xlabel(r'location, $x$')
    plt.ylabel(r'KL(log-permeability), $KL(Y)$')
    plt.legend()
    plt.savefig('doc/figures/kl_exp_log_perm.png')
    plt.close()
    plt.close(fig)

    return 1

"""
MC sampler plots
"""
def plot_mh_sampler(chain, lnprobs, disable_ticks=False):
    # Plot metropolis hastings chain and log posterios
    sample_ids = np.arange(chain.shape[0], dtype=int)
    ndim = chain.shape[1]

    # Plot mh chain
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        ax.plot(sample_ids, chain[:,i], label=f'{i}')
        if col_id == 0: 
            ax.set_ylabel(r'parameter, $z$')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'sample id, $t$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('doc/figures/mh_chain.png')

    # Plot mh lnprobs
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(sample_ids, lnprobs[:])
    axs.set_ylabel(r'log-probability, $p(u^{(t)}|X)$')
    axs.set_xlabel(r'sample id, $t$')
    axs.set_ylim((-5.,lnprobs.max()+1.))
    fig.tight_layout()
    fig.savefig('doc/figures/mh_lnprobs.png')
    plt.close(fig)

    # Plot autocorrelation of parameters as fn of lag (only useful after warmup)
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        autocorr_i = np.correlate(chain[:,i], chain[:,i], mode='full')
        autocorr_i = autocorr_i[int(autocorr_i.size/2.):] # take 0 to +inf
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        ax.plot(sample_ids, autocorr_i, label=f'{i}')
        if col_id == 0: 
            ax.set_ylabel(r'autocorrelation')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'sample id, $t$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('doc/figures/mh_autocorr.png')
    plt.close(fig)

    # Plot posterior distribution of parameters (only useful after warmup)
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        hist_i = ax.hist(chain[:,i], density=True, color="blue", alpha=0.6, label=f'{i}')#, bins=50)#bins='auto',#, lw=1) bins='auto', density=True, 
        if col_id == 0: 
            ax.set_ylabel(r'posterior $p(z|X)$')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'$z$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('doc/figures/mh_param_post.png')
    plt.close(fig)

def plot_ensemble_kf(z_post_samples,disable_ticks=False):
    ndim = z_post_samples.shape[-1]
    # Plot posterior distribution of parameters 
    nrows = np.ceil(np.sqrt(ndim)).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, dpi=300)
    for i in range(ndim):
        if ndim > 1: 
            col_id = i%nrows
            row_id = np.floor((i-col_id)/nrows).astype(int)
            ax = axs[row_id, col_id]
        hist_i = ax.hist(z_post_samples[:,i], density=True, color="blue", alpha=0.6, label=f'{i}')#, bins=50)#bins='auto',#, lw=1) bins='auto', density=True, 
        if col_id == 0: 
            ax.set_ylabel(r'posterior $p(z|X)$')
        elif disable_ticks:
            ax.set_yticks([])
        if row_id == nrows-1:
            ax.set_xlabel(r'$z$')
        elif disable_ticks:
            ax.set_xticks([])
    fig.tight_layout()
    fig.savefig('doc/figures/enkf_param_post.png')
    plt.close(fig)

def plot_accept_rates(accept_rates):
    """
    Create plot of acceptance rates
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    n_samples = accept_rates.shape[0]
    axs.plot(range(n_samples), accept_rates)
    axs.set_xlabel(r'sample_id, $t$')
    axs.set_ylabel(r'accept rates')
        
    fig.tight_layout()
    fig.savefig('doc/figures/mh_accept_rates.png')
    plt.close(fig)

"""
FNO plots
"""
def plot_fno_training_curve(loss_cfg, dir_plot):
    fig,axs = plt.subplots(1,1)
    axs.set_xlabel('epochs')
    axs.set_ylabel('mse')
    axs.set_yscale('log')
    axs.plot(loss_cfg["training"], 'r.', label="train")
    axs.plot(loss_cfg["validation"], 'k.', label="val")
    axs.legend()
    plt.savefig(dir_plot/"training_curve.png")
    plt.close(fig) 

def plot2Dfeats_fno(x, fname, featnames=None):
    nfeat = x.shape[-1]
    fig, axes = plt.subplots(1,nfeat, figsize=(nfeat*4,3))
    
    for ifeat in range(nfeat):
        if nfeat>1:
            plot = axes[ifeat].imshow(x[:,:,ifeat], cmap='jet', origin='lower')
            cax= fig.colorbar(plot, ax=axes[ifeat])
            if featnames is not None:  axes[ifeat].set_title(featnames[ifeat])
        else:
            plot = axes.imshow(x[:,:,ifeat], cmap='jet', origin='lower')
            cax = fig.colorbar(plot, ax=axes)
            if featnames is not None:  axes.set_title(featnames[ifeat])
    fig.tight_layout()
    plt.savefig(fname)
    plt.close(fig)
    return fname

def plot_fno_helmholtz(xgrid, ygrid, sol_pred, sol_target, fname='sol_pred'):
    xx, yy = np.meshgrid(xgrid, np.flip(ygrid), indexing='xy')#-1:1:.01, -1:1:.01]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), dpi=100)
    cax0 = axs[0].contourf(xx, yy, sol_pred, levels=50)
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title(r'Predicted solution, $Re(\hat u)$') 
    axs[0].set_xlabel(r'x in $m$')
    axs[0].set_ylabel(r'y in $m$')
    axs[0].set_aspect('equal', adjustable='box')
    cax1 = axs[1].contourf(xx, yy, sol_target, levels=50)
    fig.colorbar(cax1, ax=axs[1])
    axs[1].set_title(r'Target solution, $Re(u)$') 
    axs[1].set_xlabel(r'x in $m$')
    axs[1].set_ylabel(r'y in $m$')
    axs[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    plt.savefig(f'doc/figures/helmholtz/fno/{fname}.png')
    plt.close(fig)
    return 0

def plot_fno_helmholtz_diff(xgrid, ygrid, sol_pred, sol_target, fname='sol_diff'):
    xx, yy = np.meshgrid(xgrid, np.flip(ygrid), indexing='xy')#-1:1:.01, -1:1:.01]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), dpi=100)
    cax0 = axs[0].contourf(xx, yy, sol_pred-sol_target, levels=50)
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title(r'Error, $Re(\hat u_{FNO} - \hat u_{FD})$')
    axs[0].set_xlabel(r'x in $m$')
    axs[0].set_ylabel(r'y in $m$')
    axs[0].set_aspect('equal', adjustable='box')
    
    fig.tight_layout()
    plt.savefig(f'doc/figures/helmholtz/fno/{fname}.png')
    plt.close(fig)
    return 1

def plot_fno_runtimes_vs_N(Ns, ts, fname='runtimes'):
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=100)# figsize=(10,3))
    axs.plot(Ns, ts, label='FNO', color='black')

    # Plot finite difference and Fourier preconditioned data.
    ts_fno_bkup = [1.44213588e+01, 3.33246908e+00, 4.47575378e-01, 7.47245073e-02, 2.03821206e-02, 7.27848530e-03, 3.84319305e-03, 2.65371323e-03, 2.35928297e-03, 2.17975616e-03, 2.13267565e-03] 
    ts_dst_cp = [0.003278226852416992,0.002307097911834717,0.0025220203399658205,0.002579813003540039,0.0034314894676208494,0.006736712455749512,0.02950010299682617,0.24340070247650147,0.3757796049118042, 3.15043625831604, 22.484691739082336]
    ts_fd_cp = [0.002301952838897705,0.0022307801246643066,0.002913227081298828,0.005261270999908447,0.01675771951675415,0.07563373327255249,0.5706490850448609,4.241608152389526, 37.76028895378113, 487.4126236438751]
    logN = int(np.log2(4096))
    Ns_fd = np.logspace(2, logN, num=logN-1, endpoint=True, base=2.0, dtype=int, axis=0)
    # axs.plot(Ns_fd[4:], 1e-5*Ns_fd[4:]* np.log2(Ns_fd[4:]), "--", color="black", label=r'$O(N\log N)$')
    # axs.plot(Ns_fd[4:], 5e-7*np.power(Ns_fd[4:],2), "--", color="black", label=r'$O(n^2)$')
    axs.plot(Ns_fd[:-1], ts_fd_cp, label='Finite Diff.', color='orange')
    axs.plot(Ns_fd[4:], 1e-7*np.power(Ns_fd[4:],3), "--", color="orange", label=r'$O(n^3)$')
    axs.plot(Ns_fd, ts_dst_cp, label='Fourier Prec.', color='blue')
    axs.plot(Ns_fd[4:], 1e-7*np.power(Ns_fd[4:],2)* np.log2(Ns_fd[4:]), "--", color="blue", label=r'$O(n^2\log n)$')

    axs.set_title('Runtime (log-log)')
    axs.set_xlabel('n')
    axs.set_ylabel('time(inference|solve) in s')
    axs.set_yscale('log', base=2)
    axs.set_xscale('log', base=2)
    axs.legend()
    plt.tight_layout()
    plt.savefig(f'doc/figures/helmholtz/fno/{fname}.png')
    plt.close(fig)
    return 1

def plot_fno_accuracy_vs_N(Ns, mse, infty, fname='losses_over_n'):
   
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=100)# figsize=(10,3))
    axs[0].plot(Ns, mse, label='MSE', color='black')
    axs[0].set_title('MSE over n')
    axs[0].set_xlabel('n')
    axs[0].set_ylabel(r'MSE($u_{FD}, u_{FNO}$)')
    axs[0].set_yscale('log', base=10)
    axs[0].set_xscale('log', base=2)
    axs[0].legend()

    axs[1].plot(Ns, infty, label=r'$|\cdot|_\infty$', color='black')
    axs[1].set_title('Infinity loss over n')
    axs[1].set_xlabel('n')
    axs[1].set_ylabel(r'$\frac{||u_{FNO}-u_{FD}||_\infty}{||u_{FD}||_\infty}$')
    axs[1].set_yscale('log', base=10)
    axs[1].set_xscale('log', base=2)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f'doc/figures/helmholtz/fno/{fname}.png')
    plt.close(fig)

    return 1


def plot_lorenz96_fno(X, Y):
    """
    """
    # Plot ensemble
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    plt.pcolormesh(X[...,0], X[...,1], Y[...,0], cmap = cm.gray) 

    #axs.set_ylabel(r'$Y_t$')
    #axs.set_xlabel(r'time, $t$')
    #axs.legend(loc=1, prop={'size':6})

    axs.set_title("Lorenz '96")
    fig.tight_layout()
    fig.savefig('doc/figures/lorenz96/fno_test.png')
    plt.close(fig)



"""
Neural net plots
""" 
def plot_nn_k_samples(xgrid, k_samples):
    plt.figure(figsize=(15,8))
    n_samples = k_samples.shape[0]
    k_mean = k_samples.mean(axis=0)
    k_std = k_samples.std(axis=0)
    plt.plot(xgrid, k_mean, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_mean + k_std, k_mean - k_std, 
        alpha=0.3, color='blue',
        label=r'$\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"diffusion, $k$")
    plt.legend()
    plt.title(r'$k$')
    plt.savefig('doc/figures/nn_k_samples.png')
    plt.close(fig)

def plot_train_curve(loss_stats):
    #train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    plt.figure(figsize=(15,8))
    plt.plot(np.arange(len(loss_stats['train']))+1, loss_stats['train'])
    plt.yscale('log')
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title('Loss/Epoch')
    plt.savefig('doc/figures/train_val_loss.png')
    plt.close(fig)

def plot_nn_pred_1D(x, y):
    plt.figure(figsize=(15,8))
    dim_out = y.shape[1]
    for i in range(dim_out):
        plt.plot(x, y[:,i], label=r'$'+str(i)+'$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"NN")
    plt.legend()
    plt.title('NN predictions')
    plt.savefig('doc/figures/nn_pred.png')
    plt.close(fig)

def plot_nn_pred_pce_coefs_2D(x, y):
    """
    x np.array(n_samples, n_tgrid, n_xgrid, dim_in)
    y np.array(n_samples, n_tgrid, n_xgrid, dim_out)
    """
    n_samples = x.shape[0]
    dim_out = y.shape[-1]
    sample_id = 0
    for t in [0, int(x.shape[1]/2), -1]: # Plot init, mid, and terminal time
        plt.figure(figsize=(15,8))
        for i in range(dim_out):
                plt.plot(x[sample_id,t,:,1], y[sample_id,t,:,i], label=r'$C_{\alpha}$, '+str(i))
        plt.xlabel(r"location, $x$")
        plt.ylabel(r"$\hat C_\alpha(x)$")
        plt.legend()
        plt.title('PCE coefs')
        plt.savefig('doc/figures/localAdvDiff/nn_pce_coefs_t_'+str(t)+'.png')
        plt.close(fig)

def plot_nn_pred_2D(x, y_pred, y_target):
    """
    x np.array(n_samples, n_tgrid, n_xgrid, dim_in)
    y_pred np.array(n_samples, n_tgrid, n_xgrid, 1)
    y_target np.array(n_samples, n_tgrid, n_xgrid, 1)
    """
    n_samples = x.shape[0]
    n_tgrid = x.shape[1]
    n_xgrid = x.shape[2]
    x_sample_id = 0

    # Plot
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
    colors = cm.get_cmap('tab20')
    for idx, t in enumerate([0, int(n_tgrid/2), n_tgrid-1]):
        # Pred
        axs[0].plot(x[x_sample_id,t,:,1], np.mean(y_pred[:,t,:,0], axis=0),
            color=colors.colors[2*idx], label=r'$\hat y(t_{'+str(t)+'})$')
        axs[0].fill_between(x[x_sample_id,t,:,1], np.mean(y_pred[:,t,:,0], axis=0) - np.std(y_pred[:,t,:,0], axis=0), 
                    np.mean(y_pred[:,t,:,0], axis=0) + np.std(y_pred[:,t,:,0], axis=0),
                    color=colors.colors[2*idx], alpha=0.3)
        # Target
        axs[0].plot(x[x_sample_id,t,:,1], np.mean(y_target[:,t,:,0], axis=0), 
            color=colors.colors[2*idx+1], linewidth=4, linestyle=':', label=r'$y(t_{'+str(t)+'})$')
        axs[0].fill_between(x[x_sample_id,t,:,1], np.mean(y_target[:,t,:,0], axis=0) - np.std(y_target[:,t,:,0], axis=0), 
                    np.mean(y_target[:,t,:,0], axis=0) + np.std(y_target[:,t,:,0], axis=0), 
                    color=colors.colors[2*idx+1], alpha=0.3)
    axs[0].set_xlabel(r'location, $x$')
    axs[0].set_ylabel(r'$y$')
    #axs[0].set_ylim(-1, 1)
    axs[0].legend()

    for idx, xi in enumerate([0, int(n_xgrid/2), n_xgrid-1]):
        # Plot predicted
        axs[1].plot(x[x_sample_id,:,xi,0], np.mean(y_pred[:,:,xi,0], axis=0), 
            color=colors.colors[2*idx], label=r'$\hat y(x_{'+str(xi)+'})$')
        axs[1].fill_between(x[x_sample_id,:,xi,0], np.mean(y_pred[:,:,xi,0], axis=0) - np.std(y_pred[:,:,xi,0], axis=0), 
                    np.mean(y_pred[:,:,xi,0], axis=0) + np.std(y_pred[:,:,xi,0], axis=0), 
                    color=colors.colors[2*idx], alpha=0.3)
        # Plot Target
        axs[1].plot(x[x_sample_id,:,xi,0], np.mean(y_target[:,:,xi,0], axis=0), 
            color=colors.colors[2*idx+1], linewidth=4, linestyle=':', label=r'$y(x_{'+str(xi)+'})$')
        axs[1].fill_between(x[x_sample_id,:,xi,0], np.mean(y_target[:,:,xi,0], axis=0) - np.std(y_target[:,:,xi,0], axis=0), 
                    np.mean(y_target[:,:,xi,0], axis=0) + np.std(y_target[:,:,xi,0], axis=0), 
                    color=colors.colors[2*idx+1], alpha=0.3)
    axs[1].set_xlabel(r'time, $t$')
    axs[1].set_ylabel(r'$y$')
    #axs[1].set_ylim(-1, 1)
    axs[1].legend()
        
    fig.tight_layout()
    fig.savefig('doc/figures/localAdvDiff/nn_pred_2D.png')
    plt.close(fig)

def plot_nn_pred_2D_extra_std(x, y_pred, y_target):
    """
    x np.array(n_samples, n_tgrid, n_xgrid, dim_in)
    y_pred np.array(n_samples, n_tgrid, n_xgrid, 1)
    y_target np.array(n_samples, n_tgrid, n_xgrid, 1)
    """
    n_samples = x.shape[0]
    n_tgrid = x.shape[1]
    n_xgrid = x.shape[2]
    x_sample_id = 0

    # Plot
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=300)
    colors = cm.get_cmap('tab20')
    for idx, t in enumerate([0, int(n_tgrid/2), n_tgrid-1]):
        # Pred
        axs[0,0].plot(x[x_sample_id,t,:,1], np.mean(y_pred[:,t,:,0], axis=0),
            color=colors.colors[2*idx], label=r'$\hat y(t_{'+str(t)+'})$')
        axs[1,0].plot(x[x_sample_id,t,:,1], np.std(y_pred[:,t,:,0], axis=0), 
                    color=colors.colors[2*idx], label=r'$\hat y(t_{'+str(t)+'})$')
        # Target
        axs[0,0].plot(x[x_sample_id,t,:,1], np.mean(y_target[:,t,:,0], axis=0), 
            color=colors.colors[2*idx+1], linewidth=4, linestyle=':', label=r'$y(t_{'+str(t)+'})$')
        axs[1,0].plot(x[x_sample_id,t,:,1], np.std(y_target[:,t,:,0], axis=0), 
                    color=colors.colors[2*idx+1], linewidth=4, linestyle=':', label=r'$y(t_{'+str(t)+'})$')
    axs[1,0].set_xlabel(r'location, $x$')
    axs[0,0].set_ylabel(r'$y$')
    axs[1,0].set_ylabel(r'$\sigma(y)$')
    #axs[0].set_ylim(-1, 1)
    #axs[0,0].legend()
    axs[1,0].legend()

    for idx, xi in enumerate([0, int(n_xgrid/2), n_xgrid-1]):
        # Plot predicted
        axs[0,1].plot(x[x_sample_id,:,xi,0], np.mean(y_pred[:,:,xi,0], axis=0), 
            color=colors.colors[2*idx], label=r'$\hat y(x_{'+str(xi)+'})$')
        axs[1,1].plot(x[x_sample_id,:,xi,0], np.std(y_pred[:,:,xi,0], axis=0), 
                    color=colors.colors[2*idx], label=r'$\hat y(x_{'+str(xi)+'})$')
        # Plot Target
        axs[0,1].plot(x[x_sample_id,:,xi,0], np.mean(y_target[:,:,xi,0], axis=0), 
            color=colors.colors[2*idx+1], linewidth=4, linestyle=':', label=r'$y(x_{'+str(xi)+'})$')
        axs[1,1].plot(x[x_sample_id,:,xi,0], np.mean(y_target[:,:,xi,0], axis=0),
                    color=colors.colors[2*idx+1], linewidth=4, linestyle=':', label=r'$y(x_{'+str(xi)+'})$')
    axs[1,1].set_xlabel(r'time, $t$')
    #axs[1].set_ylabel(r'$y$')
    #axs[1].set_ylim(-1, 1)
    #axs[0,1].legend()
    axs[1,1].legend()
    import pdb;pdb.set_trace()

    fig.tight_layout()
    fig.savefig('doc/figures/localAdvDiff/nn_pred_2D_extra_std.png')
    plt.close(fig)

def plot_mu_k_vs_ks_nn(xgrid, k, k_nn):
    plt.figure(figsize=(15,8))
    plt.plot(xgrid, k, color='green', linewidth=4, linestyle=':', label=r'KL target: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.plot(xgrid, k_nn, color='blue',label=r'Learned: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"diffusion, $k$")
    plt.legend()
    plt.title(r'$k$')
    plt.savefig('doc/figures/nn_mu_k_nn.png')
    plt.close(fig)

def plot_std_k_vs_ks_nn(xgrid, k, k_nn):
    import pdb;pdb.set_trace()
    plt.figure(figsize=(15,8))
    plt.plot(xgrid, np.std(k, axis=0), color='green', linewidth=4, linestyle=':', label=r'KL target: $\sigma_k$')
    plt.plot(xgrid, np.std(k_nn, axis=0), color='blue',label=r'Learned: $\hat\sigma_k$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"std deviation, $\sigma_k$")
    plt.legend()
    plt.title(r'$\sigma_k$')
    plt.savefig('doc/figures/nn_std_k_nn.png')
    plt.close(fig)

def plot_k_eigvecs_vs_ks_nn(xgrid, k_eigvecs, k_eigvecs_nn):

    plt.figure(figsize=(15,8))
    kl_dim = k_eigvecs.shape[1]
    colors = cm.get_cmap('tab20')
    for p in range(kl_dim):
        plt.plot(xgrid, k_eigvecs[:,p],color=colors.colors[2*p], linewidth=4, linestyle=':', label=r'Target: $\phi_{' + str(p) + '}$')
        plt.plot(xgrid, k_eigvecs_nn[:,p], color=colors.colors[2*p+1], label=r'Learned: $\hat\phi_{' + str(p) + '}$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"eigenvector, $\phi$")
    plt.legend()
    plt.title(r'eigenvector')
    plt.savefig('doc/figures/k_eigvecs_nn.png')
    plt.close(fig)

def plot_kl_k_vs_ks_nn(xgrid, k, k_samples_nn):
    plt.figure(figsize=(15,8))
    n_samples = k_samples_nn.shape[0]
    k_samples_mean = k_samples_nn.mean(axis=0)
    k_samples_std = k_samples_nn.std(axis=0)
    plt.plot(xgrid, k_samples_mean, color='blue')#, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_samples_mean + k_samples_std, k_samples_mean - k_samples_std, 
        alpha=0.3, color='blue',
        label=r'Learned: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    k_mean = k.mean(axis=0)
    k_std = k.std(axis=0)
    plt.plot(xgrid, k_mean, color='green')#, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_mean + k_std, k_mean - k_std, 
        alpha=0.3, color='green',
        label=r'KL target: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"diffusion, $k$")
    plt.legend()
    plt.title(r'$k$')
    plt.savefig('doc/figures/nn_kl_k_samples.png')
    plt.close(fig)

def plot_k_vs_ks_nn(xgrid, k, k_samples_nn):
    plt.figure(figsize=(15,8))
    n_samples = k_samples_nn.shape[0]
    k_samples_mean = k_samples_nn.mean(axis=0)
    k_samples_std = k_samples_nn.std(axis=0)
    plt.plot(xgrid, k_samples_mean, color='blue')#, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_samples_mean + k_samples_std, k_samples_mean - k_samples_std, 
        alpha=0.3, color='blue',
        label=r'Learned: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    k_mean = k.mean(axis=0)
    k_std = k.std(axis=0)
    plt.plot(xgrid, k_mean, color='green')#, label=r'$mathbf{E}_{\xi}[k]$')
    plt.fill_between(xgrid,#range(u_mean.shape[0]),
        k_mean + k_std, k_mean - k_std, 
        alpha=0.3, color='green',
        label=r'PCE target: $\mathbf{E}_\xi[k] \pm \sigma_n$')
    plt.xlabel(r"location, $x$")
    plt.ylabel(r"diffusion, $k$")
    plt.legend()
    plt.title(r'$k$')
    plt.savefig('doc/figures/nn_k_samples.png')
    plt.close(fig)

"""
Plot Helmholtz equation
"""
def plot_helmholtz_v(xgrid, ygrid, v):
    xx, yy = np.meshgrid(xgrid, np.flip(ygrid), indexing='xy')#-1:1:.01, -1:1:.01]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), dpi=100)
    cax0 = axs[0].contourf(xx, yy, v.real)
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title(r'Forcing, $Re(v)$') 
    axs[0].set_xlabel(r'x in $m$')
    axs[0].set_ylabel(r'y in $m$')
    axs[0].set_aspect('equal', adjustable='box')
    cax1 = axs[1].contourf(xx, yy, v.imag)
    fig.colorbar(cax1, ax=axs[1])
    axs[1].set_title(r'Forcing, $Im(v)$') 
    axs[1].set_xlabel(r'x in $m$')
    axs[1].set_ylabel(r'y in $m$')
    axs[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    plt.savefig('doc/figures/helmholtz/forcing.png')
    plt.close(fig)

def plot_helmholtz_sol(xgrid, ygrid, sol):
    xx, yy = np.meshgrid(xgrid, np.flip(ygrid), indexing='xy')#-1:1:.01, -1:1:.01]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), dpi=100)
    cax0 = axs[0].contourf(xx, yy, sol.real, levels=50)
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title(r'Solution, $Re(u)$') 
    axs[0].set_xlabel(r'x in $m$')
    axs[0].set_ylabel(r'y in $m$')
    axs[0].set_aspect('equal', adjustable='box')
    cax1 = axs[1].contourf(xx, yy, sol.imag, levels=50)
    fig.colorbar(cax1, ax=axs[1])
    axs[1].set_title(r'Solution, $Im(u)$') 
    axs[1].set_xlabel(r'x in $m$')
    axs[1].set_ylabel(r'y in $m$')
    axs[1].set_aspect('equal', adjustable='box')
    fig.tight_layout()
    plt.savefig('doc/figures/helmholtz/sol.png')
    plt.close(fig)

def plot_wave_number(xgrid, ygrid, k2):
    xx, yy = np.meshgrid(xgrid, np.flip(ygrid), indexing='xy')#-1:1:.01, -1:1:.01]
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(7,3), dpi=100)
    cax0 = axs[0].contourf(xx, yy, k2.real)
    fig.colorbar(cax0, ax=axs[0])
    axs[0].set_title(r'Wave number, $Re(k)$') 
    axs[0].set_xlabel(r'x in $m$')
    axs[0].set_ylabel(r'y in $m$')
    axs[0].set_aspect('equal', adjustable='box')
    cax1 = axs[1].contourf(xx, yy, k2.imag)
    fig.colorbar(cax1, ax=axs[1])
    axs[1].set_title(r'Wave number, $Im(k)$') 
    axs[1].set_xlabel(r'x in $m$')
    axs[1].set_ylabel(r'y in $m$')
    axs[1].set_aspect('equal', adjustable='box')

    fig.tight_layout()
    plt.savefig('doc/figures/helmholtz/wave_number.png')
    plt.close(fig)

"""
Plot local advection diffusion
"""
def plot_w(zgrid, w):
    """
    Plots the vertical velocity w
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(zgrid, w)
    axs.set_xlabel(r'height, $z$')
    axs.set_ylabel(r'vertical velocity, $w$')
    fig.tight_layout()
    fig.savefig('doc/figures/vel_w.png')
    plt.close(fig)

def plot_k_diff(zgrid, k):
    """
    Plots the diffusivity k
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(zgrid, k)
    axs.set_xlabel(r'height, $z$')
    axs.set_ylabel(r'diffusivity, $k$')
    fig.tight_layout()
    fig.savefig('doc/figures/diff_k.png')
    plt.close(fig)

"""
Plot Lorenz96
"""
def plot_lorenz96(tgrid, solx, soly, solz, K):
    """
    Plots the three solutions of lorenz96 
    """
    # Plot ensemble
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300)
    for k in range(K):
        if k>3:
            break
        axs[0].plot(tgrid, solx[:,k], label=r'$X_'+str(k)+'$')
        axs[1].plot(tgrid, soly[:,k,0], label=r'$Y_{'+str(k)+',0}$')
        axs[2].plot(tgrid, solz[:,k,0,0], label=r'$Z_{'+str(k)+',0,0}$')

    axs[0].set_ylabel(r'$X_t$')
    axs[0].set_xticks([])
    axs[1].set_ylabel(r'$Y_t$')
    axs[1].set_xticks([])
    axs[2].set_ylabel(r'$Z_t$')
    axs[2].set_xlabel(r'time, $t$')
    for i in range(3):    
        axs[i].legend(loc=1, prop={'size':6})

    axs[0].set_title("Lorenz '96")
    fig.tight_layout()
    fig.savefig('doc/figures/lorenz96/ens.png')
    plt.close(fig)

    # Plot single sample
    fig, axs = plt.subplots(nrows=3, ncols=1, dpi=300)

    k = 0
    axs[0].plot(tgrid, solx[:,k], label=r'$X_'+str(k)+'$')
    axs[1].plot(tgrid, soly[:,k,k], label=r'$Y_{'+str(k)+','+str(k)+'}$')
    axs[2].plot(tgrid, solz[:,k,k,k], label=r'$Z_{'+str(k)+','+str(k)+','+str(k)+'}$')

    axs[0].set_ylabel(r'$X_t$')
    axs[0].set_xticks([])
    axs[1].set_ylabel(r'$Y_t$')
    axs[1].set_xticks([])
    axs[2].set_ylabel(r'$Z_t$')
    axs[2].set_xlabel(r'time, $t$')
    for i in range(3):    
        axs[i].legend(loc=1, prop={'size':6})

    axs[0].set_title("Lorenz '96")
    fig.tight_layout()
    fig.savefig('doc/figures/lorenz96/lorenz96.png')
    plt.close(fig)

def plot_lorenz96_y(tgrid, soly, JK):
    """
    Plots middle resolution variable of lorenz96
    """
    # Plot ensemble
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    import pdb;pdb.set_trace()
    for jk in range(JK):
        if jk>3:
            break
        axs.plot(tgrid, solx[:,jk], label=r'$Y_'+str(jk)+'$')

    axs.set_ylabel(r'$Y_t$')
    axs.set_xlabel(r'time, $t$')
    axs.legend(loc=1, prop={'size':6})

    axs.set_title("Lorenz '96")
    fig.tight_layout()
    fig.savefig('doc/figures/lorenz96/y.png')
    plt.close(fig)


def plot_lorenz96_avg(tgrid, solx0):
    """
    solx0 np.array(n_samples, tgrid)
    """
    n_samples = solx0.shape[0]
    n_tgrid = tgrid.shape[0]

    # Plot
    fig, axs = plt.subplots(nrows=1, ncols=1, dpi=300)
    axs.plot(tgrid, np.mean(solx0, axis=0))
    axs.fill_between(tgrid, np.mean(solx0, axis=0) - np.std(solx0, axis=0), 
                        np.mean(solx0, axis=0) + np.std(solx0, axis=0), alpha=0.3)
    axs.set_xlabel(r'time, $t$')
    axs.set_ylabel(r'$X_0$')
    #axs.set_ylim(-1, 1)
    axs.legend()
        
    fig.tight_layout()
    fig.savefig('doc/figures/lorenz96/avg.png')
    plt.close(fig)

def plot_nn_lorenz96_sol_xnull(tgrid, soly_true, soly_pred):
    """
    solx_pred np.array(tgrid, n_vars)
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16,8), dpi=300)
    colors = cm.get_cmap('tab20')
    for jk in range(soly_pred.shape[1]):
        if jk > 3:
            break
        plt.plot(tgrid, soly_true[:,jk], color=colors.colors[3*jk], linewidth=1, label=r'$Y_{0,' + str(jk) + '}$')
        plt.plot(tgrid, soly_pred[:,jk], color=colors.colors[3*jk+1], linewidth=2, linestyle=':')#, label=r'Learned: $\hat\phi_{' + str(k) + '}$')

    axs.set_ylim((soly_true.min()-1, soly_true.max()+1))
    axs.set_xlabel(r'time, $t$')
    axs.set_ylabel(r'$Y_{j,k}$')
    axs.legend()

    fig.tight_layout()
    fig.savefig('doc/figures/lorenz96/sol_nn_xnull.png')
    plt.close(fig)

def plot_nn_lorenz96_solx(tgrid, solx_true, solx_pred, title='solx'):
    """
    solx_pred np.array(tgrid, n_vars)
    """
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16,8), dpi=100)
    colors = cm.get_cmap('tab20')
    for k in range(solx_pred.shape[1]):
        if k > 3:
            break
        plt.plot(tgrid[::2], solx_true[::2,k], color=colors.colors[2*k], linewidth=1, label=rf'${k}$')
        plt.plot(tgrid[::2], solx_pred[::2,k], color=colors.colors[2*k+1], linewidth=2, linestyle=':')#, label=r'Learned: $\hat\phi_{' + str(k) + '}$')

    axs.set_ylim((solx_true.min()-1, solx_true.max()+1))
    axs.set_xlabel(r'time, $t$')
    axs.set_ylabel(r'$NN_m$')
    axs.legend()

    fig.tight_layout()
    fig.savefig(f'doc/figures/lorenz96/{title}_nn.png')
    plt.close(fig)

def plot_nn_lorenz96_err(tgrid, sol_true, sol_pred, title='errx'):
    """
    sol_true np.array(n_samples, tgrid, n_vars)
    sol_pred np.array(n_samples, tgrid, n_vars)
    """
    skip = 2
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(16,8), dpi=100)
    colors = cm.get_cmap('tab20')
    for k in range(sol_pred.shape[1]):
        if k > 3:
            break
        abs_err = np.abs(sol_true[:,:,k]-sol_pred[:,:,k])
        mae = np.mean(abs_err, axis=0)
        std = np.std(abs_err, axis=0)
        axs.plot(tgrid[::skip], mae[::skip], color=colors.colors[2*k], linewidth=1, label=rf'${k}$')
        axs.fill_between(tgrid[::skip], mae[::skip] - std[::skip], 
                    mae[::skip] + std[::skip], alpha=0.3)

    axs.set_ylim((mae.min()-1, mae.max()+1))
    axs.set_xlabel(r'time, $t$')
    axs.set_ylabel(r'$MAE$')
    axs.legend()

    fig.tight_layout()
    fig.savefig(f'doc/figures/lorenz96/{title}_nn.png')
    plt.close(fig)

"""
Others
"""

def plot_2d_sol(xgrid, tgrid, T_sol):
    n_xgrid = xgrid.shape[0]
    n_tgrid = tgrid.shape[0]

    every_x = int(float(n_xgrid)/4.)
    every_t = int(float(n_tgrid)/4.)

    # Plot
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
    axs[0].plot(xgrid, T_sol[1,:], label=r'$t_{'+str(1)+'}$')
    axs[0].plot(xgrid, T_sol[2,:], label=r'$t_{'+str(2)+'}$')
    for i in range(T_sol.shape[0]):
        if i%int(n_tgrid / 5)==0: #  every_t==0:
            axs[0].plot(xgrid, T_sol[i,:], label=r'$t_{'+str(i)+'}$')
    axs[0].set_xlabel(r'location, $x$')
    axs[0].set_ylabel(r'temperature, $T$')
    #axs[0].set_ylim(-1, 1)
    axs[0].legend()

    for i in range(T_sol.shape[1]):
        if i%int(n_xgrid/5)==0:
            axs[1].plot(tgrid, T_sol[:,i], label=r'$x_{'+str(i)+'}$')
    axs[1].set_xlabel(r'time, $t$')
    axs[1].set_ylabel(r'temperature, $T$')
    #axs[1].set_ylim(-1, 1)
    axs[1].legend()
        
    fig.tight_layout()
    fig.savefig('doc/figures/localAdvDiff/sample_T_sol.png')
    plt.close(fig)

def plot_2d_sol_avg(xgrid, tgrid, T_sol):
    """
    T_sol np.array(n_samples, tgrid, xgrid, 1)
    """
    n_samples = T_sol.shape[0]
    n_xgrid = xgrid.shape[0]
    n_tgrid = tgrid.shape[0]

    # Plot
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=300)
    for t in [0, int(n_tgrid/2), n_tgrid-1]:
        axs[0].plot(xgrid, np.mean(T_sol[:,t,:,0], axis=0), label=r'$t_{'+str(t)+'}$')
        axs[0].fill_between(xgrid, np.mean(T_sol[:,t,:,0], axis=0) - np.std(T_sol[:,t,:,0], axis=0), 
                    np.mean(T_sol[:,t,:,0], axis=0) + np.std(T_sol[:, t,:,0], axis=0), alpha=0.3)
    axs[0].set_xlabel(r'location, $x$')
    axs[0].set_ylabel(r'temperature, $T$')
    #axs[0].set_ylim(-1, 1)
    axs[0].legend()

    for x in [0, int(n_xgrid/4.), int(n_xgrid/2), int(3.*n_xgrid/4.), n_xgrid-1]:
        axs[1].plot(tgrid, np.mean(T_sol[:,:,x,0], axis=0), label=r'$x_{'+str(x)+'}$')
        axs[1].fill_between(tgrid, np.mean(T_sol[:,:,x,0], axis=0) - np.std(T_sol[:,:,x,0], axis=0), 
                    np.mean(T_sol[:,:,x,0], axis=0) + np.std(T_sol[:,:,x,0], axis=0), alpha=0.3)
    axs[1].set_xlabel(r'time, $t$')
    axs[1].set_ylabel(r'temperature, $T$')
    #axs[1].set_ylim(-1, 1)
    axs[1].legend()
        
    fig.tight_layout()
    fig.savefig('doc/figures/localAdvDiff/avg_T_sol.png')
    plt.close(fig)