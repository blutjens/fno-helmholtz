import os 
import argparse
import numpy as np 
import pprint # Printing dictionaries

import wandb

os.environ['DDEBACKEND'] = 'pytorch'

import pce_pinns.utils.plotting as plotting
import pce_pinns.utils.logging as logging

from pce_pinns.solver.helmholtz import HelmholtzEq# , reshape_lorenz96_to_nn
from pce_pinns.solver.sampler import sample_diffeq, sample_model
from pce_pinns.neural_nets.fno import interpolate_param_fno

def reshape_helmholtz_to_nn(sol, u_args, xgrid, ygrid):
    """
    Reshapes helmholtz.solve() output and xgrid into flexible NN in-/output

    Args:
        sol np.array((n_samples, n_x1grid, n_x2grid, 1))
        u_args np.array((n_samples, n_x1grid, n_x2grid, 1)
        xgrid np.array((n_x1grid))
        ygrid np.array((n_x2grid))
    Returns:
        u_target np.array(n_samples, n_tgrid, n_x1grid, n_x2grid, 1): Solution, e.g., u(r)
        grid np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, dim_grid)): dim_grid = 2
        y_args np.array((n_samples, n_tgrid, n_x1grid, n_x2grid, 1)): Function arguments, e.g., k^2(r)
    """
    n_samples = sol.shape[0]

    # Reshape grid
    xx,yy = np.meshgrid(xgrid, np.flip(ygrid), indexing='xy')
    tt = np.zeros(xx.shape) # Add time zeros to match general nn input shape
    grid = np.dstack((tt, xx, yy))[np.newaxis,...] # Stack empty
    grid = np.repeat(grid[np.newaxis,...], repeats=n_samples, axis=0)

    # Reshape solution
    u_target = sol[:,np.newaxis,...]
    y_args = u_args[:,np.newaxis,...]

    return u_target, grid, y_args

def sync_nested_wandb_config(sweep_config, config_defaults, verbose=True):
    """
    Syncs default config dictionary and wandb sweep config. This is a workaround 
    because wandb doesn't natively support nested dictionaries. 
    Args:
        sweep_config run.config: Non-nested sweep config from wandb
        config_defaults dict(dict()) : Nested default config
    Updates:
        wandb.config(dict()): Nested wandb config that contains dictionaries and 
            matches returned config_defaults
    Returns: 
        wandb.config(dict()): Nested wandb config that contains dictionaries and 
            matches returned config_defaults
        config_defaults: Nested default config, changed by sweep parameters 
    """
    # Pull default config into wandb.config
    sweep_config.setdefaults(config_defaults)

    # Transfer sweep config into nested parameters
    for sweep_param_key in sweep_config.keys():
        if sweep_param_key == '_wandb': # Skip placeholder key
            continue
        for parent_key in config_defaults:
            if type(config_defaults[parent_key])==dict:
                if sweep_param_key in config_defaults[parent_key]:
                    sweep_config[parent_key][sweep_param_key] = sweep_config[sweep_param_key]
    # Update config_defaults with new parameters from sweep
    for key in config_defaults.keys():
        config_defaults[key] = sweep_config[key]

    return sweep_config, config_defaults

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='helmholtz')
    # Differential equation
    parser.add_argument('--est_lorenz', default='Ynext', type=str,
            help='Name of solution variable that shall be estimated by neural net, e.g., '\
            '"X" for X_{0:K}(t) = NN(t)), '\
            '"Y" for Y_{j,k}(t) = NN(t), '\
            '"Z",'\
            '"Xres" for X_{0:K}(0:T) = NN(X_{0:K}(0:T;h=0)); not implemented'
            '"Ynext" for Y_{0:J,0:K}(t+1) = NN(X_{0:K}(t), Y_{0:J,0:K}(t)), '\
            '"superparam" for Y_{0:J,k}(t) = NN(X_k(t-1),Y_{0:J,k}(t-1))'\
            '"Y(X_null)" for Y_{0:J,k}(t+1) = NN(X_k(t), Y_{0:J,k}(t), X_k(t+1;h=0))' \
            '"full" for hc/b sum_{j=0}^J Y_{j,k}(t+1) = NN(X_k(t;h=0), hc/b sum_{j=0}^J Y_{j,k}(t))'\
            '')
    parser.add_argument('--n_samples', default=50, type=int,
            help='Number of samples in forward and inverse solution.')
    # NN approximation
    parser.add_argument('--est_param_nn', default='u', type=str,
            help='Name of parameter that shall be estimated by neural net, e.g., "\
            "pce_coefs", "k", "k_eigvecs", "k_true", "u", "u_true"')
    parser.add_argument('--path_load_simdata', default=None, type=str,
            help='Path to logged simulation data, e.g., data/pce.')
    # General
    parser.add_argument('--parallel', action="store_true",
            help='Enable parallel processing.')
    parser.add_argument('--seed', default=1, type=int,
            help='Random seed')
    parser.add_argument('--debug', action="store_true",
            help='Query low runtime debug run, e.g., reduce dimensionality, number of samples, etc.')
    parser.add_argument('--no_plot', action="store_true",
            help='Deactivate plotting of results')
    parser.add_argument('--no_wandb', action="store_true",
            help='Disables wandb')
    parser.add_argument('--eval_model_digest', default=None, type=str, 
            help='Hex code of model that should be evaluated')
    args = parser.parse_args()
    if args.parallel:
        assert args.no_plot==True, "Run model without parallel flag if plots are desired"

    np.random.seed(args.seed)
    """
    [x] update pytorch: conda install pytorch torchvision torchaudio cpuonly -c pytorch # Update to torch>=1.9
    [x] fuse config files
    [x] disable wandb for faster testing
    [x] Load model for evaluation
    [x] evaluate visual u_fno - u_fd
    [x] evaluate inf loss.
    [XX] Evaluate runtime vs. grid size.
    [XX] evaluate runtime over grid > 256
    [XX] Plot runtimes of FD on same plot as FNO 
    [] 1 -- Write approach
    [] 2 -- Write intro
    [] 3 - Create runtime vs. accuracy plot with values that I have
    [x] Plot accuracy over n. Should I compare to avg downscaled or original prediction?  
    []  add FD solver accuracies on coarsened grid into acc over n plot.
    []  add FD solver into runtime vs. acc plot.
    [] plot predictions on smaller grid without any interpolation, either as matrix or some no interpolation option in contourf
    [] Create sliced accuracy plot.
    [] extend FNO to *complex* values
    [] Normalize x and y -- transform = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    [] Synthesize results in notebook.
    [] Run with varying inputs
    [] ! write math
    [] run with jit
    """
    
    # Init config and logger
    os.environ['WANDB_SILENT'] = 'true'
    #os.environ['WANDB_MODE'] = 'dryrun'
    config_defaults = {
        'de': {
            # Differential Equation
            'xmax': 1., # Square box of [0,1]m
            'n_xgrid': 256,
            'f': 21.3e6,
            'load_rel_perm_path': "data/helmholtz/MRI_DATA.mat", # None
            'random_ic': True,
            'n_samples': args.n_samples,
            'seed': args.seed,
            'solver_name': 'sparse',
        },
        'data_loader':{
            'dir_store_fno_simdata': 'data/fno/helmholtz/data',        
            # Eval
            'test_mode': 'batch',
            'normalize': False,
            'val_size': 0.2,
            'test_size': 0.0,
            # FNO
            'statics': [],
            'chunk_size': 3,#X.shape[0],
            'batch_size': 2, # Has to be > 1 
            'n_hist': 0,
            'n_fut': 0,
            'seed': args.seed,
        },
        'model': {
            # FNO
            'depth': 1, #1
            'n_channels': 3, #3
            'n_modes': [20,20],#][5,5],
            # 'n_layers': 3,
            # 'n_units': 512,
        },
        'optimizer':{ # optimizer
            'lr': 0.1, # 0.001,
            'step_size': 10,
            'gamma': .1,
        },
        'n_epochs': 40, # 40
    }
    # Init wandb hyperparameter sweep
    sweep_config = {
        'name': 'fno-helmholtz-sweep',
        'method': 'grid',
        'metric': {
            'name': 'minloss_val',
            'goal': 'minimize'
        },
        'parameters': {
            'depth': {'value': 1},
            'n_channels': {'value': 3}
        }
    }

    def _main(config_defaults, sweep_config, args):
        """
        Main functions that's called by wandb hyperparameter sweep
        """
        if not args.no_wandb:
            config, config_defaults = sync_nested_wandb_config(
                sweep_config, config_defaults, verbose=True)
        else:
            config = config_defaults

        # Define grid
        xgrid = np.linspace(0., config['de']['xmax'], config['de']['n_xgrid'] + 1) 
        ygrid = np.linspace(0., config['de']['xmax'], config['de']['n_xgrid'] + 1)
        dx = 1./(config['de']['n_xgrid']+1.) # step size

        # Init differential equation
        helmholtzEq = HelmholtzEq(xgrid, ygrid, dx=dx, f=config['de']['f'], 
            load_rel_perm_path=config['de']['load_rel_perm_path'],
            plot=(args.no_plot==False), seed=config['de']['seed'])
        
        # Init surrogate model of differential equation
        model = sample_diffeq
        model_args = {'diffeq': helmholtzEq,
            'xgrid': None,'kl_dim': None, 'pce_dim': None}

        # Estimate various parameters with a neural network
        if args.est_param_nn == 'u':
            import torch
            torch.manual_seed(config['de']['seed'])

            # Generate dataset
            print('Generating '+str(config['de']['n_samples']) + ' target model samples')
            logs = sample_model(model=model, model_args=model_args, 
                n_samples=config['de']['n_samples'], run_parallel=args.parallel,
                path_load_simdata=args.path_load_simdata, path_store_simdata='data/helmholtz')
            u_args, u_target, rand_insts = logging.convert_log_dict_to_np_helmholtz(logs)

            # Arrange input dimensions to fit NN inputs
            u_target, grid, y_args = reshape_helmholtz_to_nn(sol=u_target, u_args=u_args,
                xgrid=xgrid, ygrid=ygrid)
            # Select grid input dims:
            grid_in_dims = () # (-1,) for all grid as input

            print('todo: DELETE: stop discarding imaginary part of input and target')
            u_target = u_target.real 
            y_args = y_args.real 
            # Use NN
            if args.est_param_nn=='u': # or args.est_param_nn=='u_true':
                u, u_pred = interpolate_param_fno(grid=grid, y=u_target,
                    y_args=y_args,
                    n_epochs=config['n_epochs'], n_layers=config['model']['depth'], 
                    n_units=config['model']['n_modes'], lr=config['optimizer']['lr'],
                    target_name=args.est_param_nn,
                    alpha_indices=None, rand_insts=rand_insts,
                    n_test_samples=args.n_samples,
                    diffeq=helmholtzEq, loss_name='mseloss', normalize=config['data_loader']['normalize'],
                    u_params=None, grid_in_dims=grid_in_dims,
                    test_mode=config['data_loader']['test_mode'],
                    plot=(args.no_plot==False), debug=args.debug, config=config_defaults,
                    eval_model_digest=args.eval_model_digest)

            else:
                raise NotImplementedError
            # plotting.plot_fno_helmholtz(xgrid, ygrid, u_pred, u)

            if not args.no_wandb:
                wandb.log({'runtime/tstep PDE.solve': helmholtzEq.cumulative_runtime/float(config['de']['n_samples'])})

    if not args.no_wandb:
        def run_sweep():
            with wandb.init() as run:
                print('Current sweep config:')
                pprint.pprint(run.config)
                _main(config_defaults, sweep_config=run.config, args=args)
        sweep_id = wandb.sweep(sweep_config, project="fno-helmholtz")# , entity='blutjens')
        wandb.agent(sweep_id, run_sweep, count=1000)
    else:
        wandb.init(mode='disabled')
        _main(config_defaults, sweep_config=None, args=args)
