import os
import time
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import wandb

from sklearn.preprocessing import MinMaxScaler

from pce_pinns.neural_nets.fno_train import train_fno
from pce_pinns.neural_nets import fno2d_gym
from pce_pinns.neural_nets.fno_dataloader import make_big_lazy_cat
from pce_pinns.neural_nets.losses import calc_infty_loss, calc_mse_loss
import pce_pinns.utils.shaping as shaping
import pce_pinns.utils.plotting as plotting

def get_paths(n_samples, dir_store_fno_simdata, n_chunks=1):
    """
    n_chunks int: Number of data chunks , e.g., int(x.shape[0]/n_samples)
    """
    d_proc = Path(dir_store_fno_simdata + f'/n{n_samples:d}_t{int(n_chunks):d}')
    paths = dict()
    paths['f_xtrain'] = d_proc / "xtrain.npy"
    paths['f_xtest'] = d_proc / "xtest.npy"
    paths['f_ytrain'] = d_proc / "ytrain.npy"
    paths['f_ytest'] = d_proc / "ytest.npy"
    paths['f_lossmsk'] = d_proc / "notlandbool.npy"

    return paths, d_proc

def save_fno_data(x, y, n_samples, val_size, dir_store_fno_simdata):
    """        
    Args:
        x np.array((n_samples*n_tgrid, n_x1grid, n_x2grid, dim_in)): 2D input array, see shaping.shape_2D_to_2D_data for info
        y np.array((n_samples*n_tgrid, n_x1grid, n_x2grid, dim_y)): 2D output array, see shaping.shape_2D_to_2D_data for info
                            dim_in = grid_in_dims + dim_y_args
    Saves:
        f_xtrain np.array((n*train_size, n_x1grid, n_x2grid, dim_in): 
        f_xtest np.array((n*val_size, n_x1grid, n_x2grid, dim_in)):
        f_ytrain np.array((n*train_size, n_x1grid, n_x2grid, dim_y))): y
        f_ytest np.array((n*val_size, n_x1grid, n_x2grid, dim_y)): 
        f_lossmsk np.array((n_x1grid, n_x2grid, dim_y)): Loss mask; zero for masked values
    """

    lossmsk = np.ones(y[0].shape, dtype=bool)
    
    # Train test split
    n_train = int((1-val_size)*x.shape[0])
    n_val = int(val_size*x.shape[0])

    # Define paths
    paths, d_proc = get_paths(n_samples, dir_store_fno_simdata, n_chunks=int(x.shape[0]/n_samples))

    # Create paths
    if not os.path.exists(d_proc): 
        os.makedirs(d_proc)

    # Save data
    np.save(paths['f_xtrain'] , x[:n_train])
    np.save(paths['f_xtest'] , x[-n_val:])
    np.save(paths['f_ytrain'] , y[:n_train])
    np.save(paths['f_ytest'] , y[-n_val:])
    np.save(paths['f_lossmsk'] , lossmsk)

    return paths

def get_specifier(config):
    """
    Returns:
        specifier string: Terse specifier of config
    """
    specifier = (f"t{config['data_loader']['n_hist']}{config['data_loader']['n_fut']}"
        f"_d{config['model']['depth']}c{config['model']['n_channels']}"
        f"m{config['model']['n_modes'][0]}{config['model']['n_modes'][1]}"
        )
    return specifier

def msr_runtime(fn, args, M=100, verbose=False):
    t = time.time()
    for m in range(M):
        _ = fn(args)
    avg_t = (time.time()-t) / float(M)
    if verbose:
        print('Avg runtime: ', avg_t)
    return avg_t

def get_trained_model_cfg(digest, dir_out):
    """
    Returns trained model cfg given hex digest code
    """
    with (dir_out / "{}_cfg.json".format(digest)).open() as jf:
        cfg = json.load( jf )
    return cfg

def interpolate_param_fno(grid, y, y_args=None,
        n_epochs=20, batch_size=None, n_layers=2, n_units=128,
        lr=0.001,
        target_name='u', 
        alpha_indices=None, n_test_samples=1, 
        diffeq=None, loss_name=None, normalize=False,
        u_params=None, grid_in_dims=(-1,),
        test_mode='batch',
        val_size=0.0, test_size=0.2,
        plot=False, rand_insts=None, debug=False,
        config=None, eval_model_digest=None):
    """
    TODO: after this function works -- make it look as close as possible to interpolate_param_nn
    Args:
        See neural_net.interpolate_param_nn
        eval_model_digest string: Hex digest code to trained model. If not None, training is skipped
    """
    device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir_out = Path('data/fno/helmholtz/config')
    dir_plot = Path('doc/figures/helmholtz/fno')

    if eval_model_digest is None:
        ## Reshape and save data
        x, y, dim_out, rand_insts, batch_size = shaping.shape_2D_to_2D_nn_input(grid=grid, 
            y=y, y_args=y_args, rand_insts=rand_insts, target_name=target_name, 
            batch_size=config['data_loader']['batch_size'], alpha_indices=alpha_indices,
            grid_in_dims=grid_in_dims)
        config['data_loader']['batch_size'] = batch_size

        if config['data_loader']['dir_store_fno_simdata'] is not None:
            paths = save_fno_data(x, y, config['de']['n_samples'], 
                config['data_loader']['val_size'], config['data_loader']['dir_store_fno_simdata'])

        # Normalize data
        if normalize:
            raise NotImplementedError('normalization not implemented')
        else:
            scalerx = None
            scalery = None

        # Save FNO config file --> TODO: fuse with main_
        specifier = get_specifier(config)
        print('specifier: ', specifier)
        
        paths['f_cfg'] = dir_out / "{}.json".format(specifier)
        with open(paths['f_cfg'], "w") as f:
            f.write(json.dumps(config))

        # Call the train function
        fno_args = " ".join([
            f"--config {paths['f_cfg']}",
            f"--trainx {paths['f_xtrain']}",
            f"--trainy {paths['f_ytrain']}",
            f"--testx {paths['f_xtest']}",
            f"--testy {paths['f_ytest']}",
            f"--lossmsk {paths['f_lossmsk']}",
            f"--outdir {str(dir_out)}",
            f"--device {device}",
            f"--epochs {str(config['n_epochs'])}",
            "--verbose",
            "--overwrite",
        ])
        print("$ python train_fno.py", fno_args)
        eval_model_digest = train_fno(fno_args.split())

    ###################
    ## Check results---
    ###################
    print(eval_model_digest)
    cfg = get_trained_model_cfg(eval_model_digest, dir_out)
    ls = cfg["loss"]["validation"]
    minls = min(ls)
    aminls = ls.index(minls)
    print("{0:.2g} @ep{1} Min Loss Val".format(minls, aminls)) 
    wandb.log({"minloss_val": minls})

    ls = cfg["loss"]["training"]
    minls = min(ls)
    aminls = ls.index(minls)
    print("{0:.2g} @ep{1} Min Loss Train".format(minls, aminls)) 
    wandb.log({"minloss_train": minls})

    if plot: plotting.plot_fno_training_curve(cfg['loss'], dir_plot)

    ###################
    ## Predict & Evaluate
    ###################
    cfg = get_trained_model_cfg(eval_model_digest, dir_out)
    try: 
        n_chunks = int(x.shape[0]/n_samples)
    except:
        n_chunks = 1
    paths, d_proc = get_paths(config['de']['n_samples'], 
        config['data_loader']['dir_store_fno_simdata'], 
        n_chunks=n_chunks)

    model_load = fno2d_gym.make_model(cfg["model"])
    model_load.load_state_dict(torch.load(str(dir_out / "{}_modstat.pt".format(eval_model_digest)))) 

    x_val = np.load(str(paths['f_xtest']), mmap_mode = 'r')
    y_val = np.load(str(paths['f_ytest']), mmap_mode = 'r')
    dl_cfg = cfg["data_loader"]
    dl_cfg["chunk_size"] = dl_cfg['batch_size']
    val_loader = make_big_lazy_cat(x_val, y_val, device="cpu", 
        statics=dl_cfg['statics'], chunk_size=dl_cfg['chunk_size'],
        n_hist=dl_cfg['n_hist'], n_fut=dl_cfg['n_fut'],
        n_repeat=dl_cfg['n_repeat'], batch_size=dl_cfg['batch_size'],
        seed=dl_cfg['seed'])

    specifier = get_specifier(cfg)
    measure_runtime=False
    infty_losses = []  
    mse_losses = []
    runtimes = []
    N_maxs = [256, 128, 64, 32, 16, 8, 4]
    # N_maxs = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4]
    m_runtime_avgs = 100*np.ones(len(N_maxs)) # Repeats time measurement m-times.
    m_runtime_avgs[0] = 1# 1
    m_runtime_avgs[1] = 5# 1
    m_runtime_avgs[2] = 10# 10
    m_runtime_avgs[3] = 20# 
    
    example = 0
    for i, (x, y) in enumerate(val_loader):
        for n_i, N_max in enumerate(N_maxs):
            xgrid = grid[example,0,0,:,1]
            ygrid = np.flip(grid[example,0,:,0,2])

            if N_max <= 256:
                xgrid = shaping.downsample1d_log2(torch.from_numpy(xgrid[np.newaxis,:].copy()), tgt_size=N_max)
                ygrid = shaping.downsample1d_log2(torch.from_numpy(ygrid[np.newaxis,:].copy()), tgt_size=N_max)
                x = shaping.downsample2d_log2(x, tgt_size=N_max)
                y = shaping.downsample2d_log2(y, tgt_size=N_max)

            # Measure runtime
            if measure_runtime and i==0:
                print(n_i, N_max)
                if N_max > 256:
                    factors = int(np.log2(N_max) - np.log2(256))
                    x_up_fake = x.clone()
                    for _ in range(factors):
                        x_up_fake = torch.cat((x_up_fake,x_up_fake), dim=(1))[:,:-1,:,:]
                        x_up_fake = torch.cat((x_up_fake,x_up_fake), dim=(2))[:,:,:-1,:]
                    x_tst = x_up_fake
                else:
                    x_tst = x
                runtimes.append(msr_runtime(model_load, x_tst, M=int(m_runtime_avgs[n_i])))
                print(f'N={N_max:5d}: Runtime {runtimes[-1]:.6f}s')

            else:
                # Predict
                y_pred = model_load(x)
                data = torch.cat((y_pred.cpu().detach()[example], y.cpu().detach()[example]), -1)
                y_pred_np = data[...,0].numpy()
                y_true_np = data[...,1].numpy()

                # Evaluate infinity error
                infty_loss = calc_infty_loss(y_pred, y, grid_dims=(1,2))
                infty_loss = torch.mean(infty_loss,axis=0).detach().cpu().numpy()[0] # Average across batch
                infty_losses.append(infty_loss)
                mse_loss = calc_mse_loss(y_pred, y, grid_dims=(1,2))
                mse_loss = torch.mean(mse_loss, axis=0).detach().cpu().numpy()[0]
                mse_losses.append(mse_loss)
                print(f'N={N_max:5d}: Err_Infty {infty_loss: .6f}; MSE {mse_loss: .6f}')

                # Plot difference
                xgrid = grid[example,0,0,:(N_max+1),1]
                ygrid = np.flip(grid[example,0,:(N_max+1),0,2])
                try:
                    plotting.plot_fno_helmholtz_diff(xgrid, ygrid, y_pred_np, y_true_np, fname=f'sol_diff_Nmax{N_max}')
                except:
                    print(f'Failed diff plot for N:{N_max}')
                # Plot solution 
                try:
                    plotting.plot_fno_helmholtz(xgrid, ygrid, y_pred_np, y_true_np, fname=f'sol_pred_Nmax{N_max}')
                except:
                    print(f'Failed sol plot for N:{N_max}')

                # Plot solution
                #plotting.plot2Dfeats_fno(data, 
                #    fname=str(dir_plot / "{}ypred.png".format(specifier)), 
                #    featnames=["pred", f"true"])
        break       

    # Plot runtime over domain size
    if measure_runtime:
        plotting.plot_fno_runtimes_vs_N(np.asarray(N_maxs), np.asarray(runtimes))

    # Plot accuracy over domain size
    plotting.plot_fno_accuracy_vs_N(np.asarray(N_maxs), np.asarray(mse_losses), np.asarray(infty_losses))

    # Plot runtime over accuracy
    return y_true_np[example], y_pred_np[example]