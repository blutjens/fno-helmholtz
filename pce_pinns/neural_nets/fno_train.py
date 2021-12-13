import argparse
import json
import sys
from pathlib import Path
from hashlib import md5

import numpy as np
import torch 
import torch.nn as nn

from pce_pinns.neural_nets import fno2d_gym as gym
import pce_pinns.utils.plotting as plotting

def dump(model, config, *, path, overwrite=False):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"{path} is not a valid directory.")

    config_json = json.dumps(config)
    digest = md5(config_json.encode("utf-8")).hexdigest()[:10]

    config_file = path / (digest + "_cfg.json")
    model_file = path / (digest + "_modstat.pt")

    if not overwrite and (config_file.is_file() or model_file.is_file()):
        raise ValueError(f"{config_file} or {model_file} already exists.")

    with open(config_file, "w") as f:
        f.write(config_json)

    torch.save(model.state_dict(), model_file)

    return digest


def read_config(path, digest, suffix="_cfg.json"):
    if not digest.endswith(suffix):
        digest += suffix

    path = Path(path) / digest
    if not path.is_file():
        raise ValueError(f"File {path} does not exist.")

    with open(path, "r") as f:
        return json.loads("".join(f.readlines()))


def read_model(path, digest, suffix_model="_modstat.pt", suffix_cfg="_cfg.json"):
    config = read_config(path, digest, suffix=suffix_cfg)
    model_config = config["model"] if "model" in config else dict()

    model_file = Path(path) / (digest + suffix_model)

    if not model_file.is_file():
        raise ValueError(f"File {path} does not exist.")

    return torch.load(model_file), model_config


def iter_configs(path, suffix="_cfg.json"):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Directory {path} does not exist.")

    for f in path.iterdir():
        if f.is_file() and str(f).endswith(suffix):
            digest = f.name[: -len(suffix)]
            yield digest, read_config(path, digest, suffix=suffix)


def json_config(arg):
    p = Path(arg)
    if not p.is_file():
        raise ValueError(f"File {p} does not exist")

    with open(p, "r") as f:
        try:
            raw = "".join(f.readlines())
            return json.loads(raw)
        except json.decoder.JSONDecodeError:
            raise ValueError("Could not decode configuration as JSON")


def npy_file(arg, mmap_read=False):
    p = Path(arg)
    if not p.is_file():
        raise ValueError(f"File {p} does not exist")

    try:
        return np.load(p, mmap_mode="r") if mmap_read else np.load(p)
    except ValueError:
        raise ValueError(f"Could not parse {p}")


def output_dir(parser, arg):
    p = Path(arg)
    if p.is_dir():
        return p

    parser.error(f"Directory {p} does not exist")


def create_parser():
    parser = argparse.ArgumentParser(description="Train FNO")
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        type=json_config,
        help="path to configuration file",
    )

    parser.add_argument(
        "--trainx",
        dest="trainx",
        required=True,
        type=lambda arg: npy_file(arg, mmap_read=True),
        help="path to training features",
    )

    parser.add_argument(
        "--trainy",
        dest="trainy",
        required=True,
        type=lambda arg: npy_file(arg, mmap_read=True),
        help="path to training targets",
    )

    parser.add_argument(
        "--testx",
        dest="testx",
        required=True,
        type=lambda arg: npy_file(arg, mmap_read=True),
        help="path to validation features",
    )

    parser.add_argument(
        "--testy",
        dest="testy",
        required=True,
        type=lambda arg: npy_file(arg, mmap_read=True),
        help="path to validation targets",
    )

    parser.add_argument(
        "--lossmsk",
        dest="lossmsk",
        required=True,
        type=npy_file,
        help="path to loss mask",
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        default=0,
        type=int,
        help="overwrite number of epochs to train from configuration file",
    )

    parser.add_argument(
        "--outdir",
        dest="outdir",
        required=True,
        type=lambda x: output_dir(parser, x),
        help="output directory",
    )

    parser.add_argument("--device", dest="device", type=str, help="device to submit to")

    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="be verbose"
    )

    parser.add_argument(
        "--pretend",
        dest="pretend",
        action="store_true",
        help="do not actually submit but print arguments",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="overwrite stored config and model",
    )

    return parser


def get_config(args, key):
    return args.config[key] if key in args.config else None


def train_fno(args):
    args = create_parser().parse_args(args)

    dl_cfg = get_config(args, "data_loader")
    model_cfg = get_config(args, "model")
    opt_cfg = get_config(args, "optimizer")
    meta = get_config(args, "meta")

    n_epochs = 0
    if "n_epochs" in args.config:
        n_epochs = args.config["n_epochs"]
    if args.epochs > 0:
        n_epochs = args.epochs

    if n_epochs == 0:
        print(
            "Error: Couldn't find number of epochs in configuration nor in arguments."
        )
        exit(1)

    verbose = args.verbose or args.pretend

    if verbose:
        print("Output directory:", args.outdir)
        print("          Device:", args.device if args.device else "<default>")
        print("        # Epochs:", n_epochs)
        print("\nShapes")
        print(" *   training x:", args.trainx.shape, f"(dtype: {args.trainx.dtype})")
        print(" *   training y:", args.trainy.shape, f"(dtype: {args.trainy.dtype})")
        print(" * validation x:", args.testx.shape, f"(dtype: {args.testx.dtype})")
        print(" * validation y:", args.testy.shape, f"(dtype: {args.testy.dtype})")
        print(" *    loss mask:", args.lossmsk.shape, f"(dtype: {args.lossmsk.dtype})")
        print("\nConfiguration")
        print(" * data loader:", dl_cfg if dl_cfg else "<default>")
        print(" *       model:", model_cfg if model_cfg else "<default>")
        print(" *   optimizer:", opt_cfg if opt_cfg else "<default>")
        print(" *        meta:", meta if meta else "<none>")

    if not args.pretend:
        model, cfg = gym.train_model(
            x_train=args.trainx,
            y_train=args.trainy,
            x_test=args.testx,
            y_test=args.testy,
            loss_mask=args.lossmsk,
#             epoch_cb=lambda loss_train, loss_test: len(loss_train) >= args.epochs,
            epoch_cb=lambda loss_train, loss_test: len(loss_train) >= n_epochs,
            dl_config=dl_cfg,
            model_config=model_cfg,
            opt_config=opt_cfg,
            meta=meta,
            seed=dl_cfg['seed'],
            device=args.device,
        )

        digest = dump(model, cfg, path=args.outdir, overwrite=args.overwrite)
    else:
        digest = "??? (dry run)"

    if verbose:
        print("\nUse digestID to load model: ", digest)

    return digest

#if __name__ == "__main__":
#    digest = train_fno(sys.argv[1:])
#    print(digest)

from sklearn.preprocessing import MinMaxScaler
def normalize_data_lorenz96(f_xtrain, f_ytrain, f_xtest, 
    f_ytest, f_lossmsk, dir_load_simdata, 
    test_mode='sequential', debug=False):

    # Load unnormalized data
    x_train = np.load(str(f_xtrain), mmap_mode = 'r')
    y_train = np.load(str(f_ytrain), mmap_mode = 'r')
    x_val = np.load(str(f_xtest), mmap_mode = 'r')
    y_val = np.load(str(f_ytest), mmap_mode = 'r')
    lossmsk = np.load(str(f_lossmsk), mmap_mode = 'r')
    x = np.concatenate((x_train,x_val), axis=0)
    y = np.concatenate((y_train,y_val), axis=0)

    norm_range_y = (0,1) # -2,5
    norm_range_x = (0,1) # -5,5
    if test_mode=='sequential':
        assert np.all(norm_range_y == norm_range_x), "In sequential"\
            "test model x and y normalization range has to be equal."
    # Normalizes feature-wise 
    scalery = MinMaxScaler(feature_range=norm_range_y)
    j = y.shape[1]
    k = y.shape[2]
    assert y.shape[-1] == 1
    # y_unnorm = y.reshape(y.shape[0],j*k*1)
    n_total_samples = y.shape[0]
    y_unnorm = y.reshape((n_total_samples*j*k,) + (y.shape[-1],))
    y_norm = scalery.fit_transform(y_unnorm)
    y = y_norm.reshape((n_total_samples,) + (j,k) + (y_norm.shape[-1],))
    # y = y_norm.reshape(y_norm.shape[0],j,k)[...,np.newaxis]
    scalerx = MinMaxScaler(feature_range=norm_range_x)
    x_dim = x.shape[-1]
    x_unnorm = x.reshape((n_total_samples*j*k,) + (x.shape[-1],))
    # x_unnorm = x.reshape(x.shape[0],j*k,x_dim)
    # x_unnorm = x_unnorm.reshape(x.shape[0],j*k*x_dim)
    x_norm = scalerx.fit_transform(x_unnorm)
    y = y_norm.reshape((n_total_samples,) + (j,k) + (y_norm.shape[-1],))
    # x_norm = x_norm.reshape(x.shape[0], j*k,x_dim)
    x = x_norm.reshape((n_total_samples,) + (j,k) + (x_dim, ))

    # Save normalized data        
    d_proc = Path(args.dir_load_simdata+f'{("_debug" if debug else ""):s}_norm')
    if not os.path.exists(d_proc): 
        os.makedirs(d_proc)
    f_xtrain = d_proc / "xtrain.npy" 
    f_xtest = d_proc / "xtest.npy" 
    f_ytrain = d_proc / "ytrain.npy" 
    f_ytest = d_proc / "ytest.npy" 
    f_lossmsk = d_proc / "notlandbool.npy" 

    y_train = y
    val_size = x_val.shape[0] / (x_train.shape[0]+x_val.shape[0])
    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    if debug:
        n_train = 3200
        n_val = 800
    np.save(f_xtrain, x[:n_train])
    np.save(f_xtest, x[-n_val:])
    np.save(f_ytrain, y[:n_train])
    np.save(f_ytest, y[-n_val:])
    np.save(f_lossmsk, lossmsk)

    return scalerx, scalery

def create_sin_cos_test_data():
    ###################
    # Create sin/cos test data
    ###################

    # Define grid and test dataset
    # X: batch_size, n, n, 2 
    # Y: batch_size, n, n, 1
    nx = 4.
    ny = 4.
    batch_size = 20
    x1 = 2.*np.pi*np.arange(nx)/nx
    x2 = 2.*np.pi*np.arange(ny)/ny
    xx, yy = np.meshgrid(x1, x2)
    X = np.concatenate((xx[:,:,np.newaxis],yy[:,:,np.newaxis]),axis=2)
    X = np.repeat(X[np.newaxis, ...], repeats=batch_size, axis=0) #
    Y = np.sin(X)
    # Y = np.prod(Y, axis=-1)[...,np.newaxis]
    # Y = np.sum(Y, axis=-1)[...,np.newaxis]
    Y = Y[...,0:1]
    # Y = np.repeat(Y, axis=-1, repeats=2) # X[-1] has to equal Y[-1]
    loss_mask = np.ones(Y[0].shape, dtype=bool)
    plotting.plot_lorenz96_fno(X[0], Y[0])

    # Save data
    d_proc = Path('data/fno/data')
    f_xtrain = d_proc / "xtrain.npy" 
    f_xtest = d_proc / "xtest.npy" 
    f_ytrain = d_proc / "ytrain.npy" 
    f_ytest = d_proc / "ytest.npy" 
    f_lossmsk = d_proc / "notlandbool.npy" 
    np.save(f_xtrain, X)
    np.save(f_xtest, X)
    np.save(f_ytrain, Y)
    np.save(f_ytest, Y)
    np.save(f_lossmsk, loss_mask)

if __name__ == "__main__":
    import os
    import sys
    import json
    from tempfile import mkstemp, TemporaryDirectory
    import matplotlib.pyplot as plt
    from pathlib import Path

    from pce_pinns.solver.lorenz96 import Lorenz96Eq, reshape_lorenz96_to_nn
    from pce_pinns.neural_nets.fno_dataloader import make_big_lazy_cat
    import pce_pinns.utils.plotting as plotting

    parser = argparse.ArgumentParser(description='test fno sine')
    parser.add_argument('--dir_load_simdata', default=None, type=str,
            help='Directory to logged simulation data, e.g., data/fno/data.')
    parser.add_argument('--dir_load_model', default=None, type=str,
            help='Directory to stored model and config, e.g., data/fno/config/.') #t00_d3c3m44 
    parser.add_argument('--debug', action="store_true",
            help='Debug mode, e.g., shortens dataset')
    args = parser.parse_args()

    # Differential equation
    np.random.seed(0)

    if not args.dir_load_simdata:
        create_sin_cos_test_data()
    else: 
        ###################
        # Load Lorenz96 data
        ###################
        is_fno = True
        config = {
            # Differential Equation
            'tmax': 2., # TODO: Generate training dataset over long time sequence to get into regime of attractor.
            'dt': 0.005,
            'n_samples': 1000,
            'J': 4,
            'K': 4,
            'test_mode': 'sequential',
            'normalize': True,
            'val_size': 0.2
        }

        d_proc = Path(args.dir_load_simdata)
        batch_size = 400
        f_xtrain = d_proc / "xtrain.npy" 
        f_xtest = d_proc / "xtest.npy" 
        f_ytrain = d_proc / "ytrain.npy" 
        f_ytest = d_proc / "ytest.npy" 
        f_lossmsk = d_proc / "notlandbool.npy" 

        normalize = True
        if normalize:
            scalerx, scalery = normalize_data_lorenz96(f_xtrain, f_ytrain, f_xtest, f_ytest, f_lossmsk, 
                args.dir_load_simdata, test_mode='sequential', debug=args.debug)
            d_proc = Path(args.dir_load_simdata+f'{("_debug" if args.debug else ""):s}_norm')
            f_xtrain = d_proc / "xtrain.npy" 
            f_xtest = d_proc / "xtest.npy" 
            f_ytrain = d_proc / "ytrain.npy" 
            f_ytest = d_proc / "ytest.npy" 
            f_lossmsk = d_proc / "notlandbool.npy" 
        else:
            scalerx = None
            scalery = None

    d_plot = Path('doc/figures/fno')
    d_out = Path('data/fno/config')
    d = {
            "data_loader": {
                "chunk_size": 32768,#X.shape[0],
                "batch_size": batch_size,
                "n_hist": 0,
                "n_fut": 0,
            },
            "model": {
                "depth": 3, #1
                "n_channels": 3, #3
                "n_modes": [4,4],#][5,5],
            },
            "optimizer": {
                "lr": 0.1,
                "step_size": 20,
                "gamma": .1,
            },
            "n_epochs": 1,
        }
    specifier = f"t{d['data_loader']['n_hist']}{d['data_loader']['n_fut']}_d{d['model']['depth']}c{d['model']['n_channels']}m{d['model']['n_modes'][0]}{d['model']['n_modes'][1]}"
    print('specifier: ', specifier)
    
    f_cfg = d_out / "{}.json".format(specifier)

    with open(f_cfg, "w") as f:
        f.write(json.dumps(d))
    DEVICE = "cpu" # "cuda:0"  

    fno_args = " ".join([
        f"--config {f_cfg}",
        f"--trainx {f_xtrain}",
        f"--trainy {f_ytrain}",
        f"--testx {f_xtest}",
        f"--testy {f_ytest}",
        f"--lossmsk {f_lossmsk}",
        f"--outdir {str(d_out)}",
        f"--device {DEVICE}",
        f"--epochs {str(d['n_epochs'])}",
        "--verbose",
    ])

    print("$ python train_fno.py", fno_args)
    digest = train_fno(fno_args.split())

    ###################
    ## Check results
    ###################
    with (d_out / "{}_cfg.json".format(digest)).open() as jf:
        cfg = json.load( jf )
    ls = cfg["loss"]["validation"]
    minls = min(ls)
    aminls = ls.index(minls)
    # print('Config: ', cfg)
    print("{0:.2g} @ep{1} Min Loss".format(minls, aminls)) 

    figl,axl = plt.subplots(1,1)
    axl.set_xlabel('epochs')
    axl.set_ylabel('mse')
    axl.set_yscale('log')
    axl.plot(cfg["loss"]["training"], 'r.', label="train")
    axl.plot(cfg["loss"]["validation"], 'k.', label="val")
    axl.legend()
    plt.savefig(d_plot/"training_curve.png")
    plt.clf() 

    ###################
    ## Predict
    ###################
    model_load = gym.make_model(cfg["model"])
    model_load.load_state_dict(torch.load(str(d_out / "{}_modstat.pt".format(digest)))) 

    x_val = np.load(str(f_xtest), mmap_mode = 'r')
    y_val = np.load(str(f_ytest), mmap_mode = 'r')
    dl_cfg = cfg["data_loader"]
    dl_cfg["chunk_size"] =  batch_size
    dl_cfg["batch_size"] =  batch_size
    val_loader = make_big_lazy_cat(x_val, y_val, device='cpu', **dl_cfg)

    example = 0
    for i, (x, y) in enumerate(val_loader):
        
        y_pred = model_load(x)
        data = torch.cat(
            (y_pred.cpu().detach()[example], y.cpu().detach()[example]),
            -1)
        plotting.plot2Dfeats_fno(data, str(d_plot / "{}ypred.png".format(specifier)), ["pred", "true"])
        break

    ###################
    # NN eval
    ###################
    import time
    # from pce_pinns.neural_nets.neural_net import eval
    from pce_pinns.neural_nets.losses import calculate_losses

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    n_t = int(config['tmax']/config['dt'])
    grid = np.ones((config['n_samples'], n_t, config["J"], config["K"], x.shape[-1]))
    grid_in_dims = () # FNO has no grid values as input
    dim_in = x.shape[-1]
    dim_y = y.shape[-1]
    dim_out = y.shape[-1]
    n_val_samples = int(config['n_samples']*config['val_size'])
    #if args.debug:
    n_val_samples_break = 2
    class Model():
        def __init__(self, f_pred):
            self.f_pred = f_pred
        def predict(self,x):
            return self.f_pred(x)
    model = Model(model_load)
    plot = True
    """
    eval_args = {'model': model, 
        'val_loader': val_loader, 
        'criterion': criterion,
        'grid': grid, 'y_args': y_args,
        'device': device, 
        'dim_in': dim_in, 'dim_out': dim_out, 
        'grid_in_dims': grid_in_dims,  
        'normalize': normalize, 'scalerx': scalerx, 'scalery': scalery, 
        'n_val_samples': n_val_samples, 'mode': config['test_mode'],
        'plot': True, 'custom_rand_inst': None}
    """
    # Init logger
    log_interval = 1
    pred_runtime = 0.
    val_epoch_loss = 0
    pce_epoch_loss = 0
    pinn_epoch_loss = 0

    x_val = np.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_in,)) # log
    # dim_y = val_loader.dataset.y_data.shape[-1] # EDITED
    y_val_pred = torch.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_y,))
    y_val_true = np.zeros((n_val_samples,) + grid.shape[1:-1] + (dim_y,))

    for batch_idx, (x_val_target, y_val_target) in enumerate(val_loader):
        x_val[batch_idx] = x_val_target.cpu().numpy().reshape(grid.shape[1:-1] + (dim_in,))
        y_val_true[batch_idx] = y_val_target.cpu().numpy().reshape(grid.shape[1:-1] + (dim_y,))
        x_val_target, y_val_target = x_val_target.to(device), y_val_target.to(device)

        if not is_fno:
            yprev = x_val_target[0:1,...,-dim_y:] # EDITED and TESTED for both x_val_target [n_tgrid, n_x1grid, n_x2grid]
        else:
            yprev = x_val_target[0:1,...,:dim_y] # WHY DO I HAVE TO FLIP concatenated X AND Y in last dim? --> check data generation
        y_val_pred[batch_idx,0] = yprev.reshape((1,) + grid.shape[2:-1] + (dim_y,)) # 1, n_x1grid, n_x2grid, ..., dim_y    
        dim_y_args_no_prev = dim_in-yprev.shape[-1]
        for t in range(1,grid.shape[1]):
            start = time.time()
            if not is_fno:
                x_t = torch.cat((x_val_target[t:t+1,...,:dim_y_args_no_prev], yprev), axis=-1) # EDITED, should work
            else:
                x_t = torch.cat((x_val_target[t:t+1,...,-dim_y_args_no_prev:], yprev), axis=-1) # EDITED, should work
            y_t = model.predict(x_t) # 1*np.prod(n_grid), dim_y

            # Use predicted value as input
            yprev = y_t

            # Log 
            pred_runtime += time.time() - start
            y_t = y_t.reshape((1,) + grid.shape[2:-1] + (dim_y,)) # 1, n_x1grid, n_x2grid, ..., dim_y    
            y_val_pred[batch_idx,t] = y_t

        # Calculate validation loss
        if dim_out == dim_y:
            y_out_pred = y_val_pred[batch_idx]
        loss_dict = calculate_losses(criterion, y_out_pred, y_val_target, y_inputs=x_val[batch_idx], batch_idx=batch_idx, custom_rand_inst=None)

        # Log 
        val_epoch_loss += loss_dict['loss'].item()
        pce_epoch_loss += loss_dict['pce_loss']
        pinn_epoch_loss += loss_dict['pinn_loss']

        if batch_idx == n_val_samples_break:
            break
    y_val_pred = y_val_pred.cpu().detach().numpy()
    if np.any(np.isnan(y_val_pred)):
        print('WARNING: predicted y_val contains nan values. Converting to 0')
        y_val_pred = np.nan_to_num(y_val_pred, copy=False, nan=0.)
    # De-normalize data
    if normalize:
        if dim_out != dim_y and mode == "sequential":
            raise NotImplementedError("NN predicting PCE coefs in time-series is not implemented.")
        x_val = x_val.reshape((n_val_samples * np.prod(grid.shape[1:-1]),) + (x_val.shape[-1],))
        x_val = scalerx.inverse_transform(x_val)
        y_val_pred = y_val_pred.reshape((n_val_samples * np.prod(grid.shape[1:-1]),)+(dim_y,))
        y_val_pred = scalery.inverse_transform(y_val_pred)
        y_val_true = y_val_true.reshape((n_val_samples * np.prod(grid.shape[1:-1]),)+(dim_y,)) # EDITED / ADDED
        y_val_true = scalery.inverse_transform(y_val_true) # EDITED / ADDED

    x_val = x_val.reshape((n_val_samples,) + grid.shape[1:-1] + (x_val.shape[-1],))
    y_val_pred = y_val_pred.reshape((n_val_samples,) + grid.shape[1:-1] + (dim_y,))
    y_val_true = y_val_true.reshape((n_val_samples,) + grid.shape[1:-1] + (dim_y,)) # EDITED / ADDED

    if plot:
        import pdb;pdb.set_trace()
        idx = 0
        tgrid = np.linspace(0,config['tmax'],n_t)
        solx_true = y_val_true[idx,:,0:3,0,0]
        solx_pred = y_val_pred[idx,:,0:3,0,0]
        plotting.plot_nn_lorenz96_solx(tgrid, solx_true, solx_pred)
         
# FNO 
# t00_d3c3m44 
# 'loss': {'training': [0.0047066809000534705], 'validation': [0.0013603367758332752]}}
# Digest: 6a0c684ba2 
# {'training': [0.0025627716150029302], 'validation': [0.006827957655768841]}}