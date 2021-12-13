from typing import Any, Dict, Optional, Union
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


from pce_pinns.neural_nets.fno_dataloader import make_big_lazy_cat
from pce_pinns.neural_nets import fno2d

def weighted_avg(x):
    y, n = zip(*x)
    return np.sum(np.multiply(y, n)) / np.sum(n)


def loss_batch(
    model,
    loss_func,
    x,
    y,
    opt=None,
    *,
    model_args=None,
    n_done=None,
    cb=None,
    del_x=False,
    del_y=False
):
    nx = len(x)

    if model_args is None:
        model_args = []

    loss = loss_func(model(x, *model_args), y)
    if del_x:
        del x

    if del_y:
        del y

    if cb:
        cb(loss / nx, n_done)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), nx


def fit_epoch(
    model,
    loss_func,
    opt,
    *,
    train_dl,
    test_dl,
    model_args=None,
    train_cb=None,
    test_cb=None,
    device=None,
    quiet=False,
    progress_bar_prefix=""
):
    desc_train = progress_bar_prefix + "  Training"
    desc_test = progress_bar_prefix + "Validation"
    
    model.train()
    loss_train = weighted_avg(
        loss_batch(
            model,
            loss_func,
            x.to(device) if device else x,
            y.to(device) if device else y,
            opt,
            model_args=model_args,
            n_done=i,
            cb=train_cb,
            del_x=device and x.device != device,
            del_y=device and y.device != device,
        )
        for i, (x, y) in enumerate(tqdm(train_dl, desc=desc_train, disable=quiet))
    )

    model.eval()
    with torch.no_grad():
        loss_test = weighted_avg(
            loss_batch(
                model,
                loss_func,
                x.to(device) if device else x,
                y.to(device) if device else y,
                model_args=model_args,
                n_done=i,
                cb=test_cb,
            )
            for i, (x, y) in enumerate(tqdm(test_dl, desc=desc_test, disable=quiet))
        )

    return loss_train, loss_test


def make_model(
    config: Dict[str, Any], *, device: Optional[Union[str, torch.device]] = None
) -> Any:
    """Creates and initializes FNO, stacked with a linear layer.

    The linear layer has a single output neuron.

    :param config: Configuration used to initialize the FNO.
    :param device: The ``torch.device`` on which tensors should be stored.
    :return: The (JIT) compiled model.
    """
    model = nn.Sequential(
        fno2d.FNO(device=device, **config),
        nn.Linear(config["n_channels"], 1),#, device=device),
    )
    #return torch.jit.script(model)
    return model

def train_model(
    *,
    x_train,
    x_test,
    y_train,
    y_test,
    loss_mask,
    epoch_cb,
    dl_config=None,
    model_config=None,
    opt_config=None,
    meta=None,
    seed=None,
    device=None,
):
    if seed is not None:
        torch.manual_seed(seed)

    if not (type(loss_mask) is torch.Tensor):
        loss_mask = torch.tensor(loss_mask)

    loss_mask = loss_mask.to(device).unsqueeze(0)
    if len(loss_mask.shape) == 3:
        loss_mask = loss_mask.unsqueeze(-1)

    if (
        x_train.shape[:-1] != y_train.shape[:-1]
        or x_test.shape[:-1] != y_test.shape[:-1]
        or x_train.shape[1:] != x_test.shape[1:]
        or y_train.shape[1:] != y_test.shape[1:]
        or y_train.shape[1:-1] != loss_mask.shape[1:-1]
    ):
        raise ValueError("Shapes of input data are inconsistent.")

    if not (y_train.shape[-1] == y_test.shape[-1] == loss_mask.shape[-1] == 1):
        raise ValueError(
            "Last dimension of targets and loss mask has to be of size one."
        )

    if dl_config is None:
        dl_config = dict()

    if model_config is None:
        model_config = dict()

    if opt_config is None:
        opt_config = dict()

    if meta is None:
        meta = dict()

    dl_config = {
        "statics": (0,), #TODO: Number of static input variables ??
        "chunk_size": 4096,
        "n_hist": 0,
        "n_fut": 0,
        "n_repeat": 0,
        "batch_size": 64,
        "seed": seed,
        **dl_config, # Default values above are overwritten by **dl_config
    }
    n_statics = len(dl_config["statics"])
    n_hist = dl_config["n_hist"]

    n_features = x_train.shape[-1]
    n_features += (n_features - n_statics) * n_hist

    model_config = {
        "depth": 1,
        "n_features": n_features,
        "n_channels": 7,
        "n_modes": (5, 5),
        **model_config,
    }

    opt_config = {
        "lr": 0.1,
        "step_size": 20,
        "gamma": 0.1,
        **opt_config,
    }
    train_dl = make_big_lazy_cat(x_train, y_train, device="cpu", 
        statics=dl_config['statics'], chunk_size=dl_config['chunk_size'],
        n_hist=dl_config['n_hist'], n_fut=dl_config['n_fut'],
        n_repeat=dl_config['n_repeat'], batch_size=dl_config['batch_size'],
        seed=dl_config['seed'])
    test_dl = make_big_lazy_cat(x_test, y_test, device="cpu", 
        statics=dl_config['statics'], chunk_size=dl_config['chunk_size'],
        n_hist=dl_config['n_hist'], n_fut=dl_config['n_fut'],
        n_repeat=dl_config['n_repeat'], batch_size=dl_config['batch_size'],
        seed=dl_config['seed'])
    model = make_model(model_config, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=opt_config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=opt_config["step_size"], gamma=opt_config["gamma"]
    )

    def loss_fct(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(((x - y) * loss_mask) ** 2)

    loss_train = []
    loss_test = []

    n_epoch = 0
    stop = False
    while not stop:
        n_epoch += 1

        loss = fit_epoch(
            model,
            loss_fct,
            opt,
            train_dl=train_dl,
            test_dl=test_dl,
            device=device,
            progress_bar_prefix=f"[Epoch #{n_epoch}] ",
        )
        loss_train.append(loss[0])
        loss_test.append(loss[1])
        stop = epoch_cb(loss_train, loss_test)

        scheduler.step()

    return model.to("cpu"), {
        "data_loader": dl_config,
        "model": model_config,
        "optimizer": opt_config,
        "meta": meta,
        "n_epochs": n_epoch,
        "loss": {
            "training": loss_train,
            "validation": loss_test,
        },
    }

