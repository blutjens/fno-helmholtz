# 18.336 Fast learning-based solutions of the Helmholtz equation with Fourier Neural Operators
## 18.336 Final class project

This repository implements Fourier neural operator to solve the 2D inhomogeneous Helmholtz equation.

# Install
```
git clone git@github.com:blutjens/fno-helmholtz.git
cd fno-helmholtz
conda env create -f environment.yml # tested on Ubuntu 18.04
conda activate fno-helmholtz
conda install pytorch torchvision torchaudio cpuonly -c pytorch # Update to torch>=1.9
pip install -e .
pip install ray # tested with 1.4.1
wandb login # Login to ML logging tool
```
## Train new model
```
python main_helmholtz.py
```
## Recreate results with trained model
```
python main_helmholtz.py --seed 1 --n_samples 128 --path_load_simdata "data/helmholtz" --no_wandb --eval_model_digest a43eee0178                                                                     
```
## Run on server
```
export WANDB_MODE=offline
```

# References 
```
@article{Li_2021,
  title = {Fourier Neural Operator for Parametric Partial Differential Equations},
  author = {Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
  year = 2021,
  journal = {ICML},
}
```
