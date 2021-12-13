"""
Differential equation solver for 2D Helmholtz Equation
Author: Björn Lütjens (lutjens@mit.edu)
"""
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

import scipy.io
from scipy.stats import multivariate_normal
import scipy.sparse as sp
import scipy.sparse.linalg as la
from scipy.fftpack import dst, dstn, idst, idstn

from pce_pinns.utils.plotting import plot_helmholtz_v, plot_helmholtz_sol, plot_wave_number
from pce_pinns.solver.diffeq import DiffEq

class HelmholtzEq(DiffEq):
    def __init__(self, xgrid, ygrid, dx, f=21.3e6, load_rel_perm_path=None, 
        plot=False, plot_params=False, seed=0):
        """
        Sets up 2D inhomogeneous Helmholtz equation as boundary value problem. 
        The equation models static waves and is based on 
        18.336 pset2.
        
        \nabla^2 u(r) + k^2(r) u(r) = v(r)

        Args:
            xgrid np.array(Nx+1): uniform 2D square grid with Nx+1 points and Nx-1 interior poits
            ygrid np.array(Nx+1):
            dx float: Step size in space
            f float: Frequency in Hz
            load_rel_perm_path string: Path to lead relative permittivity field
            e_r np.array(Nx+1, Ny+1): relative permittivity field 
        """
        self.xgrid = xgrid
        self.ygrid = ygrid
        self.Nx = self.xgrid.shape[0]-1 # Number of grid points
        self.dx = dx

        self.f = f
        self.w = 2.*np.pi*self.f # Angular frequency in radians per second
        self.c = 299792458. # Speed of light in m/s
        # Set (non-)constant wave number parameter
        self.k2 = np.power(self.w/self.c,2) # Wave number squared
        self.init_wave_number(load_rel_perm_path, plot=plot_params) # Flattens 2D-k2 field into sparse diagonal matrix
        self.v, self.v_flat, self.v_fn, self.v_pos = self.init_forcing(xgrid, ygrid, plot=plot_params)

        self.A = None
        self.A = self.init_inhom_helmholtz_2D_FD_mat(self.Nx, self.dx) # Finite difference matrix

        self.seed = seed
        self.rng = default_rng(self.seed)
        self.cumulative_runtime = 0. # Cumulative runtime for solve(). Can be divided by number of fn calls
        self.plot = plot

        # self.random_ic = random_ic
        # self.sample_random_params()


    def sample_random_ic(self, init=False):
        """ Sample random initial condition
        TODO
        Returns:
            rand_insts np.array(stoch_dim): Random instances
            ic np.array(?): Random initial conditions
        """
        return -np.ones(1), -np.ones(1) 

    def init_forcing(self, xgrid, ygrid, xc=0.6, yc=0.7, std=0.01, plot=False):
        """
        Initializes forcing. Here, the electric excitation is a Gaussian impulse 
        centered at xc,yc with std deviation, std.
        
        Args:
            xgrid np.array(Nx+1)
            ygrid np.array(Nx+1)
            xc float : Center of electric excitation in x-direction, in m
            yc float : Center of electric excitation in y-direction, in m
            std float : Std deviation/~radius of electric excitation in m 
        Returns:
            v np.array((Nx+1,Ny+1))): Electric excitation at radius r=[x,y]
            v_flat np.array((Nx+1)*(Ny+1)): Flattened electric excitation
            v_fn function(np.array((Nx+1,Ny+1))): Function that returns electric excitation at radius r=[x,y]
            v_pos np.array((Nx+1, Ny+1)): Position arguments to query electric excitation
        """
        rv = multivariate_normal([xc, yc], [[std**2, 0], [0, std**2]])

        xx, yy = np.meshgrid(xgrid,ygrid)#-1:1:.01, -1:1:.01]
        v_pos = np.dstack((xx, yy))
        v_fn = rv.pdf 
        v = v_fn(v_pos)
        v_flat = v.flatten(order='C')

        if plot:
            plot_helmholtz_v(self.xgrid, self.ygrid, v)

        return v, v_flat, v_fn, v_pos

    def read_relative_permittivity(self, path):
        """
        Read in relative permittivity MRI data
        Returns:
            e_r np.array((Nx+1, Nx+1)): relative permittivity; e_r is 'ij' indexed
        """
        mri_data = scipy.io.loadmat(path) # .mat
        x = mri_data['x'][0]
        y = mri_data['y'][0]
        e_r = mri_data['e_r']

        assert e_r.shape[0]==(self.Nx+1) and e_r.shape[1]==(self.Nx+1), (
            f'Shapes dont match -- e_r:{e_r.shape}, Nx+1:{self.Nx+1}')
        assert np.all(x==self.xgrid)
        assert np.all(y==self.ygrid)

        return e_r

    def init_wave_number(self, load_rel_perm_path=None, plot=False):
        """
        Sets wave number with given relative permittivity

        Args:
            load_rel_perm_path string: Path to lead relative permittivity field
        Sets:
            k2 sp.csr_matrix((Nx+1)**2): diagonal matrix
        """
        if load_rel_perm_path is not None:
            e_r = self.read_relative_permittivity(load_rel_perm_path)
            self.k2 = self.k2 * e_r.flatten() # row major flatten
        else:
            self.k2 = self.k2 * np.ones((self.Nx+1)**2)

        if plot:
            plot_wave_number(self.xgrid, self.ygrid, self.k2.reshape((self.Nx+1, self.Nx+1)))

        self.k2 = sp.diags(self.k2) # Convert into diagonal matrix

        return 1

    def init_inhom_helmholtz_2D_FD_mat(self, Nx, dx):
        """
        Create sparse 2D finite difference matrix for inhomogeneous Helmholtz equation  
        with homogeneous Dirichlet boundary conditions
        Args:
            Nx int: Number of grid points
            dx float: Step size
        Returns:
            A scipy.csr_matrix: 2D FD matrix
        """
        if not self.A:
            Kx = sp.diags([1.*np.ones(Nx), -2.*np.ones(Nx+1), 1*np.ones(Nx)], offsets=[-1,0, 1]) # homogenous Dirichlet BCs
            I = sp.eye(Nx+1)
            self.A = sp.kron(Kx,I) + sp.kron(I,Kx) # Convert 1D into 2D FD matrix
            if np.isscalar(self.k2):
                self.A = self.A + dx**2 * self.k2 * sp.eye((Nx+1)**2) # Add inhomogenous term
            else:
                self.A = self.A + dx**2 * self.k2
            self.A = dx**(-2) * self.A # scale by step size to ensure stability
        return self.A

    def step_rng(self):
        """ Iterate the random number generator 

        Iterates the random number generator, s.t., parallel processes have different output
        """
        self.rng.uniform(0, 1, size=(1))
        return 0

    def sample_random_params(self):
        """ Sample random parameters
    
        Sets all stochastic parameters to a new sample

        Returns:
            rand_insts np.array(stoch_dim): Random instances
            rand_param np.array((Nx+1, Ny+1)): Random parameters
        """
        # TODO: make stochastic
        return -np.ones(1), self.k2.diagonal().reshape((self.Nx+1, self.Nx+1))

    def residual(self, x, y, z, t):
        """
        Computes autodifferentiable squared residual of proposed solution.
        Args:
        """
        raise NotImplementedError
        return 0

    def step(self, u):
        """
        Has no step
        """
        return u

    def solve_sparse_scipy(self, A, v_flat):
        sol = la.spsolve(A, v_flat)
        return sol

    def solve_preconditioned_FD(self, Nx, dx, k2,  v):
        """
        Compute DST matrix

        Args:
            Nx int: Number of grid points is Nx+1
            dx float: Step size
            k2 : wave number
            v np.array(Nx+1,Ny+1): Forcing v        
        Returns:
            u np.array((Nx+1)*(Ny+1)): sol
        """
        N = Nx+1
        K = N

        # Transform RHS into Fourier domain
        v_dst = dstn(v, type=1, axes=[0,1], norm='ortho').flatten()

        # Build diagonal FD matrix
        Kx = sp.diags([1.*np.ones(Nx), -2.*np.ones(Nx+1), 1*np.ones(Nx)], offsets=[-1,0,1]) # Laplacian
        D_r3 = dstn(Kx.toarray(), type=1, axes=[1,0], norm='ortho') # Transform into Fourier
        D_r3 = sp.diags(np.diag(D_r3)) # Transform to sparse diagonal
        D_rc3 = sp.kron(D_r3,sp.eye(N)) + sp.kron(sp.eye(N),D_r3) + k2 * sp.eye(N**2) * dx**2 # Add non-constant paramer

        # Compute solution given RHS
        D_rc_inv = sp.diags(np.power(D_rc3.diagonal(),-1)) # Invert diagonal FD matrix; this is significantly faster than la.inv(D_rc3)
        D_rc_inv = dx**2 * D_rc_inv
        u_dst = D_rc_inv.dot(v_dst) # Get solution in Fourier domain

        # Transform solution into Euclidean domain
        u = idstn(u_dst.reshape(N, N), axes=[1,0], type=1, norm='ortho')

        return u

    def solve(self, solver_name='sparse'):
        """
        Solves the equation

        Args:
        Returns:
            sol (np.array(n_xgrid+1, n_ygrid+1, 1),
        """
        if solver_name is None or solver_name=='sparse':
            sol = self.solve_sparse_scipy(A=self.A, v_flat=self.v_flat)
        elif solver_name=='preconditioned_FD':
            sol = self.solve_preconditioned_FD(Nx=self.Nx, dx=self.dx, k2=self.k2, v=self.v)
        else:
            raise NotImplementedError(f'Solver {solver_name} not implemented for Helmholtz')
        sol = sol.reshape(self.Nx+1,self.Nx+1) # Nx+1, Ny+1

        if self.plot:
            plot_helmholtz_sol(self.xgrid, self.ygrid, sol)

        return sol

if __name__ == "__main__":
    """
    Create plot of test helmholtz equation. See plotting.plot_helmholtz for directory
    """
    parser = argparse.ArgumentParser(description='Helmholtz')
    parser.add_argument('--solver_name', default='sparse', type=str,
        help='Solver name e.g., "sparse", "preconditioned_FD".')
    args = parser.parse_args()  

    # Define grid
    Nx = 256
    xgrid = np.linspace(0., 1., Nx+1)
    ygrid = np.linspace(0., 1., Nx+1)
    dx = 1./(Nx+1.) # step size

    # Init differential equation
    helmholtzEq = HelmholtzEq(xgrid, ygrid, dx=dx, f=21.3e6, 
        load_rel_perm_path="data/helmholtz/MRI_DATA.mat", plot=True, seed=0)

    sol = helmholtzEq.solve(solver_name=args.solver_name)

