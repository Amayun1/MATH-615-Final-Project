import numpy as np
from scipy import sparse
from scipy import linalg
from scipy import integrate
from random import random
from tqdm import tqdm
import time



# todo
"""
    Write a code to display the errors in a nice table. During the tests.
"""

def normalize(U, h):
    """
    Args:
        U (iterable): array or vector to be normalized.
        h (float): step size for FD scheme.
    Return:
        U_norm (np.ndarray): normalized vector using finite difference 2-norm.
    """
    return U / (np.sqrt(h) * linalg.norm(U, ord=2))

def total_Energy(U, V, x, h, mass):
    """
    Calculates the total energy of the particle on the mesh x by discretizing the Hamiltonian
        Hu = -(hbar)^2/(2m)u_xx + V(x)u
    Centered finite differences are used.

    Args:
        U (np.ndarray): Wavefunction values on the mesh.
        V (function): Potential energy function.
        x (np.ndaarray): Spacial discretization mesh.
        h (float): Mesh spacing.
        mass (float): Mass of particle in kg.
    Return:
        Energy: Calculated energy for the particle in joules.
    """

    # define hbar in SI base units.
    hbar = 1

    # calculate numeric second derivative of U
    D_squared = []
    for j in range(0, len(U)):
        if j == 0:
            val = (-2*U[j] + U[j+1]) / (h ** 2)
        elif j == len(x)-1:
            val = (U[j-1] - 2 * U[j]) / (h ** 2)
        else:
            val = (U[j - 1] - 2 * U[j] + U[j + 1]) / (h ** 2)
        D_squared.append(val)

    # calculate potential energy function along the mesh
    v = [V(j) for j in x]

    # calculate Hamiltonian applied to the state U
    ham = [-((hbar**2) / (2 * mass)) * j for j in D_squared] + np.multiply(v, U)
    # calculate energy by the integral of psi* H psi using Simpson's rule.
    energy = integrate.simps(np.multiply(U, ham), dx=h)

    return energy


def Diffusion_CSFE(mass, V, x0, x1, m, tf = 5, states=3, show_iters=False):
    """
        Solves solves the time independent schrodinger equation with x in [x0,x1],
            -(hbar^2/(2m)) u_xx + V(x)u = Eu,
            u(t,x0) = 0, u(t,x1) = 0

        Centered finite differences are used in space with m+1 interior points, step size h = 1/(m+1).
        Forward Euler is used in time with temporal step size k = 2/(D*h^2).

        Source:
          R. LeVeque, Finite Difference Methods for Ordinary and Partial Differential Equations, SIAM Press 2007

        Args:
            mass (float): mass of particle for energy state calculations
            V (function): source term V on [x0,x1]
            x0 (float): lower spacial bound, in meters.
            x1 (float): upper spacial bound, in meters.
            m (int): number of interior grid points in the spacial direction
            tf (float): How long to run simulation. Default 5 seconds
            states (int): number of states to calculate. Default 3, minimum 1.
            show_iters (bool): If true, track the current iteration for each state.

        Return:
            x: list of (interior) grid points in space
            U (np.ndarray): Array of approximated function values on x for each state, row-wise.
            energy (list): list containing the energy of each energy state.
        """

    # spacial step h, and interior point mesh
    h = np.abs(x1 - x0) / (m + 1)
    x = np.linspace(x0 + h, x1 - h, m, endpoint=True)

    # set temporal spacing.
    k = (1/2)*h**2
    print("Time step k = {}".format(k))
    iters = int(round(tf/k))
    print("Total time iterations = {}".format(iters))

    # construct matrix diagonals for the method.
    main_diag = []
    upper_diag = [0]
    lower_diag = []

    for j in x:
        # define some constants used multiple times.
        alpha = 1 / (1 + (k/2) * V(j))
        beta = (1 - (k/2) * V(j)) / (1 + (k/2) * V(j))

        main_diag.append(2 * k * alpha / (h**2) + beta)
        upper_diag.append(k * alpha / (h**2))
        lower_diag.append(k * alpha / (h**2))

    # remove last element of upper_diag and shift element of lower_diag to get the correct alignment.
    del upper_diag[-1]
    del lower_diag[0]
    lower_diag.append(0)

    # construct matrix.
    diagonals = [lower_diag, main_diag, upper_diag]
    B = sparse.spdiags(diagonals, [-1, 0, 1], m, m, format='csr')

    # initialize U0 as a random initial condition, then normalize
    initial = [random() for j in x]
    U = normalize(initial, h)

    # initialize list to store the energy for each state.
    energy = []

    print('Calculating Ground State.')
    time.sleep(0.1)
    # main loop for ground state
    for j in tqdm(range(0, iters)):
        if show_iters:
            print('State = {}, iteration = {}'.format(1,j))

        # calculate next iteration and normalize
        U = B @ U
        U = normalize(U, h)
    output = np.array([U])
    energy.append(total_Energy(U, V, x, h, mass))

    # loop for each excited state to calculate
    for s in range(1, states):
        print("Calculating Excited State {}.".format(s))
        time.sleep(0.1)
        # get initial condition and remove previously calculated states
        for j in range(0, s):
            c = h * np.inner(initial, output[j]) * output[j]
            U = U - c * output[j]

        # calculate next state
        for n in tqdm(range(0, iters)):
            if show_iters:
                print('State = {}, iteration = {}'.format(s+1, n))

            # calculate next iteration, remove previous states and normalize
            U = B @ U
            for j in range(0, s):
                c = h * np.inner(U, output[j])
                U = U - c * output[j]
            U = normalize(U, h)

        # add next state to output, calculate the every of the state and append
        output = np.vstack((output, U))
        energy.append(total_Energy(U, V, x, h, mass))

    return x, output, energy