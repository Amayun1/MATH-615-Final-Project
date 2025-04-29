from Final_Project import *
from matplotlib import pyplot as plt
from tabulate import tabulate

# test case is the particle in a box on [0,1].

def state(x ,n, x0, x1):
    L = x1-x0
    return np.sqrt(2/L) * np.sin((n+1) * np.pi * x / L)

def V(x):
    return 0

def PIB_energy(n, x0, x1, mass):
    hbar = 1
    L = x1 - x0
    return (n**2 * np.pi**2 * hbar**2) / (2 * mass * L**2)

# parameters of test
mass = 1
m = 100
states = 5
tf = 10
x0 = 0
x1 = 1
# parameters of test

h = (x1-x0)/((m+1)**2)

x, U, energy = Diffusion_CSFE(mass, V, x0, x1, m, tf=tf, states=states, show_iters=False)
t_energy = [PIB_energy(j, x0, x1, mass) for j in range(1, states+1)]

# sort energy levels from lowest to highest as the algo may not extract the excited states in order.
sort_index = np.argsort(energy)

# initialize table to store the errors for the wavefunction and the energies
table = []

# graph all outputs of the test, calculate errors.
for j in range(0, states):
    plt.plot(x, U[sort_index[j]], label='FE Numeric Solution')
    plt.plot([], [], alpha=0, label='Theoretical Energy = {:.3E} J'.format(t_energy[j]))
    plt.plot([], [], alpha=0, label='Calculated Energy = {:.3E} J'.format(energy[sort_index[j]]))

    # calculate errors for both possible orthonotmal states (+-1), plot the one with lower error
    error_1 = np.sqrt(h) * linalg.norm(U[sort_index[j]] - state(x, j, x0, x1), ord=2)
    error_2 = np.sqrt(h) * linalg.norm(U[sort_index[j]] + state(x, j, x0, x1), ord=2)
    if error_1 < error_2:
        plt.plot(x, state(x, j, x0, x1), label='Analytic Solution')
        plt.plot([], [], alpha=0, label='2-Norm Error = {:.3E}'.format(error_1))
    else:
        plt.plot(x, -state(x, j, x0, x1), label='Analytic Solution')
        plt.plot([], [], alpha=0, label='2-Norm Error = {:.3E}'.format(error_2))

    # construct next row for the output table
    row = [j, min([error_1, error_2]), energy[sort_index[j]], t_energy[j], np.abs(energy[sort_index[j]]-t_energy[j])/t_energy[j]]
    table.append(row)

    plt.legend()
    plt.title('Particle in a Box, State = {}'.format(j))
    plt.show()

    # clear plot
    plt.figure()

# Print data table header
headers = ["State", "2-Norm Error", "Calculated Energy", "Theoretical Energy", "Energy Relative Error"]
print(tabulate(table, headers, floatfmt=(None, '.5e', '.3f', '.3f', '.3%')))