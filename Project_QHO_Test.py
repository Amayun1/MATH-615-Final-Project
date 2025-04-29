from Final_Project import *
from scipy import special
from matplotlib import pyplot as plt
from tabulate import tabulate


# test case is the quantum harmonic oscillator on [-10,10]

# constants
hbar = 1

# parameters of test
mass = 1
omega = 10
m = 1000
states = 5
tf = 10
x0 = -10
x1 = 10
# parameters of test

h = (x1 - x0) / ((m + 1) ** 2)


# construct states of harmonics oscillator
def state(x, n):
    a = (mass * omega) / hbar
    y = np.sqrt(a) * x
    c = np.sqrt(np.sqrt(a / np.pi)) / np.sqrt((2 ** n) * special.factorial(n))
    return c * special.eval_hermite(n, y) * np.exp(- (y ** 2) / 2)


def V(x):
    return (1 / 2) * mass * (omega ** 2) * (x ** 2)


def QHO_energy(n):
    return (n + 1 / 2) * hbar * omega


x, U, energy = Diffusion_CSFE(mass, V, x0, x1, m, tf=tf, states=states, show_iters=False)
t_energy = [QHO_energy(j) for j in range(0, states)]

# sort energy levels from lowest to highest as the algo may not extract the excited states in order.
sort_index = np.argsort(energy)

# initialize table to store the errors for the wavefunction and the energies
table = []

# graph all outputs of the test, calculate errors.
for j in range(0, states):
    # calculate errors for both possible orthonotmal states (+-1), plot the one with lower error
    plt.plot(x, U[sort_index[j]], label='FE Numeric Solution')
    plt.plot([], [], alpha=0, label='Theoretical Energy = {:.3E} J'.format(t_energy[j]))
    plt.plot([], [], alpha=0, label='Calculated Energy = {:.3E} J'.format(energy[sort_index[j]]))

    error_1 = np.sqrt(h) * linalg.norm(U[sort_index[j]] - state(x, j), ord=2)
    error_2 = np.sqrt(h) * linalg.norm(U[sort_index[j]] + state(x, j), ord=2)
    if error_1 < error_2:
        plt.plot(x, state(x, j), label='Analytic Solution')
        plt.plot([], [], alpha=0, label='2-Norm Error = {:.3E}'.format(error_1))
    else:
        plt.plot(x, -state(x, j), label='Analytic Solution')
        plt.plot([], [], alpha=0, label='2-Norm Error = {:.3E}'.format(error_2))

    # construct next row for the output table
    row = [j, min([error_1, error_2]), energy[sort_index[j]], t_energy[j], np.abs(energy[sort_index[j]]-t_energy[j])/t_energy[j]]
    table.append(row)

    plt.legend()
    plt.title('Q-Harmonic Oscillator, State = {}'.format(j))
    plt.show()

    # clear plot
    plt.figure()

headers = ["State", "2-Norm Error", "Calculated Energy", "Theoretical Energy", "Energy Relative Error"]
print(tabulate(table, headers, floatfmt=(None, '.5e', '.3f', '.3f', '.3%')))