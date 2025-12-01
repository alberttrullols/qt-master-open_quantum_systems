# 2LS and Thermal oscillator Transfer Tensor Calculation
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# latex stuff
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

dt = 0.2 # time step

Delta = 0.5 # detuning
Omega = 0 # Rabi frequency
nu = 1 # oscillator frequency
nbar = 0 # thermal occupation number
g = .1 # coupling strength
nmax = 10 # truncation of oscillator levels

# Pauli matrices
sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])

# Harmonic oscillator operators (for the bath)
n = np.diag(np.arange(nmax))
a = np.diag(np.sqrt(np.arange(nmax - 1) + 1), 1)
ad = a.T.conj()

# System operators
Ha = Delta * sz + Omega * sx
Ida = np.eye(2)

# Bath operators
Hv = nu * n
Idv = np.eye(nmax)

# Interaction
Hint = g * np.kron(sx, ad + a)

# Total Hamiltonian
H = np.kron(Ha, Idv) + np.kron(Ida, Hv) + Hint
Id = np.kron(Ida, Idv)

# oscillator initially in ground state
rhov0 = np.zeros((nmax, nmax))
rhov0[0, 0] = 1 

# Expanders and Tracers for reduced dynamics
Expander = np.kron(np.eye(4), rhov0.flatten()).T
Tracer = np.kron(np.eye(4), Idv.flatten())

# Sweep over kappa values
min_kappa = 0.0
max_kappa = 10.0 * g # set it even bigger to oberve how it goes to zero
kappa_list = np.linspace(min_kappa, max_kappa, 1000)
norm_T2_list = []

def liouvillian(kappa):
    return (1j * (np.kron(H, Id) - np.kron(Id, H.T)) +
            kappa * (nbar + 1) * (2 * np.kron(np.kron(Ida, a), np.kron(Ida, ad).T)
                                   - np.kron(np.kron(Ida, ad @ a), Id)
                                   - np.kron(Id, np.kron(Ida, ad @ a))) +
            kappa * nbar * (2 * np.kron(np.kron(Ida, ad), np.kron(Ida, a).T)
                            - np.kron(np.kron(Ida, a @ ad), Id)
                            - np.kron(Id, np.kron(Ida, a @ ad))))

for kappa in kappa_list:
    print(f"Calculating for kappa = {kappa:.2f}")

    # compute Liouvillian superoperator for current kappa
    L = liouvillian(kappa)
    
    # Full system dynamical maps
    EL = expm(L * dt)
    EL2 = expm(L * 2 * dt)

    # Reduced dynamical maps
    E1 = Tracer @ EL @ Expander
    E2 = Tracer @ EL2 @ Expander

    # Transfer tensor
    T2 = E2 - E1 @ E1

    norm_T2_list.append(np.linalg.norm(T2, 2)) 


# Remove the first two points from both lists (numerical artifact at kappa~0)
kappa_list = kappa_list[2:]
norm_T2_list = norm_T2_list[2:]

# Plot results
plt.figure()
plt.plot(kappa_list/g, norm_T2_list)
plt.tick_params(labelsize=12)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r"$\kappa/g$", fontsize=16)
plt.ylabel(r"$||T_2||$", fontsize=16)
plt.tight_layout()
plt.grid(True)
plt.savefig("transfer_tensor_norm.png")
plt.show()

