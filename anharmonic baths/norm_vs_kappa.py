# 2LS and Thermal oscillator Transfer Tensor Calculation
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
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

# Bath initially in ground state
rhov0 = np.zeros((nmax, nmax))
rhov0[0, 0] = 1 

# Initial total density matrix: system in ground state, bath in rhov0
Expander = np.kron(np.eye(4), rhov0.flatten()).T

# Partial trace over bath
Tracer = np.kron(np.eye(4), Idv.flatten())

# Sweep over kappa values
kappa_list = np.linspace(0, 100, 3)
norm_T2_list = []

for kappa in kappa_list:
    print(f"Calculating for kappa = {kappa:.2f}")
    
    # Liouvillian superoperator
    L = (1j * (np.kron(H, Id) - np.kron(Id, H.T)) +
         kappa * (nbar + 1) * (2*np.kron(np.kron(Ida, a), np.kron(Ida, ad).T)
         - np.kron(np.kron(Ida, ad@a), Id)
         - np.kron(Id, np.kron(Ida, ad@a))) +
         kappa * nbar * (2*np.kron(np.kron(Ida, ad), np.kron(Ida, a).T)
         - np.kron(np.kron(Ida, a@ad), Id)
         - np.kron(Id, np.kron(Ida, a@ad))))

    # Dynamical maps
    EL = expm(L * dt)
    EL2 = EL @ EL

    # Reduced maps
    E1 = Tracer @ EL @ Expander
    E2 = Tracer @ EL2 @ Expander

    # Transfer tensor
    T2 = E2 - E1 @ E1

    norm_T2_list.append(np.linalg.norm(T2, 2)) 

# Plot results
plt.figure()
plt.plot(kappa_list/g, norm_T2_list, label='Spectral Norm') 
# plt.legend()
plt.tick_params(labelsize=12)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.xlabel(r"$\kappa/g$")
plt.ylabel(r"||T₂||")
plt.title(r"Transfer Tensor Norm vs $\kappa$")
plt.grid(True)
plt.savefig("transfer_tensor_norm.png")
plt.show()

print("Done! Plot saved as transfer_tensor_norm.png")
