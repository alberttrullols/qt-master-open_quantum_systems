#2LS and Thermal oscillator
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

dt=0.2
tmax=1000

Delta=0.5 # detuning
Omega=0 # Rabi frequency
Gamma=0.5 # decay rate
nu=1 # oscillator frequency
kappa=0.0 # oscillator decay rate
nbar=0 # thermal occupation number
g=.1 # coupling strength

nmax=10 # maximum number of oscillator levels


# pauli matrices sigma z, sigma x, sigma plus, sigma minus
sz= np.array([[1,0],
             [0,-1]])

sx= np.array([[0,1],
             [1,0]])

sp= np.array([[0,1],
             [0,0]])

sm= np.array([[0,0],
             [1,0]])

# harmonic operators
n=np.diag(np.arange(nmax)) # number operator
a=np.diag(np.sqrt(np.arange(nmax-1)+1),1) # annihilation operator
ad=a.T.conj() # creation operator

# Bath Space Operators
Hv=nu*n # Bath Hamiltonian (Harmonic oscillator) in bath space
Idv=np.eye(nmax) # identity in bath space

# System Space Operators
Ha = Delta*sz + Omega*sx  # System Hamiltonian (2LS) in system space
Ida = np.eye(2) # identity in system space

# Total Space Operators
Hint = g*np.kron(sx,ad+a) # Interaction Hamiltonian in tensor product space (total space)

H=np.kron(Ha,Idv)+np.kron(Ida,Hv)+Hint  # Total Hamiltonian in total space
Id=np.kron(Ida,Idv) # identity in total space

Ad=np.kron(Ida,ad) # total space annihilation operator
A=np.kron(Ida,a) # total space creation operator
N=Ad@A # total space number operator
SZ=np.kron(sz,Idv) # total space sigma z operator

# Liouvillian superoperator: Lrho = -i[H,rho] + kappa*(nbar+1)*D[a]rho + kappa*nbar*D[ad]rho 
# where D[a]rho = 2a·rho·a^dagger - a^dagger·a·rho - rho·a^dagger·a
# where D[ad]rho = 2a^dagger·rho·a - a·a^dagger·rho - rho·a·a^dagger

L=(1j*(np.kron(H,Id)-np.kron(Id,H.T))+
   kappa*(nbar)*(2*np.kron(Ad,A.T)-np.kron(A@Ad,Id)-np.kron(Id,A@Ad))+
   kappa*(nbar+1)*(2*np.kron(A,Ad.T)-np.kron(Ad@A,Id)-np.kron(Id,Ad@A)))


# dynamical map for time step dt
EL=expm(L*dt)

# bath initially in ground state
rhov0=np.diag(np.zeros(nmax))
rhov0[0,0]=1

# 2LS initially in excited state
rhoa0 = np.array([[1,0],
             [0,0]])

# total initial state
rho0=np.kron(rhoa0,rhov0)

# flatten initial state ie vectorize density matrix
rhot=rho0.flatten()

# arrays to store observables
nt=np.zeros(tmax)
szt=np.zeros(tmax)

# flatten operators for expectation value calculation
nv=N.flatten()
szv=SZ.flatten()

# evolve and store observables over time
for ti in np.arange(0,tmax,1):
    nt[ti]=np.real(nv@rhot)
    szt[ti]=np.real(szv@rhot)
    rhot=EL@rhot

# plot results
plt.plot(nt)
plt.plot((szt+1)/2)
plt.xlabel("Time step")
plt.ylabel("Observable")
plt.title("Time Evolution of Observables")
plt.legend(["Oscillator Number", "2LS Excitation Probability"])

if __name__ == "__main__":
    plt.savefig("output.png")