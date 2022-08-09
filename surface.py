import numpy as np
from matplotlib import pyplot as plt
import math

# Parameters
dB = -5
cluster_length = 5
num_of_data = math.floor(1.5*cluster_length**2) # number of data qubits in each layer
num_of_X_stabilizers = math.floor(cluster_length/2)**2 # number of X stabilizers in each layer
num_of_Z_stabilizers = math.ceil(cluster_length/2)**2 # number of Z stabilizers in each layer

# Noise 
r = -1/2*np.log(10^(dB/10))

sigmaGKP2 = np.exp(-2*r)/2
sigmaGate2 = np.exp(-2*r)

sigmaGKP = np.sqrt(sigmaGKP2)
sigmaGate = np.sqrt(sigmaGate2)

sigmaGate_data = sigmaGate*np.ones(num_of_data)
sigmaGate_X_stabilizers = sigmaGate*np.ones(num_of_X_stabilizers)
sigmaGate_Z_stabilizers = sigmaGate*np.ones(num_of_Z_stabilizers)

sigmaGKP_data = sigmaGKP*np.ones(num_of_data)
sigmaGKP_X_stabilizers = sigmaGKP*np.ones(num_of_X_stabilizers)
sigmaGKP_Z_stabilizers = sigmaGKP*np.ones(num_of_Z_stabilizers)

