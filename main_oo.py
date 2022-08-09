## Author: Emil Ostergaard
## Date: 3 June 2022

from backend_oo import *

num_of_wires_horizontal = 6
num_of_wires_vertical = 6
num_of_layers = 1
squeezing = 2

cluster = Cluster(num_of_wires_horizontal, num_of_wires_vertical, num_of_layers, squeezing)  

cluster.set_default_angles()

# cluster.plot()

desired_gate = [
    [1/np.sqrt(2),-1/np.sqrt(2)],
    [1/np.sqrt(2),1/np.sqrt(2)]
]

# cluster.find_angle_single_mode(0,desired_gate)

cluster.plot()

print(cluster.wires_grid)
