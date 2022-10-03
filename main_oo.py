## Author: Emil Ostergaard
## Date: 3 June 2022

from matplotlib import projections
from backend_oo import *

num_of_wires_horizontal = 1
num_of_wires_vertical = 1
num_of_layers = 1
squeezing = 5

cluster = Cluster(num_of_wires_horizontal, num_of_wires_vertical, num_of_layers, squeezing)  

cluster.set_default_angles()

# cluster.plot()

desired_gate = [
    [1,0],
    [0,1]
]

cluster.find_angle_single_mode(0,desired_gate)

# cluster.plot(projection=1)

# print(cluster.wires_grid)
