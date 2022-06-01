from array import array
from backend import *

num_of_wires_horizontal = 4
num_of_wires_vertical = 4
num_of_layers = 2

bs_1 = num_of_wires_horizontal*2 + 2
bs_2 = num_of_wires_vertical*2 + 4
bs_3 = num_of_layers*bs_2*bs_1

wires_grid = []

for i in range(num_of_wires_vertical):
    wires_grid.append([])
    for _ in range(num_of_wires_horizontal):
        wires_grid[i].append([])

buffer_modes = []

for i in range(num_of_layers):
    buffer_modes.extend(list(range(i*bs_3,bs_1*2+i*bs_3))) #bottom layer modes
    buffer_modes.extend(list(range(bs_2*bs_1*2-1+i*bs_3,bs_2*bs_1*2-2*bs_1-1+i*bs_3,-1))) #top layer modes
    buffer_modes.extend(list(range(bs_1*2+i*bs_3,bs_2*bs_1*2-2*bs_1-1+i*bs_3,bs_1*2))) #left side modes
    buffer_modes.extend(list(range(bs_1*4-1+i*bs_3,bs_2*bs_1*2-1+i*bs_3,bs_1*2))) #right side modes
    buffer_modes.extend(list(range(bs_1*2+i*bs_3+4, bs_1*4+i*bs_3, 4))) #2D bottom layer
    buffer_modes.extend(list(range(bs_2*bs_1*2-1+i*bs_3-4*bs_1+4,bs_2*bs_1*2-1+i*bs_3-2*bs_1,4))) #2D top layer

default_pi4_rotations = []
default_minus_pi4_rotations = []

for i in range(num_of_layers):
    for k in range(bs_2-2):
        for l in range(int(bs_1/2)):
            if (l+1)%2==0:
                default_minus_pi4_rotations.append(2*bs_1*(k+1)+i*bs_3+l*4+1)
                default_minus_pi4_rotations.append(2*bs_1*(k+1)+i*bs_3+l*4+2)
            else:
                default_pi4_rotations.append(2*bs_1*(k+1)+i*bs_3+l*4+1)
                default_pi4_rotations.append(2*bs_1*(k+1)+i*bs_3+l*4+2)

for i in range(num_of_layers):
    for k in range(int((bs_2-2)/2)):
        for l in range(num_of_wires_horizontal):
            if (k+1)%2==0:
                default_minus_pi4_rotations.append(2*bs_1*(2*k+2)+4+i*bs_3+l*4)
                default_minus_pi4_rotations.append(2*bs_1*(2*k+1)+3+i*bs_3+l*4)
            else:
                default_pi4_rotations.append(2*bs_1*(2*k+2)+4+i*bs_3+l*4)
                default_pi4_rotations.append(2*bs_1*(2*k+1)+3+i*bs_3+l*4)

for i in range(num_of_layers):
    for k in range(num_of_wires_vertical):
        for l in range(num_of_wires_horizontal):
            wires_grid[k][l].append(2*bs_1*(2*k+2)+3+i*bs_3+l*4)
            wires_grid[k][l].append(2*bs_1*(2*k+3)+4+i*bs_3+l*4)

modes_and_angles = []
for mode in buffer_modes:
    modes_and_angles.append((mode,0))

for mode in default_pi4_rotations:
    modes_and_angles.append((mode,np.pi/4))

for mode in default_minus_pi4_rotations:
    modes_and_angles.append((mode,-np.pi/4))    

Z = make_cluster(1, bs_1, bs_3, 5, 2)
Z = measure_nodes_Z(modes_and_angles,Z,bs_3)

draw_cluster(Z, bs_3, 1, bs_1, bs_3/2, 2)

print(wires_grid)
