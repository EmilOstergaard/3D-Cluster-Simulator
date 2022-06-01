## Author: Emil Ostergaard
## Date: 30 May 2022

## Importing all neccesary libraries
import numpy as np
import math
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
import copy
from mpl_toolkits import mplot3d

## Defining all functions

# This function will transform an adjacency matrix, Z, according to the transformation rules from Menicucci et. al. 2011
def transform_Z(symplectic, Z):
    A = symplectic[:int(np.shape(symplectic)[0]/2),:int(np.shape(symplectic)[0]/2)]
    B = symplectic[int(np.shape(symplectic)[0]/2):,:int(np.shape(symplectic)[0]/2)]
    C = symplectic[:int(np.shape(symplectic)[0]/2),int(np.shape(symplectic)[0]/2):]
    D = symplectic[int(np.shape(symplectic)[0]/2):,int(np.shape(symplectic)[0]/2):]

    return (C + D@Z)@np.linalg.inv(A + B@Z)

# This function seperated the graph of an adjacency matrix, Z, into two bipartitions. (Note: This has only been tested for the EPR state level and works at this level).
def identify_bipartition(Z):
    modes = list(np.arange(int(np.shape(Z)[0])))
    for mode in modes:
        test_modes = copy.deepcopy(modes)
        test_modes.remove(mode)
        for test_mode in test_modes:
            if abs(Z[mode][test_mode]) > 0.1 or abs(Z[test_mode][mode]) > 0.1:
                modes.remove(test_mode)
    return modes

# This function generates a symplectic equivalent to rotating all modes of one biparition of the adjacency matrix by pi/2, thus transforming a H-graph into an approximate CV cluster state. 
def generate_S_h2c(modes, num_modes):
    S_h2c = np.identity(num_modes*4)
    for mode in modes:
        S_h2c[mode][mode] = 0
        S_h2c[mode][mode+2*num_modes] = -1
        S_h2c[mode+2*num_modes][mode] = 1
        S_h2c[mode+2*num_modes][mode+2*num_modes] = 0

    return S_h2c

# This function generates the symplectic matrix for EPR state generation from two initially squeezed modes
def epr_symplectic(num_modes):

    #Delay on B mode

    S_phase_delay = np.zeros((num_modes*4, num_modes*4))

    for i in range(num_modes):
        S_phase_delay[i*2,i*2] = 1
        S_phase_delay[(num_modes+i)*2,(num_modes+i)*2] = 1
        S_phase_delay[i*2+1,(num_modes+i)*2+1] = -1
        S_phase_delay[(num_modes+i)*2+1,i*2+1] = 1

    #Beamsplitter for EPR state generation

    S_bs0 = np.array([])

    for i in range(num_modes):
        bs_block = np.array([[1/np.sqrt(2),-1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]])
        if i == 0:
            S_bs0 = np.block([
                [bs_block, np.zeros((2,2))],
                [np.zeros((2,2)), bs_block]
            ])
        else:
            size = np.shape(S_bs0)
            S_bs0 = np.block([
                [S_bs0[:int(size[0]/2),:int(size[0]/2)],np.zeros((int(size[1]/2),2)), S_bs0[int(size[0]/2):,:int(size[0]/2)],np.zeros((int(size[1]/2),2))],
                [np.zeros((2,int(size[0]/2))), bs_block, np.zeros((2,int(size[0]/2))), np.zeros((2,2))],
                [S_bs0[int(size[0]/2):,:int(size[0]/2)], np.zeros((int(size[1]/2),2)), S_bs0[int(size[0]/2):,int(size[0]/2):], np.zeros((int(size[1]/2),2))],
                [np.zeros((2,int(size[0]/2))), np.zeros((2,2)), np.zeros((2,int(size[0]/2))), bs_block]
            ])

    return S_bs0@S_phase_delay

# This function generates a symplectic matrix for the beamsplitter network used for cluster state generation from an EPR state
def cluster_symplectic(num_modes, bs_1=0, bs_2=0, bs_3=0):

    #Beamsplitter for 1D cluster generation

    if bs_1 == 0:

        S_bs1 = np.diag(np.ones(num_modes*4))

    else:

        bs1_block = np.diag(np.ones(num_modes*2))

        for i in range(num_modes):
            if 2*i+1+(2*bs_1-1) < num_modes*2:
                bs1_block[2*i+1,2*i+1] = 1/np.sqrt(2)
                bs1_block[2*i+1,2*i+1+(2*bs_1-1)] = -1/np.sqrt(2)
                bs1_block[2*i+1+(2*bs_1-1),2*i+1] = 1/np.sqrt(2)
                bs1_block[2*i+1+(2*bs_1-1),2*i+1+(2*bs_1-1)] = 1/np.sqrt(2)

        S_bs1 = np.block([[bs1_block, np.zeros((num_modes*2, num_modes*2))],[np.zeros((num_modes*2, num_modes*2)), bs1_block]]) 

    #Beamsplitter for 2D cluster generation

    if bs_2 == 0:

        S_bs2 = np.diag(np.ones(num_modes*4))

    else:

        bs2_block = np.diag(np.ones(num_modes*2))

        for i in range(num_modes):
            if 2*i+1+(2*bs_2-1) < num_modes*2:
                bs2_block[2*i+1,2*i+1] = 1/np.sqrt(2)
                bs2_block[2*i+1,2*i+1+(2*bs_2-1)] = -1/np.sqrt(2)
                bs2_block[2*i+1+(2*bs_2-1),2*i+1] = 1/np.sqrt(2)
                bs2_block[2*i+1+(2*bs_2-1),2*i+1+(2*bs_2-1)] = 1/np.sqrt(2)

        S_bs2 = np.block([[bs2_block, np.zeros((num_modes*2, num_modes*2))],[np.zeros((num_modes*2, num_modes*2)), bs2_block]])  

    #Beamsplitter for 3D cluster generation

    if bs_3 == 0:

        S_bs3 = np.diag(np.ones(num_modes*4))
    
    else:

        bs3_block = np.diag(np.ones(num_modes*2))

        for i in range(num_modes):
            if 2*i+1+(2*bs_3-1) < num_modes*2:
                bs3_block[2*i+1,2*i+1] = 1/np.sqrt(2)
                bs3_block[2*i+1,2*i+1+(2*bs_3-1)] = -1/np.sqrt(2)
                bs3_block[2*i+1+(2*bs_3-1),2*i+1] = 1/np.sqrt(2)
                bs3_block[2*i+1+(2*bs_3-1),2*i+1+(2*bs_3-1)] = 1/np.sqrt(2)

        S_bs3 = np.block([[bs3_block, np.zeros((num_modes*2, num_modes*2))],[np.zeros((num_modes*2, num_modes*2)), bs3_block]]) 

    return S_bs3@S_bs2@S_bs1

def make_cluster(bs_1, bs_2, num_modes ,r, level=0):
    Z = np.identity(num_modes*2)*np.exp(-2*r)*1j
    Z = transform_Z(epr_symplectic(num_modes), Z)
    bipartition = identify_bipartition(Z) # Identify modes to be transformed for H-graph to Approximate CV cluster state  (ACVCS) conversion
    S_h2c = generate_S_h2c(bipartition,num_modes) # Symplectic matrix for H-graph -> ACVCS
    Z = transform_Z(S_h2c, Z) # Adjacency matrix for ACVCS for EPR state
    if level==1:
        return transform_Z(cluster_symplectic(num_modes, bs_1), Z)
    elif level==2:
        return transform_Z(cluster_symplectic(num_modes, bs_1, bs_2+1), Z)
    else:
        return Z

# This function will measure modes at a specified angle and return the result ajacency matrix after measurement
def measure_nodes_Z(modes_and_angles, Z, num_modes):
    
    S_rot = np.diag(np.ones(num_modes*4))
    for (mode, angle) in modes_and_angles:
        S_rot[mode][mode] = np.cos(angle)
        S_rot[mode+2*num_modes][mode+2*num_modes] = np.cos(angle)
        S_rot[mode][mode+2*num_modes] = np.sin(angle)
        S_rot[mode+2*num_modes][mode] = -np.sin(angle)

    Z = transform_Z(S_rot, Z)

    for (mode,angle) in modes_and_angles:
        for i in range(np.shape(Z)[0]):
            Z[mode][i] = 0
            Z[i][mode] = 0

    return Z

# Coordinates for A & B modes aligned in time
def get_coordinate(i, bs_1, bs_2, bs_3):
    is_odd = i % 2
    i = math.floor(i/2)
    y = math.floor(i/bs_3) + 0.3*is_odd
    i = i % bs_3
    z = math.floor(i/(bs_2))
    x = i % (bs_2)
    return [x,y,z]

# Coordinates for A & B modes skewed in time (B modes shifted by one time delay)
def get_coordinate_skewed(i, bs_1, bs_2, bs_3):
    is_odd = i % 2
    i = math.floor(i/2)
    y = math.floor(i/bs_3) + 0.3*is_odd
    i = i % bs_3
    z = math.floor(i/(bs_2))
    x = i % (bs_2)+1*is_odd
    return [x,y,z]

# Coordinates for A & B modes skewed in time (B modes shifted by one time delay and 2D layer shifted by time delay)
def get_coordinate_skewed_2(i, bs_1, bs_2, bs_3):
    is_odd = i % 2
    i = math.floor(i/2)
    y = math.floor(i/bs_3) + 0.3*is_odd
    i = i % bs_3
    z = math.floor(i/(bs_2))+1*is_odd
    x = i % (bs_2)+1*is_odd
    return [x,y,z]    

# Draw lines between mode coordinates
def draw_line(i,k,a,colour, coordinates, ax):
    a_coordinates = coordinates[i]
    b_coordinates = coordinates[k]
    ax.plot3D([a_coordinates[0],b_coordinates[0]],[a_coordinates[1],b_coordinates[1]],[a_coordinates[2],b_coordinates[2]], colour, alpha=a)

# This function will draw the graph state of the cluster in a cubic grid formation
def draw_cluster(Z, num_modes, bs_1, bs_2, bs_3, level=0):
    fig = plt.figure()
    ax = plt.axes(projection='3d')  

    plt.grid(b=None)
    plt.axis('off')

    a_coord, b_coord = [], []

    if level == 0:
        coordinates = np.array([get_coordinate(i, bs_1, bs_2, bs_3) for i in range(num_modes*2)])
    elif level == 1:
        coordinates = np.array([get_coordinate_skewed(i, bs_1, bs_2, bs_3) for i in range(num_modes*2)])
    elif level == 2:
        coordinates = np.array([get_coordinate_skewed_2(i, bs_1, bs_2, bs_3) for i in range(num_modes*2)])

    for i in range(len(coordinates)):
        if i%2 == 0:
            a_coord.append(list(coordinates[i]))
        else:
            b_coord.append(list(coordinates[i]))

    a_coord, b_coord = np.array(a_coord), np.array(b_coord)

    ax.scatter3D(a_coord[:,0], a_coord[:,1], a_coord[:,2], color='red')
    ax.scatter3D(b_coord[:,0], b_coord[:,1], b_coord[:,2], color='blue')
    ax.axes.set_xlim3d(left=0, right=bs_2)
    ax.axes.set_ylim3d(bottom=0, top=num_modes/bs_3-0.7) 
    ax.axes.set_zlim3d(bottom=0, top=bs_3/bs_2) 

    offset = 0.01

    cov = Z

    for i in range(num_modes*2):
        for k in range(i+1, num_modes*2):
            if cov[i][k]>(1-offset) and cov[i][k]<(1+offset):
                #weight is tanh(2r) 
                draw_line(i,k,0.5,'purple', coordinates, ax)

            elif cov[i][k]<-(1-offset) and cov[i][k]>-(1+offset):
                #weight is -tanh(2r)
                draw_line(i,k,0.5,'green', coordinates, ax)

            elif cov[i][k]>0.5*(1-offset) and cov[i][k]<0.5*(1+offset):
                #weight is 1/2*tanh(2r)
                draw_line(i,k,0.3,'purple', coordinates, ax)

            elif cov[i][k]<-0.5*(1-offset) and cov[i][k]>-0.5*(1+offset):
                #weight is -1/2*tanh(2r)
                draw_line(i,k,0.3,'green', coordinates, ax)

            elif cov[i][k]<-(1/np.sqrt(2))*(1-offset) and cov[i][k]>-(1/np.sqrt(2))*(1+offset):
                #weight is -1/sqrt(2)*tanh(2r)
                draw_line(i,k,0.35,'green', coordinates, ax)

            elif cov[i][k]>(1/np.sqrt(2))*(1-offset) and cov[i][k]<(1/np.sqrt(2))*(1+offset):
                #weight is 1/sqrt(2)*tanh(2r)
                draw_line(i,k,0.35,'purple', coordinates, ax)

            elif cov[i][k]<-(1/(2*np.sqrt(2)))*(1-offset) and cov[i][k]>-(1/(2*np.sqrt(2)))*(1+offset):
                #weight is -1/2*sqrt(2)*tanh(2r)
                draw_line(i,k,0.25,'green', coordinates, ax)

            elif cov[i][k]>(1/(2*np.sqrt(2)))*(1-offset) and cov[i][k]<(1/(2*np.sqrt(2)))*(1+offset):
                #weight is 1/2*sqrt(2)*tanh(2r)
                draw_line(i,k,0.25,'purple', coordinates, ax)

            elif cov[i][k]>(1/4)*(1-offset) and cov[i][k]<(1/4)*(1+offset):
                #weight is 1/4*tanh(2r)
                draw_line(i,k,0.2,'purple', coordinates, ax)

            elif cov[i][k]<(-1/4)*(1-offset) and cov[i][k]>(-1/4)*(1+offset):
                #weight is -1/4*tanh(2r)
                draw_line(i,k,0.2,'green', coordinates, ax)

            elif cov[i][k]>(1/(4*np.sqrt(2)))*(1-offset) and cov[i][k]<(1/(4*np.sqrt(2)))*(1+offset):
                #weight is 1/4*sqrt(2)*tanh(2r)
                draw_line(i,k,0.15,'purple', coordinates, ax)

            elif cov[i][k]<(-1/(4*np.sqrt(2)))*(1-offset) and cov[i][k]>(-1/(4*np.sqrt(2)))*(1+offset):
                #weight is -1/4*sqrt(2)*tanh(2r)
                draw_line(i,k,0.15,'green', coordinates, ax)

            elif cov[i][k]>(1/8)*(1-offset) and cov[i][k]<(1/8)*(1+offset):
                #weight is 1/8*tanh(2r)
                draw_line(i,k,0.1,'purple', coordinates, ax)

            elif cov[i][k]<(-1/8)*(1-offset) and cov[i][k]>(-1/8)*(1+offset):
                #weight is -1/8*tanh(2r)
                draw_line(i,k,0.1,'green', coordinates, ax)

            elif abs(cov[i][k])>0.001:
                if cov[i][k] < 0:
                    draw_line(i,k,0.5,'brown', coordinates, ax)
                else:
                    draw_line(i,k,0.5,'black', coordinates, ax)

    plt.show()

def save_cluster(Z, name='temp'):
    np.savetxt(name+'.csv', cov/np.tanh(2*r), delimiter=',')

def round_matrix(M):
    rounded_M = np.zeros(np.shape(M))
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            rounded_M[i][j] = round(100*np.real(M[i][j]))/100

    return rounded_M

