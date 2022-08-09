## Author: Emil Ostergaard
## Date: 3 June 2022

## Importing all neccesary libraries
from cmath import nan
import numpy as np
import math
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
import copy
from mpl_toolkits import mplot3d

# This function generates the symplectic matrix for EPR state generation from two initially squeezed modes
def epr_symplectic(num_modes):

    #Delay on B mode

    num_modes = int(num_modes/2)

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

# This function generates a symplectic equivalent to rotating all modes of one biparition of the adjacency matrix by pi/2, thus transforming a H-graph into an approximate CV cluster state. 
def generate_S_h2c(modes, num_modes):
    num_modes = int(num_modes/2)
    S_h2c = np.identity(num_modes*4)
    for mode in modes:
        S_h2c[mode][mode] = 0
        S_h2c[mode][mode+2*num_modes] = -1
        S_h2c[mode+2*num_modes][mode] = 1
        S_h2c[mode+2*num_modes][mode+2*num_modes] = 0

    return S_h2c

# This function generates a symplectic matrix for the beamsplitter network used for cluster state generation from an EPR state
def cluster_symplectic(num_modes, bs_1=0, bs_2=0, bs_3=0):

    #Beamsplitter for 1D cluster generation

    num_modes = int(num_modes/2)

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

# Coordinates for A & B modes aligned in time
def get_coordinate(i, dim_x, dim_y, dim_z):
    is_odd = i % 2
    i = math.floor(i/2)
    y = math.floor(i/dim_z) + 0.3*is_odd
    i = i % dim_z
    z = math.floor(i/(dim_x))
    x = i % (dim_x)
    return [x,y,z]

# Coordinates for A & B modes skewed in time (B modes shifted by one time delay)
def get_coordinate_skewed(i, dim_x, dim_y, dim_z):
    is_odd = i % 2
    i = math.floor(i/2)
    y = math.floor(i/dim_z) + 0.3*is_odd
    i = i % dim_z
    z = math.floor(i/(dim_x))
    x = i % (dim_x)+1*is_odd
    return [x,y,z]

# Coordinates for A & B modes skewed in time (B modes shifted by one time delay and 2D layer shifted by time delay)
def get_coordinate_skewed_2(i, dim_x, dim_y, dim_z):
    is_odd = i % 2
    i = math.floor(i/2)
    y = math.floor(i/dim_z) + 0.3*is_odd
    i = i % dim_z
    z = math.floor(i/(dim_x))+1*is_odd
    x = i % (dim_x)+1*is_odd
    return [x,y,z]    

# Draw lines between mode coordinates
def draw_line(i,k,a,colour, coordinates, ax):
    a_coordinates = coordinates[i]
    b_coordinates = coordinates[k]
    ax.plot3D([a_coordinates[0],b_coordinates[0]],[a_coordinates[1],b_coordinates[1]],[a_coordinates[2],b_coordinates[2]], colour, alpha=a)

def round_matrix(M):
    rounded_M = np.zeros(np.shape(M))
    for i in range(np.shape(M)[0]):
        for j in range(np.shape(M)[1]):
            rounded_M[i][j] = round(100*np.real(M[i][j]))/100

    return rounded_M

def gate_diff(gate, desired_gate):
    diff_matrix = gate-desired_gate
    diff = 0
    for row in diff_matrix:
        for element in row:
            diff += np.real(element)

    return diff

class Cluster():
    def __init__(self, num_of_wires_horizontal, num_of_wires_vertical, num_of_layers, squeezing):
        self.num_of_wires_horizontal = num_of_wires_horizontal
        self.num_of_wires_vertical = num_of_wires_vertical
        self.num_of_layers = num_of_layers
        self.squeezing = squeezing

        self.num_of_wires = self.num_of_wires_horizontal*self.num_of_wires_vertical
        
        self.x_dim = self.num_of_wires_horizontal*2 + 2
        self.y_dim = self.num_of_wires_vertical*2 + 4
        self.z_dim = self.x_dim*self.y_dim

        self.bs_1 = 1
        self.bs_2 = self.bs_1 + self.x_dim
        self.bs_3 = self.bs_2 + self.z_dim

        self.num_of_modes = self.z_dim*self.num_of_layers*2
        self.wires_grid = []
        self.wires_list = []

        for i in range(self.num_of_wires_vertical):
            self.wires_grid.append([])
            for _ in range(self.num_of_wires_horizontal):
                self.wires_grid[i].append([])

        self.buffer_modes = []

        for i in range(self.num_of_layers):
            self.buffer_modes.extend(list(range(i*self.z_dim*2, self.x_dim*2+i*self.z_dim*2))) #bottom layer modes
            self.buffer_modes.extend(list(range(self.y_dim*self.x_dim*2-1+i*self.z_dim*2, self.y_dim*self.x_dim*2-2*self.x_dim-1+i*self.z_dim*2, -1))) #top layer modes
            self.buffer_modes.extend(list(range(self.x_dim*2+i*self.z_dim*2, self.y_dim*self.x_dim*2-2*self.x_dim-1+i*self.z_dim*2, self.x_dim*2))) #left side modes
            self.buffer_modes.extend(list(range(self.x_dim*4-1+i*self.z_dim*2, self.y_dim*self.x_dim*2-1+i*self.z_dim*2, self.x_dim*2))) #right side modes
            self.buffer_modes.extend(list(range(self.x_dim*2+i*self.z_dim*2+4, self.x_dim*4+i*self.z_dim*2, 4))) #2D bottom layer
            self.buffer_modes.extend(list(range(self.y_dim*self.x_dim*2-1+i*self.z_dim*2-4*self.x_dim+4, self.y_dim*self.x_dim*2-1+i*self.z_dim*2-2*self.x_dim, 4))) #2D top layer 

        self.buffer_modes_and_angles = {}
        for mode in self.buffer_modes:
            self.buffer_modes_and_angles[mode]=0

        self.Z = np.identity(self.num_of_modes)*np.exp(-2*self.squeezing)*1j
        self.transform_Z(epr_symplectic(self.num_of_modes))
        self.identify_bipartition()
        self.transform_Z(generate_S_h2c(self.partition_modes,self.num_of_modes))        
        self.transform_Z(cluster_symplectic(self.num_of_modes, 1, self.x_dim+1))
        self.Z_prior_measurements = copy.deepcopy(self.Z)
        self.measure_nodes_Z(self.buffer_modes_and_angles)

        # This function will transform an adjacency matrix, Z, according to the transformation rules from Menicucci et. al. 2011
    def transform_Z(self,symplectic):
        A = symplectic[:int(np.shape(symplectic)[0]/2),:int(np.shape(symplectic)[0]/2)]
        B = symplectic[int(np.shape(symplectic)[0]/2):,:int(np.shape(symplectic)[0]/2)]
        C = symplectic[:int(np.shape(symplectic)[0]/2),int(np.shape(symplectic)[0]/2):]
        D = symplectic[int(np.shape(symplectic)[0]/2):,int(np.shape(symplectic)[0]/2):]

        self.Z = (C + D@self.Z)@np.linalg.inv(A + B@self.Z)

    # This function seperated the graph of an adjacency matrix, Z, into two bipartitions. (Note: This has only been tested for the EPR state level and works at this level).
    def identify_bipartition(self):
        self.partition_modes = list(np.arange(int(np.shape(self.Z)[0])))
        for mode in self.partition_modes:
            test_modes = copy.deepcopy(self.partition_modes)
            test_modes.remove(mode)
            for test_mode in test_modes:
                if abs(self.Z[mode][test_mode]) > 0.1 or abs(self.Z[test_mode][mode]) > 0.1:
                    self.partition_modes.remove(test_mode)

    # This function will measure modes at a specified angle and return the result ajacency matrix after measurement
    def measure_nodes_Z(self, modes_and_angles):

        num_of_modes = self.num_of_modes
        
        S_rot = np.diag(np.ones(num_of_modes*2))

        for mode in modes_and_angles:
            angle = modes_and_angles[mode]
            S_rot[mode][mode] = np.cos(angle)
            S_rot[mode+num_of_modes][mode+num_of_modes] = np.cos(angle)
            S_rot[mode][mode+num_of_modes] = np.sin(angle)
            S_rot[mode+num_of_modes][mode] = -np.sin(angle)

        self.transform_Z(S_rot)

        for mode in modes_and_angles:
            for i in range(np.shape(self.Z)[0]):
                self.Z[mode][i] = 0
                self.Z[i][mode] = 0

    def set_default_angles(self):

        self.default_pi4_rotations = []
        self.default_minus_pi4_rotations = []

        for i in range(self.num_of_layers):
            for k in range(self.y_dim-2):
                for l in range(int(self.x_dim/2)):
                    if (l+1)%2==0:
                        self.default_minus_pi4_rotations.append(2*self.x_dim*(k+1)+i*self.z_dim*2+l*4+1)
                        self.default_minus_pi4_rotations.append(2*self.x_dim*(k+1)+i*self.z_dim*2+l*4+2)
                    else:
                        self.default_pi4_rotations.append(2*self.x_dim*(k+1)+i*self.z_dim*2+l*4+1)
                        self.default_pi4_rotations.append(2*self.x_dim*(k+1)+i*self.z_dim*2+l*4+2)

        for i in range(self.num_of_layers):
            for k in range(int((self.y_dim-2)/2)):
                for l in range(self.num_of_wires_horizontal):
                    if (k+1)%2==0:
                        self.default_minus_pi4_rotations.append(2*self.x_dim*(2*k+2)+i*self.z_dim*2+l*4+4)
                        self.default_minus_pi4_rotations.append(2*self.x_dim*(2*k+1)+i*self.z_dim*2+l*4+3)
                    else:
                        self.default_pi4_rotations.append(2*self.x_dim*(2*k+2)+i*self.z_dim*2+l*4+4)
                        self.default_pi4_rotations.append(2*self.x_dim*(2*k+1)+i*self.z_dim*2+l*4+3)

        for i in range(self.num_of_layers):
            for k in range(self.num_of_wires_vertical):
                for l in range(self.num_of_wires_horizontal):
                    self.wires_grid[k][l].append(2*self.x_dim*(2*k+2)+i*self.z_dim*2+l*4+3)
                    self.wires_grid[k][l].append(2*self.x_dim*(2*k+3)+i*self.z_dim*2+l*4+4)

        for i in range(self.num_of_wires_vertical):
            for j in range(self.num_of_wires_horizontal):
                self.wires_list.append(self.wires_grid[i][j])

        self.default_modes_and_angles = {}

        for mode in self.default_pi4_rotations:
            self.default_modes_and_angles[mode]=np.pi/4

        for mode in self.default_minus_pi4_rotations:
            self.default_modes_and_angles[mode]=-np.pi/4

        self.buffer_modes = []

        for i in range(self.num_of_layers):
            self.buffer_modes.extend(list(range(i*self.z_dim*2, self.x_dim*2+i*self.z_dim*2))) #bottom layer modes
            self.buffer_modes.extend(list(range(self.y_dim*self.x_dim*2-1+i*self.z_dim*2, self.y_dim*self.x_dim*2-2*self.x_dim-1+i*self.z_dim*2, -1))) #top layer modes
            self.buffer_modes.extend(list(range(self.x_dim*2+i*self.z_dim*2, self.y_dim*self.x_dim*2-2*self.x_dim-1+i*self.z_dim*2, self.x_dim*2))) #left side modes
            self.buffer_modes.extend(list(range(self.x_dim*4-1+i*self.z_dim*2, self.y_dim*self.x_dim*2-1+i*self.z_dim*2, self.x_dim*2))) #right side modes
            self.buffer_modes.extend(list(range(self.x_dim*2+i*self.z_dim*2+4, self.x_dim*4+i*self.z_dim*2, 4))) #2D bottom layer
            self.buffer_modes.extend(list(range(self.y_dim*self.x_dim*2-1+i*self.z_dim*2-4*self.x_dim+4, self.y_dim*self.x_dim*2-1+i*self.z_dim*2-2*self.x_dim, 4))) #2D top layer 

        for mode in self.buffer_modes:
            self.default_modes_and_angles[mode]=0

        self.measure_nodes_Z(self.default_modes_and_angles)

    def plot(self, projection=2):
        fig = plt.figure()
        ax = plt.axes(projection='3d')  

        plt.grid(b=None)
        plt.axis('off')

        a_coord, b_coord = [], []

        if projection == 0:
            coordinates = np.array([get_coordinate(i, self.x_dim, self.y_dim, self.z_dim) for i in range(self.num_of_modes)])
        elif projection == 1:
            coordinates = np.array([get_coordinate_skewed(i, self.x_dim, self.y_dim, self.z_dim) for i in range(self.num_of_modes)])
        elif projection == 2:
            coordinates = np.array([get_coordinate_skewed_2(i, self.x_dim, self.y_dim, self.z_dim) for i in range(self.num_of_modes)])

        for i in range(len(coordinates)):
            if i%2 == 0:
                a_coord.append(list(coordinates[i]))
            else:
                b_coord.append(list(coordinates[i]))

        a_coord, b_coord = np.array(a_coord), np.array(b_coord)

        ax.scatter3D(a_coord[:,0], a_coord[:,1], a_coord[:,2], color='red')
        ax.scatter3D(b_coord[:,0], b_coord[:,1], b_coord[:,2], color='blue')
        ax.axes.set_xlim3d(left=0, right=self.x_dim)
        ax.axes.set_ylim3d(bottom=0, top=self.num_of_layers-0.7) 
        ax.axes.set_zlim3d(bottom=0, top=self.y_dim) 

        offset = 0.01

        cov = self.Z

        for i in range(self.num_of_modes):
            for k in range(i+1, self.num_of_modes):
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

    def save_cluster(self, name='temp'):
        np.savetxt(name+'.csv', self.Z/np.tanh(2*self.squeezing), delimiter=',')

    def apply_identity_gate(self, wire, layer):
        pass

    def calculate_gate(self, modes_and_angles):
        A = np.block([[
            np.zeros((self.num_of_wires,self.num_of_wires)), 
            np.zeros((self.num_of_wires,np.shape(self.Z_prior_measurements)[0]))
            ],[
            np.zeros((np.shape(self.Z_prior_measurements)[0],self.num_of_wires)),
            self.Z_prior_measurements
        ]])

        S_cz = np.block([
            [np.identity(np.shape(A)[0]), np.zeros(np.shape(A))],
            [A, np.identity(np.shape(A)[0])]
        ])

        B = np.identity(np.shape(A)[0])
        counter = 0
        for i in range(self.num_of_wires_vertical):
            for j in range(self.num_of_wires_horizontal):
                mode = self.wires_grid[i][j][1] + self.num_of_wires
                B[counter][counter] = 1/np.sqrt(2)
                B[mode][mode] = 1/np.sqrt(2)
                B[counter][mode] = -1/np.sqrt(2)
                B[mode][counter] = 1/np.sqrt(2)
                counter += 1

        S_bs = np.block([
            [B, np.zeros(np.shape(B))],
            [np.zeros(np.shape(B)), B]
        ])

        S_rot = np.diag(np.ones((self.num_of_modes+self.num_of_wires)*2))

        for mode in modes_and_angles:
            angle = modes_and_angles[mode]
            S_rot[mode][mode] = np.cos(angle)
            S_rot[mode+(self.num_of_modes+self.num_of_wires)][mode+(self.num_of_modes+self.num_of_wires)] = np.cos(angle)
            S_rot[mode][mode+(self.num_of_modes+self.num_of_wires)] = np.sin(angle)
            S_rot[mode+(self.num_of_modes+self.num_of_wires)][mode] = -np.sin(angle)

        self.symplectic = S_rot@S_bs@S_cz

        # plt.matshow(np.real(self.symplectic))
        # plt.show()

        for i in range(self.num_of_wires):
            
            if i == 0:
                Z_top = np.hstack((self.symplectic[self.wires_list[i][-2]+self.num_of_wires,0:self.num_of_wires],self.symplectic[self.wires_list[i][-2]+self.num_of_wires,self.num_of_modes+self.num_of_wires:]))
                Z_bottom = np.hstack((self.symplectic[self.wires_list[i][-2]+self.num_of_wires*2+self.num_of_modes,0:self.num_of_wires],self.symplectic[self.wires_list[i][-2]+self.num_of_wires*2+self.num_of_modes,self.num_of_modes+self.num_of_wires:]))
                Y_top = self.symplectic[self.wires_list[i][-2]+self.num_of_wires,self.num_of_wires:self.num_of_modes+self.num_of_wires]
                Y_bottom = self.symplectic[self.wires_list[i][-2]+2*self.num_of_wires+self.num_of_modes,self.num_of_wires:self.num_of_modes+self.num_of_wires]
            
                U = np.array(self.symplectic[0:self.wires_list[i][-2]+self.num_of_wires,self.num_of_wires:self.num_of_modes+self.num_of_wires])
                V = np.hstack((self.symplectic[0:self.wires_list[i][-2]+self.num_of_wires, 0:self.num_of_wires], self.symplectic[0:self.wires_list[i][-2]+self.num_of_wires, self.num_of_modes+self.num_of_wires:]))

            else:
                Z_top = np.vstack([
                    Z_top,
                    np.hstack((self.symplectic[self.wires_list[i][-2]+self.num_of_wires,0:self.num_of_wires],self.symplectic[self.wires_list[i][-2]+self.num_of_wires,self.num_of_modes+self.num_of_wires:]))
                ])
                Z_bottom = np.vstack([
                    Z_bottom,
                    np.hstack((self.symplectic[self.wires_list[i][-2]+self.num_of_wires*2+self.num_of_modes,0:self.num_of_wires],self.symplectic[self.wires_list[i][-2]+self.num_of_wires*2+self.num_of_modes,self.num_of_modes+self.num_of_wires:]))
                ])
                Y_top = np.vstack([
                    Y_top,
                    self.symplectic[self.wires_list[i][-2]+self.num_of_wires,self.num_of_wires:self.num_of_modes+self.num_of_wires]
                ])
                Y_bottom = np.vstack([
                    Y_bottom,
                    self.symplectic[self.wires_list[i][-2]+self.num_of_wires*2+self.num_of_modes,self.num_of_wires:self.num_of_modes+self.num_of_wires]
                ])

                U = np.vstack([
                    U,
                    self.symplectic[self.wires_list[i-1][-2]+self.num_of_wires+1:self.wires_list[i][-2]+self.num_of_wires,self.num_of_wires:self.num_of_modes+self.num_of_wires]
                ])
                V = np.vstack([
                    V,
                    np.hstack((self.symplectic[self.wires_list[i-1][-2]+self.num_of_wires+1:self.wires_list[i][-2]+self.num_of_wires, 0:self.num_of_wires], self.symplectic[self.wires_list[i-1][-2]+self.num_of_wires+1:self.wires_list[i][-2]+self.num_of_wires, self.num_of_modes+self.num_of_wires:]))
                ])

        U = np.vstack([
            U,
            self.symplectic[self.wires_list[-1][-2]+self.num_of_wires+1:self.num_of_modes+self.num_of_wires,self.num_of_wires:self.num_of_modes+self.num_of_wires]
        ])

        V = np.vstack([
            V,
            np.hstack((self.symplectic[self.wires_list[-1][-2]+self.num_of_wires+1:self.num_of_modes+self.num_of_wires, 0:self.num_of_wires], self.symplectic[self.wires_list[-1][-2]+self.num_of_wires+1:self.num_of_modes+self.num_of_wires, self.num_of_modes+self.num_of_wires:]))
        ])

        Z = np.vstack([
            Z_top,
            Z_bottom
        ])

        Y = np.vstack([
            Y_top,
            Y_bottom
        ])

        # print(np.shape(U))
        # plt.matshow(np.real(U))
        # # plt.show()

        # plt.matshow(np.real(self.symplectic))
        
        # plt.matshow(np.real(V))
        # # plt.matshow(np.real(Z))
        # # plt.matshow(np.real(Y))
        # plt.show()

        self.M = Z - Y@np.linalg.inv(U)@V

        self.G = self.M[:self.num_of_wires*2,:self.num_of_wires*2]

        self.N = self.M[:self.num_of_wires*2,self.num_of_wires*2:]

        

    def modes_to_consider_2_mode(self, wire_1, wire_2):
        modes = []
        for i in range(len(self.wires_list[wire_1])):
            modes.append(self.wires_list[wire_1][i])
            modes.append(self.wires_list[wire_2][i])
            modes.append(int((self.wires_list[wire_1][i]+self.wires_list[wire_2][i])/2))

    def modes_to_consider_single_mode(self, wire):
        return self.wires_list[wire]

    def find_angle_single_mode(self, wire, gate):
        modes = []
        modes.extend(self.modes_to_consider_single_mode(wire))

        for i in range(len(modes)):
            modes[i] = modes[i]+self.num_of_wires

        modes.append(wire)

        modes.sort()

        gate_modes_and_angles = {}

        for mode, angle in self.default_modes_and_angles.items():
            gate_modes_and_angles[mode+self.num_of_wires]=angle

        del modes[-2]

        for mode in modes:
            gate_modes_and_angles[mode]=np.random.ranf()*np.pi-np.pi/2

        self.calculate_gate(gate_modes_and_angles)

        implemented_gate = np.array([
            [self.G[wire][wire], self.G[wire][wire+self.num_of_wires]],
            [self.G[wire+self.num_of_wires][wire], self.G[wire+self.num_of_wires][wire+self.num_of_wires]]
        ])

        diff = gate_diff(implemented_gate, gate)

        while diff > 0.05:
            gradients = {}
            step_size = 0.001*np.pi
            decent_size = step_size
            for mode in modes:
                gate_modes_and_angles_temp = copy.deepcopy(gate_modes_and_angles)
                gate_modes_and_angles_temp[mode]=gate_modes_and_angles_temp[mode]+step_size
                # print(gate_modes_and_angles_temp[mode]-gate_modes_and_angles[mode])
                # print(gate_modes_and_angles_temp[modes[0]], gate_modes_and_angles_temp[modes[1]])
                self.calculate_gate(gate_modes_and_angles_temp)
                implemented_gate_temp = np.array([
                    [self.G[wire][wire], self.G[wire][wire+self.num_of_wires]],
                    [self.G[wire+self.num_of_wires][wire], self.G[wire+self.num_of_wires][wire+self.num_of_wires]]
                ])
                diff_plus = gate_diff(implemented_gate_temp, gate)

                gate_modes_and_angles_temp[mode]-=step_size
                self.calculate_gate(gate_modes_and_angles_temp)
                implemented_gate_temp = np.array([
                    [self.G[wire][wire], self.G[wire][wire+self.num_of_wires]],
                    [self.G[wire+self.num_of_wires][wire], self.G[wire+self.num_of_wires][wire+self.num_of_wires]]
                ])
                diff_minus = gate_diff(implemented_gate_temp, gate)

                gradients[mode] = (diff_plus-diff_minus)/(step_size)

            for mode in gradients:
                gate_modes_and_angles[mode]-= decent_size*gradients[mode]

            self.calculate_gate(gate_modes_and_angles)

            implemented_gate = np.array([
                [self.G[wire][wire], self.G[wire][wire+self.num_of_wires]],
                [self.G[wire+self.num_of_wires][wire], self.G[wire+self.num_of_wires][wire+self.num_of_wires]]
            ])

            diff = gate_diff(implemented_gate, gate)

            print(implemented_gate)
            print(diff)

        for mode in modes:
            print(gate_modes_and_angles[mode])