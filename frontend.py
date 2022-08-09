## Author: Emil Ostergaard
## Date: 3 June 2022

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib

# sg.Window(title="3D Cluster Simulator", layout=[[]], margins=(500, 400)).read()

user_input_column = [
    [
        sg.Text("Number of wires:"), sg.InputText()
    ],
    [
        sg.Text("Number of layers:"), sg.InputText()
    ],
    [
        sg.Submit(), sg.Cancel()
    ]

]

user_output_column = [
    [
        sg.Text("User Ouput 1")
    ],
    [
        sg.Text("User Ouput 2")
    ]

]

layout = [
    [
        sg.Column(user_input_column),
        sg.VSeperator(),
        sg.Column(user_output_column),
    ]
]

# Create the window
window = sg.Window("Simulator", layout, margins=(500, 400))

# Create an event loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

window.close()