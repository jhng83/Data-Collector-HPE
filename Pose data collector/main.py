import PySimpleGUI as sg
from bodypose import Collect_Pose
from headpose import Head_Pose_Estimation

sg.theme('DarkAmber')  # Set the theme for the UI

# Define the layout of the UI
layout = [
    [sg.Button('Run Body Pose Estimation', size=(20, 2), key='-BODYPOSE-')],
    [sg.Button('Run Head Pose Estimation', size=(20, 2), key='-HEADPOSE-')],
    [sg.Button('Exit', size=(20, 2))]
]

# Create the window
window = sg.Window('Pose Estimation UI', layout)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == '-BODYPOSE-':
        Collect_Pose()
    elif event == '-HEADPOSE-':
        Head_Pose_Estimation()

# Close the window
window.close()
