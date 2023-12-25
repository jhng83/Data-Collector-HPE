import PySimpleGUI as sg

def create_layout_HPE():
    
    layout = [
        [sg.Text("Head Pose Estimation", font=("Helvetica", 20), justification="center", size=(30, 1), relief=sg.RELIEF_RIDGE)],
        [sg.Text("Pitch:", font=("Helvetica", 16), size=(10, 1)), sg.Text("", key="-PITCH-", font=("Helvetica", 16), size=(10, 1))],
        [sg.Text("Yaw:", font=("Helvetica", 16), size=(10, 1)), sg.Text("", key="-YAW-", font=("Helvetica", 16), size=(10, 1))],
        [sg.Image(key="-IMAGE-", size=(400, 400))],
        [sg.Text("", key="-STATUS-", size=(30, 1), justification="center")],  # New Text element for status
        [sg.Button("Take Snapshot", size=(15, 1), font=("Helvetica", 14)), sg.Button("Back", size=(10, 1), font=("Helvetica", 14))]
    ]
    return layout

def create_layout_BPE():
    
    layout = [
        [sg.Text("Body Pose Estimation", font=("Helvetica", 20), justification="center", size=(30, 1), relief=sg.RELIEF_RIDGE)],
        [sg.Image(key="-IMAGE-", size=(400, 400))],
        [sg.Text("", key="-STATUS-", size=(30, 1), justification="center")],  # New Text element for status
        [sg.Button("Take Snapshot", size=(15, 1), font=("Helvetica", 14)), sg.Button("Back", size=(10, 1), font=("Helvetica", 14))]
    ]
    return layout

def create_window(name,layout):
    
    window = sg.Window("Head Pose Estimation", layout, resizable=True, finalize=True)
    
    return window

def close_window(event):
    
    if event == sg.WIN_CLOSED or event == "Back":
        ret = 1
    
    else:
        ret = 0
        
    return ret
    