import cv2
import mediapipe as mp
import numpy as np
import sys
import PySimpleGUI as sg
from datetime import datetime
import os
import csv

def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)

def initialize_camera():
    return cv2.VideoCapture(0)

def preprocess_image(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    scale_percent = 80
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image.flags.writeable = False
    return image

def postprocess_image(image):
    image.flags.writeable = True
    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), dim, interpolation=cv2.INTER_AREA)

def create_layout():
    layout = [
        [sg.Text("Head Pose Estimation", font=("Helvetica", 20), justification="center", size=(30, 1), relief=sg.RELIEF_RIDGE)],
        [sg.Text("Pitch:", font=("Helvetica", 16), size=(10, 1)), sg.Text("", key="-PITCH-", font=("Helvetica", 16), size=(10, 1))],
        [sg.Text("Yaw:", font=("Helvetica", 16), size=(10, 1)), sg.Text("", key="-YAW-", font=("Helvetica", 16), size=(10, 1))],
        [sg.Image(key="-IMAGE-", size=(400, 400))],
        [sg.Text("", key="-STATUS-", size=(30, 1), justification="center")],  # New Text element for status
        [sg.Button("Take Snapshot", size=(15, 1), font=("Helvetica", 14)), sg.Button("Exit", size=(10, 1), font=("Helvetica", 14))]
    ]
    return layout

def save_snapshot_to_folder_with_csv(image, pitch, yaw, folder_path, csv_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(folder_path, "Snapshots")
    
    # Create the Snapshots folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Count existing files in the folder to determine the sequence number
    sequence_number = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    filename = f"{timestamp}_{sequence_number + 1}"
 
    # Save the image to the Snapshots folder
    cv2.imwrite(os.path.join(folder_path, f"{filename}.jpg"), image)

    # Write pitch and yaw values to CSV file
    csv_filepath = os.path.join(folder_path, csv_filename)
    with open(csv_filepath, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([filename, pitch, yaw])

    return filename  # Return the filename for status message

def main():
    face_mesh = initialize_face_mesh()
    cap = initialize_camera()
    count = 0

    # Create the PySimpleGUI window with finalize=True
    window = sg.Window("Head Pose Estimation", create_layout(), resizable=True, finalize=True)

    # Create the "Snapshots" folder if it doesn't exist
    folder_path = os.path.join(os.getcwd(), "Snapshots")
    os.makedirs(folder_path, exist_ok=True)

    # Create or open the CSV file for writing
    csv_filename = "head_pose_data.csv"
    csv_filepath = os.path.join(folder_path, csv_filename)
    if not os.path.isfile(csv_filepath):
        with open(csv_filepath, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Filename", "Pitch", "Yaw"])

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        count += 1
        image = preprocess_image(image)

        # Get the result
        results = face_mesh.process(image)

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []

        # if face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        # Get the 2D Coordinates
                        face_2d.append([x, y])
                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)
                
                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                
                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)
                noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                noseEndPoint2D, jacobian = cv2.projectPoints(
                    noseEndPoints3D, rot_vec, trans_vec, cam_matrix, dist_matrix)

                #  draw nose line
                #p1 = (int(nose_2d[0]), int(nose_2d[1]))
                #p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
                # cv2.line(image, p1, p2, (110, 220, 0),thickness=2, lineType=cv2.LINE_AA)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                pitch = round(angles[0] * 360, 0) - 5
                yaw = round(angles[1] * 360, 0)

                # Update PySimpleGUI elements
                window["-PITCH-"].update(str(pitch))
                window["-YAW-"].update(str(yaw))

                # Save snapshot with pitch, yaw, and filename to CSV when "Take Snapshot" button is clicked
                event, values = window.read(timeout=20)
                if event == "Take Snapshot":
                    filename = save_snapshot_to_folder_with_csv(cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR), pitch, yaw, os.getcwd(), csv_filename)
                    window["-STATUS-"].update(f"Snapshot saved: {filename}")

        image = postprocess_image(image)
        window["-IMAGE-"].update(data=cv2.imencode(".png", image)[1].tobytes())

        event, values = window.read(timeout=20)

        if event == sg.WIN_CLOSED or event == "Exit":
            break

    window.close()
    cap.release()

if __name__ == "__main__":
    main()

