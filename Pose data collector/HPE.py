import os
import csv

import cv2
import numpy as np

import util
import cal_pose
import create_UI
import process_img

def Head_Pose_Estimation():
    face_mesh = cal_pose.initialize_face_mesh()
    cap = process_img.initialize_camera()
    count = 0
    
    # Create the PySimpleGUI window 
    window = create_UI.create_window("Head Pose Estimation",create_UI.create_layout_HPE())
 
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
        image = process_img.preprocess_image(image)

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
                    filename = util.save_snapshot_to_folder_with_csv(cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR), pitch, yaw, os.getcwd(), csv_filename)
                    window["-STATUS-"].update(f"Snapshot saved: {filename}")

        image = process_img.postprocess_image(image)
        window["-IMAGE-"].update(data=cv2.imencode(".png", image)[1].tobytes())

        event, values = window.read(timeout=20)
        
        ret = create_UI.close_window(event)

        if ret == 1:
            break

    window.close()
    cap.release()

if __name__ == "__main__":
    Head_Pose_Estimation()

