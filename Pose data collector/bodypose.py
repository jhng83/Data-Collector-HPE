import os
import csv

import cv2
import mediapipe as mp

import util
import cal_pose
import create_UI
import process_img

# Function to process a single image
def process_image(image_path, pose_landmarks):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Extract x, y coordinates of each joint
    landmarks = {f"Landmark_{i}": (lm.x * w, lm.y * h) if lm.x and lm.y else "na" for i, lm in enumerate(pose_landmarks.landmark, start=1)}

    return landmarks

# Function to process all images in a folder
def process_folder(folder_path, csv_file):
    # Create or load CSV file
    if not os.path.exists(csv_file):
        # Create a new CSV file with headers
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ["Image"] + [f"Landmark_{i}" for i in range(1, 33)]
            writer.writerow(headers)
    else:
        # Load existing CSV file
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Read the header row
            existing_data = {row[0]: row[1:] for row in reader}

    pose = cal_pose.initialize_body_pose()

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read image
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            h, w, _ = image.shape

            # Convert image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Run Mediapipe Pose
            results = pose.process(image)

            # Process landmarks if pose is detected
            if results.pose_landmarks:
                landmarks = process_image(image_path, results.pose_landmarks)

                # Add landmarks to the CSV file
                row_data = [filename] + [landmarks.get(header, "na") for header in headers[1:]]

                if not os.path.exists(csv_file):
                    with open(csv_file, 'a', nadmarksewline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(row_data)
                else:
                    # Check if data already exists
                    if filename not in existing_data or existing_data[filename] != row_data[1:]:
                        existing_data[filename] = row_data[1:]
                        with open(csv_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(row_data)

def Collect_Pose():
    
    # Specify the folder containing images
    image_folder_path = "Snapshots"
    
    # Specify the CSV file name
    csv_file_name = "bodypose.csv"
    
    cap = process_img.initialize_camera()
    
    # Create the PySimpleGUI window 
    window = create_UI.create_window("Body Pose Estimation",create_UI.create_layout_BPE())
    # Create the "Snapshots" folder if it doesn't exist
    folder_path = os.path.join(os.getcwd(), "Snapshots")
    os.makedirs(folder_path, exist_ok=True)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        image = process_img.preprocess_image(image)
        
        # Save snapshot with pitch, yaw, and filename to CSV when "Take Snapshot" button is clicked
        event, values = window.read(timeout=20)
        if event == "Take Snapshot":
            filename = util.save_snapshot_to_folder(cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR), os.getcwd())
            window["-STATUS-"].update(f"Snapshot saved: {filename}")
            
        image = process_img.postprocess_image(image)
        window["-IMAGE-"].update(data=cv2.imencode(".png", image)[1].tobytes())
        
        event, values = window.read(timeout=20)
        
        ret = create_UI.close_window(event)

        if ret == 1:
            break

    window.close()
    cap.release()
    process_folder(image_folder_path, csv_file_name)

if __name__ == "__main__":
    Collect_Pose()
