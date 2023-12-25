from datetime import datetime
import os
import csv
import cv2

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

def save_snapshot_to_folder(image,folder_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(folder_path, "Snapshots")
    
    # Create the Snapshots folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Count existing files in the folder to determine the sequence number
    sequence_number = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    filename = f"{timestamp}_{sequence_number + 1}"
 
    # Save the image to the Snapshots folder
    cv2.imwrite(os.path.join(folder_path, f"{filename}.jpg"), image)

    return filename  # Return the filename for status message

def Elucidean_dist(point1, point2):
   """ Euclidean distance between two points point1, point2 """
   diff_point1 = (point1[0] - point2[0]) ** 2
   diff_point2 = (point1[1] - point2[1]) ** 2
   return (diff_point1 + diff_point2) ** 0.5


