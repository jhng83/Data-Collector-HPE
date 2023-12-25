import mediapipe as mp

def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)

def initialize_body_pose():
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.2, min_tracking_confidence=0.2)