import mediapipe as mp

def initialize_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3)