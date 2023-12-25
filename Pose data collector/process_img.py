import cv2

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