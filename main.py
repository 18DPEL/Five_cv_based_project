import dlib
import cv2
import numpy as np

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the image
image = cv2.imread('ayuvb.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop over the faces
for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray, face)

    # Loop over the 68 facial landmarks
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y

        # Draw a circle at each landmark position
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Display the image with landmarks
cv2.imshow('Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
