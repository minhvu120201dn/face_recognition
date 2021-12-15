import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.__file__[:-12] + '\\data\\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # Detect the face
    faces = face_cascade.detectMultiScale(frame, minNeighbors=6)
    for x,y,w,h in faces:
        color = (255,0,0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()