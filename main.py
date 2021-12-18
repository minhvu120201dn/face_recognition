import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.__file__[:-12] + '\\data\\haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

labels = {}
with open('labels.pickle', 'rb') as file:
    og_labels = pickle.load(file)
    labels = {v:k for k,v in og_labels.items()}
    del og_labels

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame by frame
        ret, frame = cap.read()

        # Detect the face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5)
        for x,y,w,h in faces:
            color = (255,0,0) #BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = labels[id_]
            #text += ', conf = ' + str(round(conf,2))
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, text, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            #print(id_)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) == 27 or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()