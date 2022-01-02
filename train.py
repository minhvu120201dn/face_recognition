import os
import cv2
import numpy as np
import pickle
from imgaug import augmenters as iaa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv2.CascadeClassifier(cv2.__file__[:-12] + '\\data\\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

if __name__ == '__main__':
    # Get the data
    label_ids = {}
    X = []
    Y = []
    current_id = 0
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                path = os.path.join(root, file)
                image = cv2.imread(path, 0)
                label = os.path.basename(os.path.dirname(path))

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                
                X.append(image)
                Y.append(id_)
    del current_id

    # Face cascade and data augmentation
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-15, 15)),
        iaa.AdditiveGaussianNoise(scale=(10, 45))
    ])
    
    features = []
    labels = []
    for i in range(len(X)):
        label = Y[i]
        images = [X[i]] * 5
        images = seq(images=images)
        images.append(X[i])
        for image in images:
            detected_faces = face_cascade.detectMultiScale(image, minNeighbors=5)
            if len(detected_faces) == 0:
                continue
            for x, y, w, h in detected_faces:
                features.append(cv2.resize(image[y:y+h, x:x+w], (300,300)))
                labels.append(label)
                break

    # Train the model and save
    recognizer.train(features, np.array(labels))
    recognizer.save('trainer.yml')
    print('recognizer -> trainer.yml')

    with open('labels.pickle', 'wb') as file:
        pickle.dump(label_ids, file)
    print('label_ids =', label_ids, '-> labels.pickle')