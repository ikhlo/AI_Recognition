# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:28:58 2021

@author: ikhla
"""
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np
import cvlib as cv
import cv2

labels = ['Man', 'Woman']

model = load_model('gender_detection_64_966.h5')

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    
    status, frame = webcam.read()
    
    face, confidence = cv.detect_face(frame)
    
    for f in face:
        X_start, Y_start, X_end, Y_end = f[:4]
        
        cv2.rectangle(frame, (X_start, Y_start), (X_end, Y_end), (0, 255, 0), 2)
        
        try :
            cropface = frame[Y_start:Y_end, X_start:X_end].copy()
        
            cropface = cv2.resize(cropface, (64,64))
            cropface = img_to_array(cropface) / 255.0
            
            
            prediction = model.predict(np.expand_dims(cropface, axis=0))[0]
            label_idx = np.argmax(prediction)
            
            gender = labels[label_idx]
            result = f"{gender} : {np.round(prediction[label_idx]*100, 1)} %"
            
            cv2.putText(frame, result, (X_start, Y_start-10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)
            cv2.imshow('gender recognition', frame)
        except : 
            pass
        
    if cv2.waitKey(1) & 0xFF == ord('q'): break

webcam.release()
cv2.destroyAllWindows()

