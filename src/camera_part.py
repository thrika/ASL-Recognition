
#Dataset preprocess
#1. RGB to HSV
#2. Morphological operation
#3. Save the binary image
#4. Gabor feature Extraction

#Training the dataset
#5. PCA, model save
#6. SVM, save model

#Testing the model
#7. Open web cam
#8. Detect hand using pre trained model
#9. RGB to HSV
#10. Morphological operation
#12. Save the binary image
#13. Gabor feature Extraction
#14. PCA
#15. Predict


import numpy as np
import cv2

bg = None
top, right, bottom, left = 10, 150, 310, 450

cap = cv2.VideoCapture(0)


def initialize():
    global cap, bg, top, right, bottom, left
    while (True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[top:bottom, right:left]
        roi2 = frame[top:bottom, right:left]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        
        img_HSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        img_HSV = cv2.medianBlur(img_HSV,3)
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        roi = cv2.bitwise_not(HSV_mask)


        et,thresh1 = cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('fframe', frame)
        cv2.imshow('fthresh', thresh1)
        
        k = cv2.waitKey(1)

        if k & 0xFF == ord('r'):
            cv2.imwrite("roi_e.png", roi2)
            cv2.destroyAllWindows()
            return


            

            


print("Initializing Frame, Keep Camera Still, Wait for few Seconds")
initialize()


