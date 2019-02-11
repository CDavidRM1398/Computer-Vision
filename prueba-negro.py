import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Negro 
    negro_bajos = np.array([0,0,0], dtype=np.uint8)
    negro_altos = np.array([35, 35, 35], dtype=np.uint8)
        
    mascara_negro = cv2.inRange(hsv, negro_bajos, negro_altos)
    
    kernel = np.ones((7,7),np.uint8)
    mascara_negro = cv2.morphologyEx(mascara_negro, cv2.MORPH_CLOSE, kernel)
    mascara_negro = cv2.morphologyEx(mascara_negro, cv2.MORPH_OPEN, kernel)
    mascara_negro = cv2.morphologyEx(mascara_negro, cv2.MORPH_GRADIENT, kernel)
    #mascara_negro = cv2.erode(mascara_negro,kernel,iterations = 1)
    
    mask = cv2.add(mascara_negro, mascara_negro)

    #res = cv2.bitwise_and(frame,frame, mask= mask)
    
    #gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(mask, (5,5), 0)
    canny = cv2.Canny(mask, 1, 2)
    im2, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    areas = [cv2.contourArea(c) for c in contours]
    i = 0
    for extension in areas:
        if extension > 600:
            actual = contours[i]
            approx = cv2.approxPolyDP(actual,0.05*cv2.arcLength(actual,True),True)
            if len(approx)==3:
                cv2.drawContours(frame,[actual],0,(0,0,255),2)
                cv2.drawContours(mask,[actual],0,(0,0,255),2)
            i = i+1
    
    
    #p=cv2.drawContours(mask, contours, -1, (0,0,255), 2)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()