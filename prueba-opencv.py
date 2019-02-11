import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Verdes 
    verde_bajos = np.array([28,84,45], dtype=np.uint8)
    verde_altos = np.array([100, 255, 210], dtype=np.uint8)
    #Azules:
    azul_bajos = np.array([100,65,75], dtype=np.uint8)
    azul_altos = np.array([130, 255, 255], dtype=np.uint8)
    #Rojos:
    rojo_bajos1 = np.array([0,65,75], dtype=np.uint8)
    rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
    rojo_bajos2 = np.array([240,65,75], dtype=np.uint8)
    rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)
    
    mascara_azul = cv2.inRange(hsv, azul_bajos, azul_altos)
    mascara_verde = cv2.inRange(hsv, verde_bajos, verde_altos)
    mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
    mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)

    kernel = np.ones((1,1),np.uint8)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, kernel)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, kernel)
    mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_CLOSE, kernel)
    mascara_azul = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernel)
    mascara_rojo1 = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_CLOSE, kernel)
    mascara_rojo1 = cv2.morphologyEx(mascara_rojo1, cv2.MORPH_OPEN, kernel)
    mascara_rojo2 = cv2.morphologyEx(mascara_rojo2, cv2.MORPH_CLOSE, kernel)
    mascara_rojo2 = cv2.morphologyEx(mascara_rojo2, cv2.MORPH_OPEN, kernel)
    
    mask = cv2.add(mascara_rojo1, mascara_rojo2)
    mask = cv2.add(mask, mascara_verde)
    mask = cv2.add(mask, mascara_azul)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()