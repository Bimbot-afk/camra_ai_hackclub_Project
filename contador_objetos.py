import imutils
import cv2
import imutils
import numpy as np

imagen_o=cv2.imread('moneda.jpg')
imagen=cv2.imread('moneda.jpg', 0)
imagen=imutils.resize(imagen, width=1000)
imagen_o=imutils.resize(imagen_o, width=1000)

_,ther=cv2.threshold(imagen, 220, 255, cv2.THRESH_BINARY_INV)
cntrs, _ = cv2.findContours(ther, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

'''
print('contornos:', len(cntrs))

cv2.drawContours(imagen_o, cntrs, -1, (0,0,255), 2)

cv2.imshow('imagen', imagen_o)
cv2.imshow('ther', ther)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
cv2.imshow('1', ther)
font= cv2.FONT_HERSHEY_SIMPLEX
moneda_num = 1
objetos_validos = [] # Lista para guardar solo los objetos reales

for c in cntrs:
     # Ignorar contornos pequeños que son ruido
     if cv2.contourArea(c) < 1000:
         continue
         
     objetos_validos.append(c) # Guardamos el contorno de la moneda
         
     M=cv2.moments(c)
     if(M["m00"]==0): M["m00"]=1
     x=int(M["m10"]/M["m00"])
     y=int(M["m01"]/M["m00"])

     mensaje= 'Num:' + str(moneda_num)
     cv2.putText(imagen_o,mensaje,(x-40,y), font,0.75,(0,0,255),2,cv2.LINE_AA)
     cv2.drawContours(imagen_o, [c], 0, (0,0,255), 3)
     cv2.imshow('imagen', imagen_o)
     cv2.waitKey(0)
     moneda_num += 1

cv2.waitKey(0)
print('Objetos:', len(objetos_validos))
cv2.destroyAllWindows()