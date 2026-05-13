import cv2
import numpy as np

def dibujar_contorno(frame, mask, color):
    # --- MEJORAS PARA REDUCIR RUIDO (Contornos erráticos) ---
    # 1. 'Opening': Elimina pequeños puntos blancos sueltos en el fondo (ruido)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    # 2. 'Closing': Rellena pequeños agujeros negros dentro del objeto detectado
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    # --------------------------------------------------------

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area=cv2.contourArea(c)
        if area > 3000:
            M=cv2.moments(c)
            if(M["m00"]==0): M["m00"]=1
            x=int(M["m10"]/M["m00"])
            y=int(M["m01"]/M["m00"])
            cv2.circle(frame,(x,y),7,color,-1)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, '{},{}'.format(x,y),(x+10,y), font, 0.75,color,1,cv2.LINE_AA)
            nuevo_contorno=cv2.convexHull(c)
            cv2.drawContours(frame, [nuevo_contorno], 0, color, 3)

captura = cv2.VideoCapture(1, cv2.CAP_DSHOW)

green_bajo=np.array([45, 50,20], np.uint8)
green_alto=np.array([75, 255,255], np.uint8)

red_bajo1=np.array([0, 100,20], np.uint8)
red_alto1=np.array([10, 255,255], np.uint8)

red_bajo2=np.array([165, 100,20], np.uint8)
red_alto2=np.array([179, 255,255], np.uint8)


yellow_bajo=np.array([25, 100,20], np.uint8)
yellow_alto=np.array([35, 255,255], np.uint8)

blue_bajo=np.array([90, 50,20], np.uint8)
blue_alto=np.array([125, 230,255], np.uint8)

captura.set(3, 1280) 
captura.set(4, 720)

while(captura.isOpened()):
    ret,frame=captura.read()

    if ret==True:
        # Efecto espejo
        frame = cv2.flip(frame, 1)

        # Suavizamos la imagen (Blur) antes de convertirla a HSV
        # Esto reduce muchísimo la "granulación" o ruido de la cámara
        frame_suavizado = cv2.GaussianBlur(frame, (5, 5), 0)
        frame_hsv=cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)
        mask_green=cv2.inRange(frame_hsv, green_bajo, green_alto)
        mask_red1=cv2.inRange(frame_hsv, red_bajo1, red_alto1)
        mask_red2=cv2.inRange(frame_hsv, red_bajo2, red_alto2)
        mask_yellow=cv2.inRange(frame_hsv, yellow_bajo, yellow_alto)
        mask_blue=cv2.inRange(frame_hsv, blue_bajo, blue_alto)
        dibujar_contorno(frame, mask_green, (0, 255, 0))
        dibujar_contorno(frame, mask_red1, (0, 0, 255))
        dibujar_contorno(frame, mask_red2, (0, 0, 255))
        dibujar_contorno(frame, mask_yellow, (0, 255, 255))
        dibujar_contorno(frame, mask_blue, (255,0,0))

        cv2.imshow('video' ,frame)
        cv2.imshow('mask', mask_green)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

captura.release()
cv2.destroyAllWindows()
