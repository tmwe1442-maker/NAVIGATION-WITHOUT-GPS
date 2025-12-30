from djitellopy import tello
from time import sleep 

drone = tello.Tello()
drone.connect()
print (drone.get_battery())

#BASIC MOVEMENT
#NEG (LEFT, BACK) POS (RIGHT, FOR) YAW (TURN)
drone.takeoff()
drone.send_rc_control (0,50,0,0)
sleep (2)
drone.send_rc_control (50,0,0,0)
sleep (2)
drone.send_rc_control (0,0,0,0)
drone.land()

#IMAGE CAPTURE
import cv2 
drone.streamon()
while True:
     img = drone.get_frame_read().frame
     img = cv2.resize(img,(360, 240))
     cv2.imshow("STREAMING SCREEN",img)
     cv2.waitkey(1)

