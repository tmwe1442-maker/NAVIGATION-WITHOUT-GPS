from djitellopy import tello
import time
import KEYMODULE as dr
import cv2 

dr.init()
drone = tello.Tello()
drone.connect()
global img 
print (drone.get_battery())
drone.streamon()

def getKeyboardInput():
    lr,fb,ud,yv = 0,0,0,0
    speed = 50
    if dr.getKey("LEFT"): lr = -speed
    elif dr.getKey("RIGHT"): lr = speed

    if dr.getKey("DOWN"): fb = -speed
    elif dr.getKey("UP"): fb = speed

    if dr.getKey("s"): ud = -speed
    elif dr.getKey("w"): ud = speed

    if dr.getKey("d"): yv = -speed
    elif dr.getKey("a"): yv = speed

    if dr.getKey("q"): drone.land(); time.sleep(3)
    elif dr.getKey("e"): drone.takeoff()

    if dr.getKey("c"): 
        cv2.imwrite(f'Resources/Image/{time.time()}.jpg',img)

    return [lr,fb,ud,yv]

while True:
    val = getKeyboardInput()
    drone.send_rc_control(val[0],val[1],val[2],val[3])
    time.sleep(0.05)
    img = drone.get_frame_read().frame
    img = cv2.resize(img,(360, 240))
    cv2.imshow("STREAMING SCREEN",img)
    cv2.waitkey(1)