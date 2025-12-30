from djitellopy import tello
import time
import KEYMODULE as dr
import cv2 
import numpy as np 
import math

############## PARAMETER #############
fspeed = 117 / 10 #forward speed in cm/s
aspeed = 360 / 10 #angular speed in degrees/s
interval = 0.25

dinterval = fspeed * interval
ainterval = aspeed * interval 
#######################################

x,y = 500,500
a = 0
yaw = 0
points = []

dr.init()
#drone = tello.Tello()
#drone.connect()
global img 
#print (drone.get_battery())

def getKeyboardInput():
    lr,fb,ud,yv = 0,0,0,0
    speed = 25
    global yaw, x, y, a
    d = 0

    if dr.getKey("LEFT"): 
        lr = -speed
        d = dinterval 
        a = -180

    elif dr.getKey("RIGHT"): 
        lr = speed
        d = -dinterval 
        a = 180
        
    if dr.getKey("DOWN"): 
        fb = -speed
        d = dinterval 
        a = 90

    elif dr.getKey("UP"): 
        fb = speed
        d = -dinterval 
        a = -270

    if dr.getKey("s"): ud = -speed
    elif dr.getKey("w"): ud = speed

    if dr.getKey("d"): 
        yv = -speed
        yaw += ainterval
    elif dr.getKey("a"): 
        yv = speed
        yaw -= ainterval

    #if dr.getKey("q"): drone.land(); time.sleep(3)
    #elif dr.getKey("e"): drone.takeoff()

    if dr.getKey("c"): 
        cv2.imwrite(f'Resources/Image/{time.time()}.jpg',img)

    time.sleep(interval)
    a += yaw
    x += int (d * math.cos(math.radians(a))) 
    y += int (d * math.sin(math.radians(a))) 

    return [lr,fb,ud,yv,x,y]

def drawPoints(img, points):
    for point in points:
        cv2.circle(img,point,1,(0,0,255),cv2.FILLED)
    cv2.circle(img,points[-1],3,(0,255,0),cv2.FILLED)
    cv2.putText(img,f'({(points[-1][0]-500)/100},{(-points[-1][1]+500)/100})m',
                (points[-1][0]+10, points[-1][1]+30), cv2.FONT_HERSHEY_PLAIN,1,
                (255,0,255),1)

while True:
    val = getKeyboardInput()
    #drone.send_rc_control(val[0],val[1],val[2],val[3])
    img = np.zeros((1000,1000,3), np.uint8)
    points.append((val[4],val[5]))
    drawPoints(img,points)
    cv2.imshow("OUTPUT",img)
    cv2.waitKey(1)