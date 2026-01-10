from djitellopy import tello
import time
import KEYMODULE as dr
import cv2 
import numpy as np 
import math

############## PARAMETERS #############
fspeed = 117 / 10 # Forward speed
aspeed = 360 / 10 # Angular speed
interval = 0.25 

dinterval = fspeed * interval
ainterval = aspeed * interval 
#######################################

x, y = 500, 500 
a = 0
yaw = 0
points = []
global img 

# DRONE CONNECTION
drone = tello.Tello()
drone.connect()
print(f"BATTERY: {drone.get_battery()}%")
drone.streamon()

# DRONE CONTROL WINDOW
dr.init() 

def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50 
    global yaw, x, y, a
    d = 0
    move_angle = 0 
    moving = False

    # LEFT / RIGHT
    if dr.getKey("LEFT"): 
        lr = -speed
        d = dinterval 
        move_angle = 180 
        moving = True
    elif dr.getKey("RIGHT"): 
        lr = speed
        d = dinterval 
        move_angle = 0 
        moving = True
        
    # FORWARD / BACKWARD 
    if dr.getKey("UP"): 
        fb = speed
        d = dinterval 
        move_angle = 270
        moving = True
    elif dr.getKey("DOWN"): 
        fb = -speed
        d = -dinterval 
        move_angle = 90 
        moving = True

    # ROTATE 
    if dr.getKey("d"): 
        yv = speed
        yaw += ainterval
    elif dr.getKey("a"): 
        yv = -speed
        yaw -= ainterval

    # UP / DOWN 
    if dr.getKey("w"): ud = speed
    elif dr.getKey("s"): ud = -speed

    # LAND / TAKEOFF
    if dr.getKey("q"): drone.land(); time.sleep(2)
    elif dr.getKey("e"): drone.takeoff()
    
    # CAPTURE IMAGE FOR NAVIGATION
    if dr.getKey("c"): 
        cv2.imwrite(f'Resources/Image/{time.time()}.jpg', img_cam)


    time.sleep(interval) # REAL TIME DELAY 
    
    # OXY COORDINATES 
    if moving:
        angle_rad = math.radians(yaw + move_angle)        
        x += int(d * math.cos(angle_rad))
        y += int(d * math.sin(angle_rad))
    return [lr, fb, ud, yv, x, y]

# MAPPING 
def drawPoints(img, points):

    # DRONE TRAJECTORY 
    for point in points:
        cv2.circle(img, point, 2, (0, 0, 255), cv2.FILLED) 
    
    # DRONE POSITION AT TIME t 
    cv2.circle(img, points[-1], 5, (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'({(points[-1][0]-500)/100:.2f}, {-(points[-1][1]-500)/100:.2f})m',
                (points[-1][0]+10, points[-1][1]+30), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0, 255), 1)

while True:
    vals = getKeyboardInput()
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    # VIDEO 
    try:
        frame_read = drone.get_frame_read()
        myFrame = frame_read.frame
        if myFrame is not None:
            img_cam = cv2.resize(myFrame, (360, 240))
            cv2.imshow("TELLO'S CAMERA", img_cam)
    except Exception as e:
        print("CAMERA ERROR:", e)
    img_map = np.zeros((1000, 1000, 3), np.uint8)
    if len(points) == 0 or points[-1] != (vals[4], vals[5]):
        points.append((vals[4], vals[5]))
    drawPoints(img_map, points)

    cv2.imshow("TRAJECTORY MAP", img_map)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break
