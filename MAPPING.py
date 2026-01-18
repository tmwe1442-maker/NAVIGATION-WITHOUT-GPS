# FILE: my_drone_control.py
from djitellopy import tello
import time
import cv2 
import numpy as np 
import math
import pygame

class DroneController:
    def __init__(self):
        # 1. CẤU HÌNH
        self.USE_SIMULATION_ONLY = False 
        
        # 2. KHỞI TẠO PYGAME
        pygame.init()
        self.win = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("CLICK VAO DAY DE DIEU KHIEN")
        
        # 3. THÔNG SỐ VẬT LÝ
        self.yaw = 90  # Cố định đầu hướng Lên (90 độ) để vẽ cho đẹp
        self.x, self.y = 1000, 1000
        self.points = []
        
        # Vận tốc (Chỉ còn X và Y)
        self.vel_x = 0.0
        self.vel_y = 0.0
        
        # Tinh chỉnh độ mượt
        self.ACCEL = 0.3       
        self.FRICTION = 0.9    
        self.MAX_SPEED = 7.0   

        # 4. KẾT NỐI DRONE
        self.drone = tello.Tello()
        self.has_drone = False
        
        if not self.USE_SIMULATION_ONLY:
            print("Dang tim Tello (Doi 5-7s)...")
            try:
                self.drone.connect()
                self.drone.streamon()
                self.has_drone = True
                print(f"KET NOI THANH CONG! Pin: {self.drone.get_battery()}%")
            except Exception as e:
                print("KET NOI THAT BAI -> CHUYEN SANG GIA LAP.")
                self.has_drone = False

    def get_control_step(self):
        # Refresh Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pass

        keys = pygame.key.get_pressed()
        
        force_x = 0
        force_y = 0
        
        # --- LOGIC DI CHUYỂN (Chỉ có X/Y) ---
        
        # TRÁI / PHẢI (A / D)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:  force_x = -1 
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: force_x = 1  
        
        # LÊN / XUỐNG (W / S)
        if keys[pygame.K_UP] or keys[pygame.K_w]:    force_y = -1 # Lên
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:  force_y = 1  # Xuống
        
        # BỎ HOÀN TOÀN PHÍM XOAY Q/E

        # Chức năng khác
        if keys[pygame.K_SPACE] and self.has_drone: self.drone.takeoff()
        if keys[pygame.K_TAB] and self.has_drone: self.drone.land()
        
        # Chụp ảnh
        img_cam = None
        if self.has_drone:
            frame_read = self.drone.get_frame_read()
            img_cam = frame_read.frame
            if keys[pygame.K_c]: 
                cv2.imwrite(f'Resources/{time.time()}.jpg', img_cam)
                time.sleep(0.2) 

        # --- TÍNH TOÁN VẬT LÝ ---
        
        # 1. Tăng tốc
        if force_x != 0: self.vel_x += force_x * self.ACCEL
        if force_y != 0: self.vel_y += force_y * self.ACCEL

        # 2. Ma sát
        self.vel_x *= self.FRICTION
        self.vel_y *= self.FRICTION

        # 3. Giới hạn Max Speed
        self.vel_x = np.clip(self.vel_x, -self.MAX_SPEED, self.MAX_SPEED)
        self.vel_y = np.clip(self.vel_y, -self.MAX_SPEED, self.MAX_SPEED)

        # 4. Ngắt nếu quá chậm
        if abs(self.vel_x) < 0.1: self.vel_x = 0
        if abs(self.vel_y) < 0.1: self.vel_y = 0

        # --- GỬI LỆNH CHO DRONE THẬT ---
        if self.has_drone:
            # Gửi lệnh bay: LR, FB, UD, Yaw=0 (Luôn không xoay)
            self.drone.send_rc_control(
                int(self.vel_x * 4), 
                int(-self.vel_y * 4), 
                0, 
                0 # Yaw luôn bằng 0
            )

        # --- CẬP NHẬT MAP ---
        # Không cập nhật self.yaw nữa -> Mũi tên luôn hướng lên trên
        
        dx = self.vel_x
        dy = self.vel_y
        
        self.x += int(dx)
        self.y += int(dy)

        if len(self.points) == 0 or self.points[-1] != (self.x, self.y):
            self.points.append((self.x, self.y))

        # Hiển thị Pygame
        self.win.fill((0,0,0))
        font = pygame.font.SysFont(None, 24)
        status = "DRONE: ON" if self.has_drone else "SIMULATION"
        img = font.render(f"{status} | NO ROTATION", True, (0, 255, 0))
        self.win.blit(img, (20, 20))
        pygame.display.update()

        return np.array([float(dx), float(dy)]), img_cam, self.points

    def get_drone_height(self):
        if self.has_drone: return self.drone.get_height()
        return 0