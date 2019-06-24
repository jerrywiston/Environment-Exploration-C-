import numpy as np
import random
import cv2
import math
import copy
import matplotlib.pyplot as plt
from Bot2D.MotionModel import *
from Bot2D.utils import *

class SingleBotLaser2D:
    def __init__(self, bot_pos, bot_param, fname, motion):
        self.bot_pos = bot_pos
        self.bot_param = bot_param
        self.img_map = self.Image2Map(fname)
        self.motion = motion
        self.path = [bot_pos]

        scale = 1
        img = self.Image2Map(fname)
        img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
        self.img_map = img

    def SearchPos(self, min_dist):
        x = self.img_map.shape[1]
        y = self.img_map.shape[0]
        bot_pos = None
        while(1):
            done = True
            bot_pos = np.array([np.random.randint(x), np.random.randint(y), np.random.randint(360)])
            if self.img_map[bot_pos[1], bot_pos[0]] < 0.9:
                done = False
            for i in range(120):
                if self.RayCast(np.array((bot_pos[0], bot_pos[1])), 3*i) < min_dist:
                    done = False
                    break
            if done:
                break
        return bot_pos
    
    def RandomPos(self, min_dist=30):
        self.bot_pos = self.SearchPos(min_dist=min_dist)
        self.path = [self.bot_pos]
        return self.bot_pos

    def BotAction(self, action):
        if action == 0:
            pos_new = self.motion.Sample(self.bot_pos, self.bot_param[4], 0, 0)
        if action == 1:
            pos_new = self.motion.Sample(self.bot_pos, 0, 0, -self.bot_param[5])
        if action == 2:  
            pos_new = self.motion.Sample(self.bot_pos, 0, 0, self.bot_param[5])

        # Judge if collision
        collision = self.Collision(pos_new)
        if collision == False:
            self.bot_pos = pos_new

        self.path.append(self.bot_pos)
        return collision
    
    def Collision(self, pos_new):
        x0, x1 = int(round(self.bot_pos[0])), int(round(pos_new[0]))
        y0, y1 = int(round(self.bot_pos[1])), int(round(pos_new[1]))
        rec = Bresenham(x0, x1, y0, y1)
        rec.append((x1,y1))
        
        collision = False
        for i in range(len(rec)):
            if rec[i][1] >= self.img_map.shape[0] or rec[i][0] >= self.img_map.shape[1]:
                collision = True
            elif self.img_map[rec[i][1], rec[i][0]] < 0.5:
                #print("Detect:",(rec[i][1], rec[i][0]))
                collision = True
        return collision

    def Sensor(self):
        sense_data = []
        inter = (self.bot_param[2] - self.bot_param[1]) / (self.bot_param[0]-1)
        for i in range(self.bot_param[0]):
            theta = self.bot_pos[2] + self.bot_param[1] + i*inter
            sense_data.append(self.RayCast(np.array((self.bot_pos[0], self.bot_pos[1])), theta)+np.random.randn())
        return sense_data

    def RayCast(self, pos, theta):
        end = np.array((pos[0] + self.bot_param[3]*np.cos(np.deg2rad(theta-90)), pos[1] + self.bot_param[3]*np.sin(np.deg2rad(theta-90))))

        x0, y0 = int(pos[0]), int(pos[1])
        x1, y1 = int(end[0]), int(end[1])
        plist = Bresenham(x0, x1, y0, y1)
        i = 0
        dist = self.bot_param[3]
        for p in plist:
            if p[1] >= self.img_map.shape[0] or p[0] >= self.img_map.shape[1] or p[1]<0 or p[0]<0:
                continue
            if self.img_map[p[1], p[0]] < 0.5:
                tmp = math.pow(float(p[0]) - pos[0], 2) + math.pow(float(p[1]) - pos[1], 2)
                tmp = math.sqrt(tmp)
                if tmp < dist:
                    dist = tmp
        return dist

    def Image2Map(self, fname):
        im = cv2.imread(fname)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m

def SensorData2PointCloud(sensor_data, bot_pos, bot_param):
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    tmp = []
    for i in range(len(sensor_data)):
        if sensor_data[i] > bot_param[3]-1 or sensor_data[i] < 1:
            continue
        tmp.append(plist[i])
    tmp = np.array(tmp)
    return tmp
 
def Rotation2Deg(R):
    cos = R[0,0]
    sin = R[1,0]
    theta = np.rad2deg(np.arccos(np.abs(cos)))
    
    if cos>0 and sin>0:
        return theta
    elif cos<0 and sin>0:
        return 180-theta
    elif cos<0 and sin<0:
        return 180+theta
    elif cos>0 and sin<0:
        return 360-theta
    elif cos==0 and sin>0:
        return 90.0
    elif cos==0 and sin<0:
        return 270.0
    elif cos>0 and sin==0:
        return 0.0
    elif cos<0 and sin==0:
        return 180.0

