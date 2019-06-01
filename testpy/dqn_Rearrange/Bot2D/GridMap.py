import numpy as np
from Bot2D.utils import *
import cv2

class GridMap:
    def __init__(self, map_param, gsize=1.0):
        self.map_param = map_param
        self.gmap = {}
        self.gsize = gsize
        self.boundary = [9999,-9999,9999,-9999]

    def getObs(self, pos, lx, ly):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        ang = pos[2] + 90
        obs = np.zeros((2*ly,2*lx))
        coord = np.zeros((2*ly,2*lx,2))
        idx = 0
        for i in range(-lx,lx):
            idy = 0
            for j in range(-ly,ly):
                coord[idy, idx, 0] = i
                coord[idy, idx, 1] = j
                idy += 1
            idx += 1
        rx = x + coord[:,:,0]*np.cos(np.deg2rad(ang)) - coord[:,:,1]*np.sin(np.deg2rad(ang))
        ry = y + coord[:,:,0]*np.sin(np.deg2rad(ang)) +  coord[:,:,1]*np.cos(np.deg2rad(ang))
        
        idx = 0
        for i in range(-lx,lx):
            idy = 0
            for j in range(-ly,ly):
                obs[idy, idx] = self.GetGridProb((int(round(rx[idy, idx])), int(round(ry[idy, idx]))))
                idy += 1
            idx += 1
        
        return obs

    def GetGridProb(self, pos):
        if pos in self.gmap:
            return np.exp(self.gmap[pos]) / (1.0 + np.exp(self.gmap[pos]))
        else:
            return 0.5

    def GetCoordProb(self, pos):
        x, y = int(round(pos[0]/self.gsize)), int(round(pos[1]/self.gsize))
        return self.GetGridProb((x,y))

    def GetMapProb(self, x0, x1, y0, y1):
        map_prob = np.zeros((y1-y0, x1-x0))
        idx = 0
        for i in range(x0, x1):
            idy = 0
            for j in range(y0, y1):
                map_prob[idy, idx] = self.GetGridProb((i,j))
                idy += 1
            idx += 1
        return map_prob

    def GridMapLine(self, x0, x1, y0, y1, hit=True):
        info_gain = 0

        # Scale the position
        x0, x1 = int(round(x0/self.gsize)), int(round(x1/self.gsize))
        y0, y1 = int(round(y0/self.gsize)), int(round(y1/self.gsize))

        rec = Bresenham(x0, x1, y0, y1)
        for i in range(len(rec)):
            p= self.GetGridProb(rec[i])
            ent_last = p*np.log2(p) + (1-p)*np.log2(1-p)

            if i < len(rec)-4:
                change = self.map_param[1]
            elif hit==True:
                change = self.map_param[0]
            else:
                break

            if rec[i] in self.gmap:
                self.gmap[rec[i]] += change
            else:
                self.gmap[rec[i]] = change
                if rec[i][0] < self.boundary[0]:
                    self.boundary[0] = rec[i][0]
                elif rec[i][0] > self.boundary[1]:
                    self.boundary[1] = rec[i][0]
                if rec[i][1] < self.boundary[2]:
                    self.boundary[2] = rec[i][1]
                elif rec[i][1] > self.boundary[3]:
                    self.boundary[3] = rec[i][1]

            if self.gmap[rec[i]] > self.map_param[2]:
                self.gmap[rec[i]] = self.map_param[2]
            if self.gmap[rec[i]] < self.map_param[3]:
                self.gmap[rec[i]] = self.map_param[3]
            
            p= self.GetGridProb(rec[i])
            ent_new = p*np.log2(p) + (1-p)*np.log2(1-p)
            info_gain += (ent_new - ent_last)
        return info_gain
    
    def MapEntropy(self,x0,x1,y0,y1):
        m = self.GetMapProb(x0,x1,y0,y1)
        ent = 0
        for i in range(m.shape[1]):
            for j in range(m.shape[0]):
                p = m[j,i]
                ent += p*np.log2(p) + (1-p)*np.log2(1-p)
        return ent

def AdaptiveGetMap(gmap):
    mimg = gmap.GetMapProb(
        gmap.boundary[0]-20, gmap.boundary[1]+20, 
        gmap.boundary[2]-20, gmap.boundary[3]+20 )
    #mimg = gmap.GetMapProb(0,500,0,500)
    mimg = (255*mimg).astype(np.uint8)
    mimg = cv2.cvtColor(mimg, cv2.COLOR_GRAY2RGB)
    return mimg

def SensorMapping(m, bot_pos, bot_param, sensor_data, gsize=4):
    inter = (bot_param[2] - bot_param[1]) / (bot_param[0]-1)
    info_gain = 0
    for i in range(bot_param[0]):
        sensor_data_temp = sensor_data[i] + 2*gsize
        hit = True
        if sensor_data_temp < 20:
            continue
        if sensor_data_temp > bot_param[3]-3:
            hit = False


        theta = bot_pos[2] + bot_param[1] + i*inter
        info_gain += m.GridMapLine(
        int(bot_pos[0]), 
        int(bot_pos[0]+(sensor_data_temp+2)*np.cos(np.deg2rad(theta-90))),
        int(bot_pos[1]),
        int(bot_pos[1]+(sensor_data_temp+2)*np.sin(np.deg2rad(theta-90))),
        hit = hit
        )
    return info_gain

if __name__ == '__main__':
    #lo_occ, lo_free, lo_max, lo_min
    map_param = [0.9, -0.7, 5.0, -5.0]
    m = GridMap(map_param)
    pos = (0.0,0.0)
    m.gmap[pos] = 0.1
    print(m.GetProb(pos))
    print(m.GetProb((0,0)))
