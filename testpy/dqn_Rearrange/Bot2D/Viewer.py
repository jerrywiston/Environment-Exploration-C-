import numpy as np
import random
from Bot2D.utils import *
import cv2
import math

def Map2Image(m):
    img = (255*m).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def DrawEnv(img_map, scale=1.0):
    img = img_map.copy()
    img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
    img = Map2Image(img)
    return img

def DrawBot(img, bot_pos, sensor_data, bot_param, color=(0,255,0), scale=1.0):
    plist = EndPoint(bot_pos, bot_param, sensor_data)
    for pts in plist:
        cv2.line(
            img, 
            (int(scale*bot_pos[0]), int(scale*bot_pos[1])), 
            (int(scale*pts[0]), int(scale*pts[1])),
            color, 1)

    cv2.circle(img,(int(scale*bot_pos[0]), int(scale*bot_pos[1])), int(5*scale), (0,0,255), -1)
    return img

def DrawPath(img, path, color=(255,50,50), scale=1.0):
    for i in range(len(path)-1):
        cv2.line(
            img, 
            (int(scale*path[i][0]), int(scale*path[i][1])), 
            (int(scale*path[i+1][0]), int(scale*path[i+1][1])),
            color, 2)
    return img