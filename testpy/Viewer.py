import numpy as np
import random
import utils
import cv2
import math
def Map2Image(m):
    img = (255*m).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def DrawEnv(img_map, scale, bot_pos, sensor_data, bot_param):
    img = img_map.copy()
    img = cv2.resize(img, (round(scale*img.shape[1]), round(scale*img.shape[0])), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    for pts in plist:
        cv2.line(
            img, 
            (int(scale*bot_pos[0]), int(scale*bot_pos[1])), 
            (int(scale*pts[0]), int(scale*pts[1])),
            (255,0,0), 1)
    cv2.circle(img,(int(scale*bot_pos[0]), int(scale*bot_pos[1])), int(5*scale), (0,0,255), -1)
    return img
def DrawTraj(img_map, scale, path, color, copy=False):
    img = img_map
    if copy:
        img = img_map.copy()
    for i in range(len(path)-1):
        cv2.line(
            img, 
            (int(scale*path[i][0]), int(scale*path[i][1])), 
            (int(scale*path[i+1][0]), int(scale*path[i+1][1])),
            color, 2)
        #cv2.circle(img,(int(scale*pose[0]), int(scale*pose[1])), int(1*scale), color, -1)
    return img

def SensorMapping(gmap, bot_pos, sensor_data, bot_param):
    info_gain = 0
    plist = utils.EndPoint(bot_pos, bot_param, sensor_data)
    for pts, dist in zip(plist, sensor_data):
        # Don't map points that are larger than certain distance.
        if dist < bot_param[3] - 2:
            info_gain += gmap.line((bot_pos[0], bot_pos[1]), pts)
    return info_gain

def DrawParticle(img, plist, scale=1.0):
    for p in plist:
        cv2.circle(img,(int(scale*p.pos[0]), int(scale*p.pos[1])), int(2), (0,200,0), -1)
    return img

def DrawPath(img, path, color=(50,200,50), scale=1.0):
    for i in range(len(path)-1):
        cv2.line(
            img, 
            (int(scale*path[i][0]), int(scale*path[i][1])), 
            (int(scale*path[i+1][0]), int(scale*path[i+1][1])),
            color, 2)
    return img

def DrawAlign(Xc, Pc, R, T):
    if Xc.shape[0] < 20 or Pc.shape[0] < 20:
        return 255*np.ones((100,100), np.uint8)
    shift = 10.0
    Xc_ = Icp2d.Transform(Xc, R, T)
    max_x = np.max(np.array([np.max(Xc_[:,0]), np.max(Pc[:,0])]) )
    max_y = np.max(np.array([np.max(Xc_[:,1]), np.max(Pc[:,1])]) )
    min_x = np.min(np.array([np.min(Xc_[:,0]), np.min(Pc[:,0])]) )
    min_y = np.min(np.array([np.min(Xc_[:,1]), np.min(Pc[:,1])]) )
    img = 255*np.ones((int(max_y-min_y+2*shift),int(max_x-min_x+2*shift),3), np.uint8)
    for i in range(Xc_.shape[0]):
        cv2.circle(img, (int(Xc[i,0]-min_x+shift), int(Xc[i,1]-min_y+shift)), int(2), (200,200,200), -1)
    for i in range(Xc.shape[0]):
        cv2.circle(img, (int(Xc_[i,0]-min_x+shift), int(Xc_[i,1]-min_y+shift)), int(2), (0,0,255), -1)
    for i in range(Pc.shape[0]):
        cv2.circle(img, (int(Pc[i,0]-min_x+shift), int(Pc[i,1]-min_y+shift)), int(2), (255,0,0), -1)
    return img
