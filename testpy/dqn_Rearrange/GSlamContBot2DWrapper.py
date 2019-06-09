import numpy as np
import cv2
import copy
import gslam
from Bot2D.Viewer import *

params = {
    'sensor_size': 60, 
    'start_angle': -30, 
    'end_angle': 210, 
    'max_dist': 160, 
    'velocity': 6, 
    'rotate_step': 10}

def EndPoint(pos, bot_param, sensor_data, gsize=4):
    pts_list = []
    inter = (bot_param['end_angle'] - bot_param['start_angle']) / (bot_param['sensor_size']-1)
    for i in range(bot_param['sensor_size']):
        sensor_data_temp = sensor_data[i] + 2*gsize
        theta = pos[2] + bot_param['start_angle'] + i*inter
        pts_list.append(
            ( pos[0]+sensor_data_temp*np.cos(np.deg2rad(theta - 90)),
              pos[1]+sensor_data_temp*np.sin(np.deg2rad(theta - 90))) )
    return pts_list

def SensorMapping(gmap, bot_pos, bot_param, sensor_data, gsize=4):
    info_gain = 0
    plist = EndPoint(bot_pos, bot_param, sensor_data, gsize)
    for pts, dist in zip(plist, sensor_data):
        # Don't map points that are larger than certain distance.
        if dist < bot_param['max_dist'] - 2:
            info_gain += gmap.line((bot_pos[0], bot_pos[1]), pts, True)
        else:
            info_gain += gmap.line((bot_pos[0], bot_pos[1]), pts, False)
    return info_gain

################################################################

class Bot2DEnv:
    def __init__(self, 
        # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
        bot_param = params,
        # lo_occ, lo_free, lo_max, lo_min
        map_param = [0.4, -0.4, 5.0, -5.0],
        # NormalVar, TangentVar, AngularVar
        obs_size = 64,
        grid_size = 2.,
        goal_dist = 20,
        map_path = 'Image/map.png',
        task = 'Navigation'
    ):
        self.bot_param = bot_param
        self.bot_param_list = [bot_param['sensor_size'], bot_param['start_angle'], bot_param['end_angle'], bot_param['max_dist'], bot_param['velocity'], bot_param['rotate_step']]
        self.map_param = map_param
        self.obs_size = obs_size
        self.grid_size = grid_size
        self.goal_dist = goal_dist
        self.map_path = map_path
        self.task = task
    
    def reset(self):
        self.img_map = cv2.imread(self.map_path, cv2.IMREAD_GRAYSCALE)
        self.env = gslam.SingleBotLaser2DGrid((0,0,0), self.img_map, self.bot_param)
        temp = self.SearchPos(self.goal_dist, self.img_map)
        self.env.setPose(temp)
        self.traj = [temp]
        
        self.path = [self.env.pose]
        self.nav_pos = self.SearchPos(self.goal_dist, self.img_map)
        self.map = gslam.GridMap(self.map_param, gsize=self.grid_size)

        self.sensor_data = self.env.scan()
        info_gain = SensorMapping(self.map, self.env.pose, self.bot_param, self.sensor_data['data'], self.grid_size)

        fsize = self.obs_size
        self.obs = self.map.getObserv(self.env.pose[0:2],self.env.pose[2],int(fsize/2),int(fsize/2)).reshape([fsize,fsize,1])
        self.dist = np.abs(self.env.pose[0] - self.nav_pos[0]) + np.abs(self.env.pose[1] - self.nav_pos[1])
        sdata = np.array(self.sensor_data['data'], dtype=np.float32)/self.bot_param['max_dist']
        return {"map":self.obs, "sensor": sdata, "goal":self.getRelPos(), "info_gain":info_gain}

    def SearchPos(self, min_dist, img_map):
        x = img_map.shape[1]
        y = img_map.shape[0]
        bot_pos = None
        while(1):
            done = True
            bot_pos = np.array([np.random.randint(x), np.random.randint(y), np.random.randint(360)])
            if img_map[bot_pos[1], bot_pos[0]] < 0.9:
                done = False
            for i in range(120):
                if self.env.rayCast((float(bot_pos[0]), float(bot_pos[1]), float(3*i))) < min_dist:
                    done = False
                    break
            if done:
                break
        return bot_pos

    def step(self, action):
        action0 = 6*(action[0]+1)
        action1 = 10*action[1]

        collision = not self.env.continuous_action(action0, action1)
        self.traj.append(self.env.pose)
        fsize = self.obs_size
        self.obs = self.map.getObserv((self.env.pose[0], self.env.pose[1]), self.env.pose[2] ,int(fsize/2),int(fsize/2)).reshape([fsize,fsize,1])
        self.sensor_data = self.env.scan()
        info_gain = SensorMapping(self.map, self.env.pose, self.bot_param, self.sensor_data['data'], self.grid_size)
        sdata = np.array(self.sensor_data['data'], dtype=np.float32)/self.bot_param['max_dist']
        
        #now_dist = np.sqrt(np.square(self.env.bot_pos[0] - self.nav_pos[0]) + np.square(self.env.bot_pos[1] - self.nav_pos[1]))
        now_dist = np.abs(self.env.pose[0] - self.nav_pos[0]) + np.abs(self.env.pose[1] - self.nav_pos[1])
        
        rel_pos = self.getRelPos()
        
        # Reward Calculate
        done = 1.
        if self.task == 'Navigation':
            reward = (self.dist - now_dist)
            
            if np.linalg.norm(rel_pos) < self.goal_dist:
                reward += 200
                done = 0.
            if collision:
                reward += -200
                done = 0.
        elif self.task == 'Exploration':
            reward = info_gain
            if collision:
                reward += -200
                done = 0.
        
        self.dist = now_dist
        return {"map":self.obs, "sensor":sdata, "goal":self.getRelPos(), "info_gain":info_gain}, reward, done, 

    def getRelPos(self):
        p_ = np.array([[self.nav_pos[0]], [self.nav_pos[1]]])
        p = np.array([[self.env.pose[0]], [self.env.pose[1]]])
        d = -np.deg2rad(self.env.pose[2])
        pos = np.matmul(np.array([[np.cos(d),-np.sin(d)],[np.sin(d),np.cos(d)]]), (p_ - p))
        return np.reshape(pos, 2)

    def render(self):
        # Initialize OpenCV Windows
        cv2.namedWindow('env', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('obs', cv2.WINDOW_AUTOSIZE)

        # Get Image
        img = DrawEnv(self.img_map / 255.)
        if self.task == 'Navigation':
            img = cv2.circle(img,(int(self.nav_pos[0]), int(self.nav_pos[1])), self.goal_dist-3, (255,0,255), 6)
        img = DrawPath(img, self.traj)
        img = DrawBot(img, self.env.pose, self.sensor_data['data'], self.bot_param_list)
        mimg = self.map.getWholeMapProb()
        mimg = self.Map2Image(mimg)
        fsize = self.obs_size
        
        obs_re = cv2.resize(self.obs,(int(self.obs.shape[1]*3),int(self.obs.shape[0]*3)),interpolation=cv2.INTER_CUBIC)
        img_re = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)),interpolation=cv2.INTER_CUBIC)
        mimg_re = cv2.resize(mimg,(int(mimg.shape[1]*1),int(mimg.shape[0]*1)),interpolation=cv2.INTER_CUBIC)

        cv2.imshow('env',img_re)
        cv2.imshow('map',mimg_re)
        cv2.imshow('obs',obs_re)
        cv2.waitKey(1)

    def Image2Map(self, map_path):
        im = cv2.imread(map_path)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m
    
    def Map2Image(self, m):
        img = (255*m).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
