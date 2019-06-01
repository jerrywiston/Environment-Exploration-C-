import numpy as np
import cv2
from Bot2D.SingleBotLaser2Dgrid import *
from Bot2D.GridMap import *
from Bot2D.MotionModel import *
from Bot2D.Viewer import *
import copy

class Bot2DEnv:
    def __init__(self, 
        # SensorSize, StartAngle, EndAngle, MaxDist, Velocity, Angular
        bot_param = [60, -30.0, 210.0, 160.0, 6.0, 10.0],
        # lo_occ, lo_free, lo_max, lo_min
        map_param = [-0.5, 0.5, 2.0, -2.0],
        # NormalVar, TangentVar, AngularVar
        motion_param = [0.2, 0.1, 0.1], # Simple Model
        #motion_param = [0.01, 0.01, 0.2, 0.2, 0.01, 0.01], # Velocity Model
        obs_size = 64,
        grid_size = 2.,
        goal_dist = 20,
        map_path = 'Image/map.png'
    ):
        self.bot_param = bot_param
        self.map_param = map_param
        self.motion = SimpleMotionModel(motion_param[0], motion_param[1], motion_param[2])
        self.obs_size = obs_size
        self.grid_size = grid_size
        self.goal_dist = goal_dist
        self.map_path = map_path
    
    def reset(self):
        self.env = SingleBotLaser2D([0,0,0], self.bot_param, self.map_path, self.motion)
        self.bot_pos = self.env.RandomPos()
        self.nav_pos = self.env.SearchPos(min_dist=self.goal_dist)
        self.map = GridMap(self.map_param, gsize=self.grid_size)
        self.sensor_data = self.env.Sensor()
        SensorMapping(self.map, self.env.bot_pos, self.bot_param, self.sensor_data)
        fsize = self.obs_size
        self.obs = self.map.getObs(self.env.bot_pos,int(fsize/2),int(fsize/2)).reshape([fsize,fsize,1])
        self.dist = np.abs(self.env.bot_pos[0] - self.nav_pos[0]) + np.abs(self.env.bot_pos[1] - self.nav_pos[1])
        sdata = np.array(self.sensor_data, dtype=np.float32)/self.bot_param[3]
        return {"map":self.obs, "sensor": sdata, "goal":self.getRelPos()}

    def step(self, action):
        action = action
        collision = self.env.BotAction(action)
        fsize = self.obs_size
        #self.obs = self.map.getObs(self.env.bot_pos,int(fsize/2),int(fsize/2)).reshape([fsize,fsize,1])
        self.sensor_data = self.env.Sensor()
        #info_gain = SensorMapping(self.map, self.env.bot_pos, self.bot_param, self.sensor_data)
        sdata = np.array(self.sensor_data, dtype=np.float32)/self.bot_param[3]
        #now_dist = np.sqrt(np.square(self.env.bot_pos[0] - self.nav_pos[0]) + np.square(self.env.bot_pos[1] - self.nav_pos[1]))
        now_dist = np.abs(self.env.bot_pos[0] - self.nav_pos[0]) + np.abs(self.env.bot_pos[1] - self.nav_pos[1])
        rel_pos = self.getRelPos()
        
        # Reward Calculate
        reward = (self.dist - now_dist)
        done = 1.
        if np.linalg.norm(rel_pos) < self.goal_dist:
            reward += 200
            done = 0.
        if collision:
            reward += -200
            done = 0.
        
        self.dist = now_dist
        return {"map":self.obs, "sensor":sdata, "goal":self.getRelPos()}, reward, done, 

    def getRelPos(self):
        p_ = np.array([[self.nav_pos[0]], [self.nav_pos[1]]])
        p = np.array([[self.env.bot_pos[0]], [self.env.bot_pos[1]]])
        d = -np.deg2rad(self.env.bot_pos[2])
        pos = np.matmul(np.array([[np.cos(d),-np.sin(d)],[np.sin(d),np.cos(d)]]), (p_ - p))
        return np.reshape(pos, 2)

    def render(self):
        # Initialize OpenCV Windows
        cv2.namedWindow('env', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('map', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('obs', cv2.WINDOW_AUTOSIZE)

        # Get Image
        img = DrawEnv(self.env.img_map)
        img = cv2.circle(img,(int(self.nav_pos[0]), int(self.nav_pos[1])), self.goal_dist-3, (255,0,255), 6)
        img = DrawPath(img, self.env.path)
        img = DrawBot(img, self.env.bot_pos, self.sensor_data, self.bot_param)
        mimg = AdaptiveGetMap(self.map)
        fsize = self.obs_size
        
        obs_re = cv2.resize(self.obs,(int(self.obs.shape[1]*3),int(self.obs.shape[0]*3)),interpolation=cv2.INTER_CUBIC)
        img_re = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)),interpolation=cv2.INTER_CUBIC)
        mimg_re = cv2.resize(mimg,(int(mimg.shape[1]*1),int(mimg.shape[0]*1)),interpolation=cv2.INTER_CUBIC)

        cv2.imshow('env',img_re)
        #cv2.imshow('map',mimg_re)
        #cv2.imshow('obs',obs_re)
        cv2.waitKey(1)

    def Image2Map(self, map_path):
        im = cv2.imread(map_path)
        m = np.asarray(im)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        return m
