import gslam
import cv2
import Viewer
import numpy as np

params = {'sensor_size': 240, 
    'start_angle': -30, 
    'end_angle': 210, 
    'max_dist': 150, 
    'velocity': 6, 
    'rotate_step': 6}
pyparam = [params['sensor_size'], params['start_angle'], params['end_angle'], params['max_dist'], params['velocity'], params['rotate_step']]
im = cv2.imread('../build/map3.png', cv2.IMREAD_GRAYSCALE)
print(im)
bot = gslam.SingleBotLaser2DGrid((120.0, 80.0, 180.0), im, params)
#bot.pose = (140.0, 80.0)
pose = bot.pose
print(pose)
scan = bot.scan()
img = Viewer.DrawEnv(im, 1.0, pose, scan['data'], pyparam)
cv2.imshow('env', img)


grid = gslam.GridMap([0.4, -0.4, 5.0, -5.0], 2)
pf = gslam.ParticleFilter(pose, params, grid, 100)
traj = pf.getParticle(0).getTraj().copy()
print(traj)
traj_truth = [(120.0, 80.0, 180.0)]

# Main Loop
while(1):
    # Input Control
    action = -1
    k = cv2.waitKey(1)
    if k==ord('w'):
        action = 1
    if k==ord('s'):
        action = 2
    if k==ord('a'):
        action = 3
    if k==ord('d'): 
        action = 4 
    
    if k==ord('i'):
        action = 5
    if k==ord('j'):
        action = 6
    if k==ord('l'):
        action = 7
    if k==ord('k'):
        action = 8

    if action > 0:
        if not bot.action(action):
            continue
        scan = bot.scan()
        pose = bot.pose
        traj_truth.append(pose)
        neff = pf.feed(action, scan)
        bid = pf.bestSampleId()
        bp = pf.getParticle(bid)

        gain = Viewer.SensorMapping(grid, pose, scan['data'], pyparam)
        print(gain)
        mmmm = grid.getObserv((pose[0], pose[1]), pose[2], 32, 32)
        mmmm = Viewer.Map2Image(mmmm)
        cv2.imshow('mmmm', mmmm)
        img = Viewer.DrawEnv(im, 1.0, pose, scan['data'], pyparam)
        
        for i in range(100):
            if i != bid:
                traj = pf.getParticle(i).getTraj()
                #print(traj)
                img = Viewer.DrawTraj(img, 1, traj, (0, 255, 0))

        img = Viewer.DrawTraj(img, 1, bp.getTraj(), (0, 255, 0))
        img = Viewer.DrawTraj(img, 1, traj_truth, (255, 0, 0))
        cv2.imshow('env', img)
        mm = grid.getWholeMapProb()
        mm = Viewer.Map2Image(mm)
        mm2 = bp.getMap().getWholeMapProb()
        mm2 = Viewer.Map2Image(mm2)
        cv2.imshow('map', mm)
        cv2.imshow('map_p', mm2)

        if neff < 0.5:
            pf.resampling()
