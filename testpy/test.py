import gslam
import cv2
import Viewer
import numpy as np

params = {'sensor_size': 240, 
    'start_angle': -30, 
    'end_angle': 210, 
    'max_dist': 150, 
    'noise_nor': 0.1,
    'noise_tan': 0.0,
    'noise_ang':0.1}
pyparam = [params['sensor_size'], params['start_angle'], params['end_angle'], params['max_dist']]
im = cv2.imread('map_01.png', cv2.IMREAD_GRAYSCALE)
print(im)
bot = gslam.SingleBotLaser2DGrid((120.0, 80.0, 180.0), im, params)
#bot.pose = (140.0, 80.0)
pose = bot.pose
print(pose)
scan = bot.scan()
img = Viewer.DrawEnv(im, 1.0, pose, scan['data'], pyparam)
cv2.imshow('env', img)


grid = gslam.GridMap([0.4, -0.4, 5.0, -5.0], 2)
pf_size = 5
pf = gslam.ParticleFilter(pose, params, grid, pf_size)
traj = pf.getParticle(0).getTraj().copy()
print(traj)
traj_truth = [(120.0, 80.0, 180.0)]
timestemp = 0
pf.markParticles(0)
print("Rec: ")
for i in range(pf_size):
    print(pf.getParticle(i).getIdRecord())
# Main Loop
while(1):
    # Input Control
    action = -1
    action0 = -1
    action1 = -1
    k = cv2.waitKey(1)
    if k==ord('w'):
        action = 1
        action0 = 6
        action1 = 0
    if k==ord('s'):
        action = 1
        action0 = -6
        action1 = 0
    if k==ord('a'):
        action = 1
        action0 = 0
        action1 = -6
    if k==ord('d'): 
        action = 1
        action0 = 0
        action1 = 6
    if action > 0:
        timestemp += 1
        if not bot.action(action0, action1):
            continue
        scan = bot.scan()
        pose = bot.pose
        traj_truth.append(pose)
        neff = pf.feed(action0, action1, scan)
        bid = pf.bestSampleId()
        bp = pf.getParticle(bid)

        gain = Viewer.SensorMapping(grid, pose, scan['data'], pyparam)
        print(gain)
        mmmm = grid.getObserv((pose[0], pose[1]), pose[2], 32, 32)
        mmmm = Viewer.Map2Image(mmmm)
        cv2.imshow('mmmm', mmmm)
        img = Viewer.DrawEnv(im, 1.0, pose, scan['data'], pyparam)
        
        for i in range(pf_size):
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
            pf.markParticles(timestemp)
            pf.resampling()
            print("Rec: ")
            for i in range(pf_size):
                print(pf.getParticle(i).getIdRecord())
        
