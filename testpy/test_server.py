import gslam
import cv2
import Viewer
import numpy as np
import websocket
from websocket import create_connection
import threading
import struct
import time

ip = "localhost:9002"
actor = "Actor1"
ws = create_connection("ws://%s/%s/controller/session"%(ip, actor))

reading = [0] * 61
def on_message(ws, message):
    global reading
    reading = list(struct.unpack("<%df"%(len(message)//4), message))
    
ws2 = websocket.WebSocketApp("ws://%s/%s/lidar/subscribe"%(ip, actor), on_message = on_message)
wst = threading.Thread(target=ws2.run_forever)
wst.daemon = True
wst.start()

max_dist = 500.0
scale = 10.
params = {'sensor_size': 61, 
    'start_angle': -30, 
    'end_angle': 210, 
    'max_dist': max_dist, 
    'velocity': scale, 
    'rotate_step': 6}


grid = gslam.GridMap([0.4, -0.4, 5.0, -5.0], 4)
pf = gslam.ParticleFilter((0,0,270), params, grid, 200)
traj = pf.getParticle(0).getTraj().copy()
print(traj)
traj_truth = [(120.0, 80.0, 180.0)]
cv2.imshow("trash", np.zeros((500,500)))

# Main Loop
while(1):
    # Input Control
    action = -1
    k = cv2.waitKey(1)
    if k==ord('w'):
        ws.send('forward')
        action = 1
    if k==ord('s'):
        ws.send('backward')
        action = 2
    if k==ord('a'):
        ws.send('left')
        action = 3
    if k==ord('d'): 
        ws.send('right')
        action = 4
    data_ready = False
    if action > 0:
        time.sleep(0.1)
        print(ws.recv())
        ws.send("is collided?")
        print(ws.recv())
        reading_temp = [scale*i for i in reading]
        scan = {'sensor_size': 61, 
                'start_angle': -30.0, 
                'end_angle': 210.0, 
                'max_dist': max_dist,
                'data': reading_temp}
        #print(reading_temp)
        neff = pf.feed(action, scan)
        bid = pf.bestSampleId()
        bp = pf.getParticle(bid)

        mm2 = bp.getMap().getWholeMapProb()
        mm2 = Viewer.Map2Image(mm2)
        cv2.imshow('map_p', mm2)

        if neff < 0.5:
            pf.resampling()
