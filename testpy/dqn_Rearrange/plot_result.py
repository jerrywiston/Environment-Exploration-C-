import json
import matplotlib.pyplot as plt
import numpy as np

def errorfill(x, y, yerr, color='r', alpha_fill=0.3, ax=None):
    plt.figure()
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def TotalReward(rec):
    return np.array(rec).sum()

def DescountReward(rec, reward_decay=0.9):
    temp = 0
    for i in range(len(rec)):
        temp = rec[len(rec)-1-i] + reward_decay*temp
    return temp

def SmoothReward(rec, smooth=20):
    rlist = []
    vlist = []
    for i in range(len(rec)-smooth):
        temp = np.asarray(rec[i:i+smooth])
        rlist.append(np.mean(temp))
        vlist.append(np.std(temp))
    return rlist, vlist

def PlotNavAchieve(fname, inter=20):
    f = open(fname, "r")
    rec = json.load(f)
    rlist = []
    for i in range(len(rec)-inter):
        count = 0.0
        for j in range(i,i+inter):
            if rec[j][-1] >= 19:
                count += 1.0
        rlist.append(count/inter)
    plt.figure()
    plt.plot(rlist, 'g')
        

def PlotResult(fname, smooth=20, color='b'):
    f = open(fname, "r")
    rec = json.load(f)

    r1list = []
    r2list = []
    for i in range(len(rec)):
        r1list.append(DescountReward(rec[i]))
        r2list.append(TotalReward(rec[i]))
    s1list, v1list = SmoothReward(r1list, smooth)
    s2list, v2list = SmoothReward(r2list, smooth)
    #errorfill(range(len(s1list)), np.array(s1list), np.array(v1list), color="r")
    errorfill(range(len(s2list)), np.array(s2list), np.array(v2list), color=color)

json_name = "OX.json"
PlotResult(json_name, smooth=20)
PlotNavAchieve(json_name, inter=20)

plt.show()