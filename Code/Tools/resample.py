import numpy as np
import scipy.interpolate as interpolate
# from .file_io import get_data
from tqdm import tqdm
import os
# from .file_io import *
def resample_mean2(points,target_points=-1,target_step_size=0.0):
    # calculate distances between adjacent points
    # distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    # print("old:"+str(np.mean(distances)))
    # define target step size
    if target_points==-1:
        distance = np.sum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        if target_step_size==0: 
            # print("old:"+str(np.mean(distances)))
            target_step_size = distance/4
    # create B-spline curve with evenly spaced knots
    
    num_points = len(points)
    knots = np.linspace(0, 1, num_points)
    spline = interpolate.make_interp_spline(knots, points, k=2)
    if target_points!=-1:
        new_num = target_points
    else:
        new_num = round(distance/target_step_size)+1
    t=np.linspace(0, 1, new_num)
    new_points = []
    for t1 in t:
        # evaluate spline
        p = spline(t1)
        new_points.append(p)
    new_points = np.array(new_points)
    return new_points
def resample(strands,segments):
    new_strands = np.array([[0,0,0]])
    new_segments = []
    start=0
    for i in range(0,len(segments)):
        new_strand = resample_mean2(strands[start:start+segments[i]], 100)
        distances = np.sqrt(np.sum(np.diff(new_strand, axis=0) ** 2, axis=1))          
        # print("new:"+str(np.mean(distances)))
        start+=segments[i]
        # draw_3d(strands[i],new_strand)
        new_strands=np.append(new_strands,new_strand,axis=0)
        new_segments.append(new_strand.shape[0])
    return new_strands[1:],new_segments