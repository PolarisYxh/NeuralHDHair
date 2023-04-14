from Tools.render import render
from Tools.render_strand import render_seg_strand
import trimesh
import os
import cv2
import struct
import numpy as np

os.environ['PYOPENGL_PLATFORM'] = 'osmesa' 
def transform(points):
    '''

    :param points: 原始点云
    :return: 体素化的点云
    '''
    mul=1
    stepInv = 1. / (0.00567194/mul)
    gridOrg= np.array([-0.3700396, 1.22352, -0.261034], dtype=np.float32)

    points -= gridOrg
    points *= np.array([1., -1., -1.], dtype=np.float32) * stepInv  #opengl中xyz坐标轴与python中不一样，此处为一个调整，可以不管，/stepInv  step就是每个体素的边长
    points += np.array([0, 128*mul, 96*mul], dtype=np.float32)   #使所有点坐标的值落在[0-96,0-128,0-128]之间

    # points = np.maximum(points,
    #                     np.array([0, 0, 0], dtype=np.float32))  # note that voxels out of boundaries are minus
    # points = np.minimum(points, np.array([127.9, 127.9, 95.9], dtype=np.float32))



    return points
def load_strand(d,trans=True):
    file = os.path.join(d, "hair_delete.hair").replace("\\", "/")
    with open(file, mode='rb')as f:
        num_strand = f.read(4)
        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)
        (point_count,) = struct.unpack('I', point_count)

        # print("num_strand:",num_strand)
        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)
        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)
    f.close()
    points=list(points)
    # points=[[points[i*3+0],points[i*3+1],points[i*3+2]] for i in range(len(points)//3)]
    points=np.array(points)
    points=np.reshape(points,(-1,3))
    if trans:
        points=transform(points)

    return segments,points
if __name__=="__main__":
    segments,strand = load_strand("data/Train_input/DB1",trans=False)

    img = render_seg_strand(segments,strand)[0]
    cv2.imwrite("1.png",img)
    # cv2.waitKey()
    # bodymesh = trimesh.load(os.path.join(os.path.dirname(__file__),'female_halfbody_medium.obj'))
    # img = render(bodymesh,out_img[0,:,:,:,:],isshow=True)
    