# from Train.src.preprocessing import gen_RT_matrix, get_rendered_convdata,get_rendered_convdata_from_data, gen_vis_weight, gasuss_noise
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.set_loglevel("notset") 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import trimesh
import os
from skimage import transform
def show3Dhair(axis, strands, mask):
    """
    strands: [32, 32, 300]
    mask: [32, 32] bool
    """
    # strands = strands.T.reshape(-1,300)
    # mask = mask.T.reshape(-1)
    for i in range(1024):
        for j in range(0, 100-1):
            if mask[i][j]==10:
                # transform from graphics coordinate to math coordinate
                x,y,z = [strands[i,j:j+2][:,k] for k in range(3)]
                axis.plot(x, y, z, linewidth=0.2, color='lightskyblue')

    RADIUS = 0.3  # space around the head
    xroot, yroot, zroot = 0, 0, 1.65
    axis.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    axis.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    axis.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])

    # Get rid of the ticks and tick labels
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])

    axis.get_xaxis().set_ticklabels([])
    axis.get_yaxis().set_ticklabels([])
    axis.set_zticklabels([])
    axis.set_aspect('auto')

def render(body_mesh,convdata,current_RT_mat=np.identity(4), isshow=False):
    # convdata = get_rendered_convdata_from_data(convdata, current_RT_mat)
    position = np.hstack((body_mesh.vertices, np.ones(body_mesh.vertices.shape[0]).reshape(-1,1)))
    position = np.dot(position,current_RT_mat).reshape(-1,4)
    position[:,0] = position[:,0]/position[:,3]
    position[:,1] = position[:,1]/position[:,3]
    position[:,2] = position[:,2]/position[:,3]
    body_mesh.vertices = position[:,:3]
    # body_mesh.apply_transform(current_RT_mat.transpose())

    pos = convdata[:,0:3,...]
    cur = convdata[:,3,...]

    strands = pos.T
    strands = strands.swapaxes(2,3).reshape((strands.shape[0]*strands.shape[1],-1,3))
    # mask = current_visweight.T
    # mask = mask.reshape((mask.shape[0]*mask.shape[1],-1))
    from strandhair.render_strand import render_strand


    img = render_strand(strands,body_mesh)[0]
    if isshow:
        cv2.imshow("1",img)
        cv2.waitKey()
    return img
    
# if __name__=="__main__":  
#     current_RT_mat = gen_RT_matrix("/home/yxh/Documents/HairNet_DataSetGeneration/Train/test/strands00002_00004_10000_v0.txt")
#     # current_RT_mat = np.identity(4)
#     current_convdata_path = "/home/yxh/Documents/HairNet_DataSetGeneration/Train/convdata/strands00002_00004_10000.convdata"
#     # current_convdata = get_rendered_convdata(current_convdata_path, current_RT_mat)
#     current_convdata = np.load(current_convdata_path).reshape(100, 4, 32, 32)
#     mask_path = "/home/yxh/Documents/HairNet_DataSetGeneration/Train/test/strands00002_00004_10000_v0.vismap"
#     current_visweight = gen_vis_weight(mask_path)

#     rgb_image = cv2.imread("/home/yxh/Documents/HairNet_DataSetGeneration/Train/test/strands00002_00004_10000_v0.png")
#     # hairstyle = "/home/yxh/Documents/HairNet"
#     workdir = os.path.dirname(__file__)
    
#     mesh = trimesh.load(os.path.join(workdir,'female_halfbody_medium.obj'))
#     render(mesh, current_convdata)
#     render(mesh, current_convdata, current_RT_mat)
    

    # for matplot img show3dhair
    # fig = plt.figure(figsize=(18, 6))
    # fig.set_tight_layout(False)
    # gs = gridspec.GridSpec(1, 3)
    # gs.update(wspace=0.05)
    # plt.axis('off')

    # # rgb_image = np.zeros((256, 256, 3), dtype=int)
    # # rgb_image[..., [0, 3]] = image * 255
    # # plot orientation map
    # ax1 = plt.subplot(gs[0])
    # ax1.imshow(rgb_image)

    # # plot hair ground truth
    # ax2 = plt.subplot(gs[1], projection='3d')
    # show3Dhair(ax2, strands, mask)

    # # plot predict hair
    # # ax3 = plt.subplot(gs[2], projection='3d')
    # # show3Dhair(ax3, pos, mask)
    # plt.gca().set_aspect('auto')
    # f = plt.gcf()  #获取当前图像
    # #plt.show()
    # f.savefig(f"/home/yxh/points_1.png")
    # plt.show()