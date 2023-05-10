from Tools.utils import *
import os
from scipy.ndimage import binary_dilation
def get_vox_total_pic(V, dd=1):
    flag = False
    maskA = None
    Img = np.zeros(shape=[V.shape[1], V.shape[2]], dtype=np.float32)

    for sliceID in range(V.shape[0] // dd):# 从前往后遍历96个体素
        sliceImg = V[sliceID, :, :]
        maskB = sliceImg!=0  # H * W
        # if np.max(maskB):
        #     print(sliceID)
        if (not flag):
            flag = True
            maskA = maskB.copy()
            Img[maskB] = sliceImg[maskB]
        else:
            mask = np.logical_xor(np.logical_or(maskA, maskB), maskA)#类似累加，上一次已经设置了，这次就不设置了
            Img[mask] = sliceImg[mask]
            maskA = np.logical_or(maskA, maskB)

    return Img * 255


def get_vox_slice_pic(V, sliceID=48):
    sliceImg = V[sliceID, :, :, :].copy()
    mask = (sliceImg ** 2).sum(-1) > 1e-3
    sliceImg[mask, :] = (sliceImg[mask, :] + 1.0) * 0.5
    sliceImg = np.clip(sliceImg, 0, 1)
    return sliceImg*255

def draw_occ_by_projection(fileDir, hair_occ, test=False):
    h = 128
    w = 128
    d = 96
    flip = True
    noise = True
    ori = os.path.join(fileDir, "Ori.png").replace("\\", "/")
    # target = cv2.imread(ori)
    # target = cv2.resize(target, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    # target = cv2.flip(target,flipCode=1)
    target = np.zeros((1024,1024,3)).astype('uint8')
    # hair_ori = get_ground_truth_3D_ori(fileDir, flip)
    iter="gt"
    image = get_vox_total_pic(hair_occ)/255 # image 128*128*3, (0,1)
    # cv2.imwrite("2.png",image.astype('uint8'))
    mask = image > 0


    for hh in range(h):
        for ww in range(w):
            if mask[hh, ww]:
                center = np.array([ww * 8 + 4, hh * 8 + 4])
                cv2.circle(target,center=center,radius=4,color=(0,255,0))

    cv2.imwrite(os.path.join(fileDir,f"occ_{iter}.jpg"), target)
    
# voxel_sample = scipy.io.loadmat("voxel_sample.mat")['voxel_sample'].transpose(2, 0, 1)
# root_path = "/home/yangxinhang/NeuralHDHair/data/cache"
# voxel_sample=np.load(os.path.join(root_path,"strands00420","sample_voxel.npy")).astype('bool')
# voxel_sample = np.logical_not(voxel_sample)

# draw_occ_by_projection("/home/yangxinhang/NeuralHDHair",voxel_sample)
if __name__=="__main__":
    root_path = "/home/yangxinhang/NeuralHDHair/data/cache"
    files = os.listdir(root_path)
    occ = np.zeros((96,128,128))
    
    struct1 = np.ones((11, 11, 11), dtype=np.int)
    for dir_name in files:
        voxel_sample = np.zeros((96,128,128))
        gt_orientation = get_ground_truth_3D_ori(os.path.join(root_path,dir_name), False, growInv=False)
        x = np.sum(gt_orientation,axis=-1)
        y = x!=0
        occ[y]=1
        # draw_occ_by_projection(os.path.join(root_path,dir_name),occ)
        dilated_occ = binary_dilation(occ, structure=struct1)
        # draw_occ_by_projection(os.path.join(root_path,dir_name),dilated_occ)
        voxel_sample[dilated_occ==False]=1
        # draw_occ_by_projection("/home/yangxinhang/NeuralHDHair",voxel_sample)
        voxel_sample=voxel_sample.astype('int')[None,...,None]
        np.save(os.path.join(root_path,dir_name,"sample_voxel.npy"),voxel_sample)
    # scipy.io.savemat("voxel_sample.mat", {"voxel_sample": voxel_sample.transpose(1, 2, 0).astype('int')})
