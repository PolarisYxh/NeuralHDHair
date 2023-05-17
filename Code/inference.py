from options.inference_options import InferenceOptions
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
from Tools.drawTools import draw_arrows_by_projection1,draw_gt_arrows_by_projection
from Tools.to_unity import *
from image_filter import filter_crop
import cv2
class strand_inference:
    def __init__(self,rFolder) -> None:
        opt=InferenceOptions().initialize()
        self.iter = {}
        opt.gpu_ids=[0]
        gpu_str = [str(i) for i in opt.gpu_ids]
        gpu_str = ','.join(gpu_str)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        opt.current_path = rFolder
        opt.name="2023-05-06_bust_test"
        opt.save_dir="data/Train_input"
        opt.is_Train = False
        
        opt.model_name="GrowingNet"
        opt.save_root="checkpoints/GrowingNet"
        opt.which_iter=40000
        self.iter[opt.model_name] = opt.which_iter
        opt.check_name="2023-05-11_bust_prev3"
        opt.condition=True
        self.growing_solver = GrowingNetSolver()
        self.growing_solver.initialize(opt)
        
        opt.model_name="HairSpatNet"
        opt.save_root="checkpoints/HairSpatNet"
        opt.which_iter=1640000
        opt.no_use_L = True
        opt.no_use_depth=True
        opt.blur_ori = True
        self.iter[opt.model_name] = opt.which_iter
        opt.check_name="2023-04-17_bust"
        self.spat_solver = HairSpatNetSolver()
        self.spat_solver.initialize(opt)
        
        # opt.model_name=='HairModeling'
        # self.hd_solver=HairModelingHDSolver()
        # self.hd_solver.initialize(opt)
        self.img_filter = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code")

        
    def inference(self,image):
        ori2D,mask = self.img_filter.pyfilter2neuralhd(image)
        # ori2D = image
        orientation = self.spat_solver.inference(ori2D,mask)
        # draw_arrows_by_projection1(os.path.join("/home/yxh/Documents/company/NeuralHDHair/data/Train_input/","DB1"),self.iter["GrowingNet"],draw_occ=True,hair_ori=orientation)

        points,segments = self.growing_solver.inference(orientation)
        # 打开realistic-exe-linux项目可执行文件进行渲染
        reset()
        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.75],translation=[0.003389,-1.2727,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        trans_hair(points,segments)
        # render(f"/home/yxh/Documents/company/strandhair/{file.split('.')[0]}1.png")
        
if __name__=="__main__":
    img = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/data/test/Screenshot from 2023-03-15 15-36-32.png")
    hair_infe = strand_inference(os.path.dirname(os.path.dirname(__file__)))
    hair_infe.inference(img)