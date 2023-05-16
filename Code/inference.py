from options.test_options import TestOptions
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
from Tools.drawTools import draw_arrows_by_projection1,draw_gt_arrows_by_projection
from Tools.to_unity import *
class orientation_inference:
    def __init__(self) -> None:
        
        pass
class strand_inference:
    def __init__(self) -> None:
        opt=TestOptions().parse()
        opt.gpu_ids=[0,1,2,3,4]
        gpu_str = [str(i) for i in opt.gpu_ids]
        gpu_str = ','.join(gpu_str)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        opt.name="2023-05-06_bust_test"
        opt.save_dir="data/Train_input"
        
        opt.is_Train = False
        opt.which_iter=5000
        opt.check_name="2023-05-06_bust"
        
        opt.model_name="GrowingNet"
        self.growing_solver = GrowingNetSolver()
        self.growing_solver.initialize(opt)
        opt.model_name=="HairSpatNet"
        self.spat_solver = HairSpatNetSolver()
        self.spat_solver.initialize(opt)
        opt.model_name=='HairModeling'
        self.hd_solver=HairModelingHDSolver()
        self.hd_solver.initialize(opt)


        
    def inference(self,image):
        
        orientation = self.spat_solver.inference(image)
        points,segments = self.growing_solver.inference(orientation)
        # if opt.model_name=="HairSpatNet":
        #     draw_gt_arrows_by_projection(os.path.join(opt.save_dir,dir_name))
        #     draw_arrows_by_projection1(os.path.join(opt.save_dir,dir_name),opt.which_iter,draw_occ=True)
        #     draw_arrows_by_projection1(os.path.join(opt.save_dir,dir_name),opt.which_iter,draw_occ=False)
        # 打开realistic-exe-linux项目可执行文件进行渲染
        reset()
        # points,segments = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.75],translation=[0.003389,-1.2727,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        # write_strand2abc1("/home/yxh/Documents/company/strandhair/strands00184.abc",segments,points)
        trans_hair(points,segments)
        render(f"/home/yxh/Documents/company/strandhair/{file.split('.')[0]}1.png")