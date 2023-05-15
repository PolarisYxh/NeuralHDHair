from options.test_options import TestOptions
from dataload import data_loader
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
from Tools.drawTools import draw_arrows_by_projection1,draw_gt_arrows_by_projection
from Tools.to_unity import *
from Tools.resample import resample
opt=TestOptions().parse()
gpu_str = [str(i) for i in opt.gpu_ids]
gpu_str = ','.join(gpu_str)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
dataloader=data_loader(opt)
if opt.model_name=="GrowingNet":
    g_slover = GrowingNetSolver()
elif opt.model_name=="HairSpatNet":
    g_slover = HairSpatNetSolver()
elif opt.model_name=='HairModeling':
    g_slover=HairModelingHDSolver()


g_slover.initialize(opt)
dir_names = os.listdir(opt.save_dir)
for dir_name in dir_names:
    opt.test_file=dir_name
    g_slover.test(dataloader)
    if opt.model_name=="HairSpatNet":
        draw_gt_arrows_by_projection(os.path.join(opt.save_dir,dir_name))
        draw_arrows_by_projection1(os.path.join(opt.save_dir,dir_name),opt.which_iter,draw_occ=True)
        draw_arrows_by_projection1(os.path.join(opt.save_dir,dir_name),opt.which_iter,draw_occ=False)
    elif opt.model_name=="GrowingNet":
        # 打开realistic-exe-linux项目可执行文件进行渲染
        reset()
        segments,points = readhair(os.path.join(opt.save_dir,dir_name,f"hair_{opt.which_iter}.hair"))
        points,segments = resample(points,segments)
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.75],translation=[0.003389,-1.2727,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        # write_strand2abc1("/home/yxh/Documents/company/strandhair/strands00184.abc",segments,points)
        trans_hair(points,segments)
        render(f"/home/yxh/Documents/company/strandhair/{file.split('.')[0]}1.png")