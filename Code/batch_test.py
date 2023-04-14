from options.test_options import TestOptions
from dataload import data_loader
from solver.GrowingNetSolver import GrowingNetSolver
from solver.HairSpatNetSolver import HairSpatNetSolver
from solver.HairModelingHDSolver import HairModelingHDSolver
import os
from Tools.drawTools import draw_arrows_by_projection1,draw_gt_arrows_by_projection
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
    draw_gt_arrows_by_projection(os.path.join(opt.save_dir,dir_name))
    draw_arrows_by_projection1(os.path.join(opt.save_dir,dir_name),opt.which_iter,draw_occ=True)
    draw_arrows_by_projection1(os.path.join(opt.save_dir,dir_name),opt.which_iter,draw_occ=False)