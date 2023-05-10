# NeuralHDHair
The source code of the networks for our paper ["NeuralHDHair: Automatic High-fidelity Hair Modeling from a Single Image Using Implicit Neural Representations"](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_NeuralHDHair_Automatic_High-Fidelity_Hair_Modeling_From_a_Single_Image_Using_CVPR_2022_paper.pdf) (CVPR2022)



# Pipeline #
![Pipeline](Pipeline.png)

# Prerequisites

- Linux
- Python 3.8
- Pytorch 1.8.1
- NVIDIA GPU + CUDA 11.1


# get data #
1. Code/Tools/get_voxelsample.py to compute sample_voxel.mat

# Train #
1.Train Coarse module, intput: Ori.png、mask.png, output gt: Ori_gt.mat

    python main.py --name=yourname --model_name=HairSpatNet --blur_ori --no_use_depth --no_use_L --gpu_ids=0 --batch_size=1
2.Train Global module

    python main.py --name=yourname --model_name=HairModelingHD --blur_ori --no_use_depth  --gpu_ids=0 --batch_size=1 --pretrain_path=pretrain_model_path

3.Train GrowingNet, input:;output:hair_delete.hair

    python main.py --name=yourname --model_name=GrowingNet  --batch_size=1 --sd_per_batch=800 --pt_per_strand 72
# Test #
1.Test Coarse module
    python test.py --name=yourname --model_name=HairSpatNet --blur_ori --no_use_depth --no_use_L --gpu_ids=0 --batch_size=1
2. Test GrowingNet,
    
# visualize in tensorboard #
```
in server
tensorboard --logdir=/home/yangxinhang/NeuralHDHair/checkpoints/HairSpatNet/2023-04-13_bust/logs/train --port=10086
```
```
in 本地
ssh -L 10086:127.0.0.1:10086 151 -N -v -v
151即为
```
# Tips #
1.Data, some data processing and rendering code do not have permissions and cannot open source temporarily.

2.Use high-quality images as much as possible, the quality of reconstruction depends largely on the quality of  generated orientation map.

3.The hair should be aligned with the bust as much as possible, you may need the face alignment algorithm to calculate the affine transformation.(data/Train_input/DB1 contains standard input samples)


# Citation #
    @inproceedings{wu2022neuralhdhair,
    title={NeuralHDHair: Automatic High-fidelity Hair Modeling from a Single Image Using Implicit Neural Representations},
    author={Wu, Keyu and Ye, Yifan and Yang, Lingchen and Fu, Hongbo and Zhou, Kun and Zheng, Youyi},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={1526--1535},
    year={2022}
    }

  
  
