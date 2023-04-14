import re
import os
#导入库
import matplotlib.pyplot as plt
import numpy as np
with open("/home/yangxinhang/NeuralHDHair/checkpoints/HairSpatNet/2023-04-07_bust1/logs/loss_log.txt") as f:
    info = f.readlines()
errors = {"ori_loss":[],"occ_loss":[]}
for epoch in range(32,467,1):
    y=list(filter(lambda a:f"epoch: {epoch}," in a, info))
    y=str(y)
    tim = re.findall("ori_loss: [0-9]*\.*[0-9]+",y)
    iter = re.findall("iters: [0-9]*,",y)
    # print(tim)
    t=0
    num0 = 0
    for i,x in enumerate(tim):
        y1 = float(x[9:])
        # num = int(iter[i][7:-1])-num0
        # num0=int(iter[i][7:-1])
        t+=y1
        num0+=1
    # print(t)
    # print(t/num0)
    errors['ori_loss'].append(t/num0)
    tim = re.findall("occ_loss: [0-9]*\.*[0-9]+",y)
    # print(tim)
    t=0
    num0 = 0
    for x in tim:
        y1 = float(x[9:])
        # num = int(iter[i][7:-1])-num0
        # num0=int(iter[i][7:-1])
        t+=y1
        num0+=1
    # print(t)
    # print(t/num0)
    errors['occ_loss'].append(t/num0)

#设定画布。dpi越大图越清晰，绘图时间越久
# fig=plt.figure(figsize=(4, 4), dpi=300)
# #导入数据
# x=list(np.arange(1, len(errors["occ_loss"])+1))
# #绘图命令
# plt.plot(x, errors['occ_loss'], lw=4, ls='-', c='b', alpha=0.1)
# plt.plot()
# #show出图形
# plt.show()
# #保存图片
# fig.savefig("画布")

fig=plt.figure(figsize=(16, 16), dpi=600)
#导入数据
x=list(np.arange(1, len(errors["ori_loss"])+1))
#绘图命令
plt.plot(x, errors['occ_loss'], c='red', label="occ")
# plt.scatter(x, errors['occ_loss'], c='red')

plt.plot(x, errors['ori_loss'], c='b', label="ori")
# plt.scatter(x, errors['ori_loss'], c='b')
plt.plot()
#show出图形
plt.show()
#保存图片
fig.savefig("画布9")