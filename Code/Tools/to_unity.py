import requests
import json
import numpy as np
from .file_io import *
from skimage import transform
def reset():
    ret = requests.get(f"http://127.0.0.1:11111/?type=revert&gender=male")
    if "完毕" not in ret.text:
        print(ret.text)
def render(save_name):
    ret = requests.get(f"http://127.0.0.1:11111/?type=capture&filePath={save_name}")
    print(ret.text)
def trans_hair(points,segments):
    js = {
        "type": "vertices",
        "name": "hair",
        "gender": "male"
    }
    js['vertices']=points.reshape((-1)).tolist()
    new_segments = np.zeros_like(segments)
    for i in range(0,len(segments)-1):
        new_segments[i+1]=segments[i]+new_segments[i]
    js['strands']=new_segments.tolist()
    ret = requests.post(f"http://127.0.0.1:11111",data=json.dumps(js))
    if "完毕" not in ret.text:
        print(ret.text)
def segment_trans(segments,points_num):
    last = segments[:-1]
    aft = segments[1:]
    seg = aft-last
    seg = np.append(seg,points_num-segments[-1])
    return seg
if __name__=="__main__":
    data_dir = "/home/yxh/Documents/company/strandhair/hair/"
    files = os.listdir(data_dir)
    for file in files:
        # file = "strands00184.hair"
        reset()
        segments,points = readhair(os.path.join(data_dir,f"{file}"))
        m = transform.SimilarityTransform(scale=[0.82,0.75,0.75],translation=[0.003389,-1.2727,-0.033233],dimensionality=3)#将blender的变换y,z互换后z取反
        points=transform.matrix_transform(points,m.params)
        # write_strand2abc1("/home/yxh/Documents/company/strandhair/strands00184.abc",segments,points)
        trans_hair(points,segments)
        render(f"/home/yxh/Documents/company/strandhair/{file.split('.')[0]}1.png")