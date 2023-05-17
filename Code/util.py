#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import os, logging


def Process(cmd):
    logging.info(cmd)
    os.system(cmd)


import base64


def file_b64encode(file_path):
    with open(file_path, 'rb') as f:
        file_string = f.read()
    file_b64encode = str(base64.b64encode(file_string), encoding='utf-8')
    return file_b64encode


def file_b64decode(file_Base64, file_path):
    missing_padding = len(file_Base64) % 4
    if missing_padding != 0:
        file_Base64 += ('=' * (4 - missing_padding))
    file_encode = base64.b64decode(file_Base64)
    with open(os.path.normcase(file_path), 'wb') as f:
        f.write(file_encode)
    return


import cv2
import numpy as np


def cvmat2base64(img_np, houzhui='.png'):
    #opencv的Mat格式转为base64
    image = cv2.imencode(houzhui, img_np)[1]
    base64_data = str(base64.b64encode(image))
    return base64_data[2:-1]


def base642cvmat(base64_data):
    #base64转为opencv的Mat格式
    imgData = base64.b64decode(base64_data)
    nparr = np.frombuffer(imgData, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    return img_np


import json


def readjson(file):
    try:
        with open(file, 'r', encoding="utf-8") as load_f:
            load_dict = json.load(load_f)
    except:
        with open(file, 'r', encoding="utf-8-sig") as load_f:
            load_dict = json.load(load_f)
    return load_dict


def writejson(file, write_dict):
    with open(file, "w", encoding="utf-8") as dump_f:
        json.dump(write_dict, dump_f, ensure_ascii=False)


import numpy as np


def readmat(fn):
    size_cv2np = [1, 1, 2, 2, 4, 4, 8, 2]
    type_cv2np = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64, np.float16]
    with open(fn, 'rb') as f:
        rows = np.frombuffer(f.read(4), dtype=np.int32)[0]  # first 4 byte
        cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
        type = np.frombuffer(f.read(4), dtype=np.int32)[0]
        mat = np.frombuffer(f.read(size_cv2np[type] * rows * cols), dtype=type_cv2np[type]).reshape([rows, cols])
        return mat


def readmats(fn):
    size_cv2np = [1, 1, 2, 2, 4, 4, 8, 2]
    type_cv2np = [np.uint8, np.int8, np.uint16, np.int16, np.int32, np.float32, np.float64, np.float16]
    mats = []
    with open(fn, 'rb') as f:
        while True:
            try:
                rows = np.frombuffer(f.read(4), dtype=np.int32)[0]  # first 4 byte
                cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
                type = np.frombuffer(f.read(4), dtype=np.int32)[0]
                mat = np.frombuffer(f.read(size_cv2np[type] * rows * cols), dtype=type_cv2np[type]).reshape([rows, cols])
                mats.append(mat)
            except:
                return mats


def writemat(fn, mat):
    ts = {'uint8': 0, 'int8': 1, 'uint16': 2, 'int16': 3, 'int32': 4, 'float32': 5, 'float64': 6, 'float16': 7}
    from struct import pack
    b = pack('iii', *mat.shape, ts[mat.dtype.name])
    b += mat.tobytes()
    with open(fn, 'bw') as f:
        f.write(b)


def timeCost(func):
    import time

    def wrapper(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        end = time.time()
        response_time = end - start
        logging.info(f"{func.__qualname__} response_time = {round(response_time, 3)}")
        return result

    return wrapper


from skimage import transform as trans
import numpy as np
import cv2


def transform(center, output_size, scale, rotation, data=None):
    scale_ratio = scale
    rot = float(rotation)
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = None
    if data is not None:
        cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return M, cropped


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def getScalingRotationTranslateLoss(srcPts, dstPts, scale_same=False):
    '''最小二乘法计算源点集到目标点集的旋转,缩放,平移和变换后的l2距离(即形状差异)
    param:
        scale_same:True x,y方向上缩放相等,否则不相等
    '''
    from skimage import transform
    if scale_same:
        tp = 'similarity'
    else:
        tp = 'affine'
    tform = transform.estimate_transform(tp, srcPts, dstPts)
    mt = transform.matrix_transform(srcPts, tform.params)
    loss = np.average(np.sqrt(np.sum((mt - dstPts)**2, axis=1)), axis=0)
    return tform, loss, mt


def drawLms(img, lms, color=(0, 255, 0)):
    img = img.copy()
    for lm in lms:
        cv2.circle(img, tuple(lm), 2, color, 1)
    return img


def SampleRecColor(img, point, w, h):
    """以point(x,y)为中心，在图片img上，构建宽为w，高为h的矩形，计算img中矩形区域的颜色均值，并将该矩形区域颜色置为0，返回颜色均值
    
    Args:
        img (array): image
        point (list of int): (x,y)
        w (int): width 
        h (int): height

    Returns:
        list of float: 矩形区域的颜色均值
    """
    w = max(int(w), 1)
    h = max(int(h), 1)
    roi = img[int(point[1] - h):int(point[1] + h), int(point[0] - w):int(point[0] + w), :]
    avg_color = np.mean(np.mean(roi, axis=0), axis=0)
    avg_color = list(avg_color)
    img[int(point[1] - h):int(point[1] + h), int(point[0] - w):int(point[0] + w), :] = roi * 0
    #cv2.imshow("test", img)
    #cv2.waitKey(0)
    return avg_color


def GetFaceColor(img, lms, sampleColorConfig, bShow=False):
    """根据人脸特征点坐标lms以及采样点配置sampleColorConfig，在img上采样不同区域的颜色值

    Args:
        img (_type_): 人脸照片
        lms (_type_): 与img对应的人脸特征点坐标
        sampleColorConfig (_type_): 不同区域采样时所采用的人脸特征点索引配置
        bShow (bool, optional): 是否显示. Defaults to False.

    Returns:
        _type_: color_face_rgb, color_chun_rgb, color_mei_rgb
    """
    # img = img1.copy()
    config_face = sampleColorConfig["face"]
    #sampling point of face
    sp_face = ((lms[config_face[0][0]] + lms[config_face[0][1]]) / 2).astype(np.int64)
    color_face0 = SampleRecColor(img, sp_face, 11, 11)
    #绘制十字标志
    cv2.drawMarker(img, tuple(sp_face), color_face0, thickness=3)
    sp_face = ((lms[config_face[1][0]] + lms[config_face[1][1]]) / 2).astype(np.int64)
    color_face1 = SampleRecColor(img, sp_face, 11, 11)
    cv2.drawMarker(img, tuple(sp_face), color_face1, thickness=3)
    color_face = list(np.array(color_face0 + np.array(color_face1)) / 2)
    color_face_rgb = [int(color_face[2]), int(color_face[1]), int(color_face[0])]

    config_chun = sampleColorConfig["chun"]
    sp_chun0 = ((lms[config_chun[0][0]] + lms[config_chun[0][1]]) / 2).astype(np.int64)
    color_chun0 = SampleRecColor(img, sp_chun0, abs(lms[config_chun[0][0]][1] - lms[config_chun[0][1]][1]) / 4, abs(lms[config_chun[0][0]][1] - lms[config_chun[0][1]][1]) / 4)
    cv2.drawMarker(img, tuple(sp_chun0), color_chun0)
    sp_chun1 = ((lms[config_chun[1][0]] + lms[config_chun[1][1]]) / 2).astype(np.int64)
    color_chun1 = SampleRecColor(img, sp_chun1, abs(lms[config_chun[1][0]][1] - lms[config_chun[1][1]][1]) / 4, abs(lms[config_chun[1][0]][1] - lms[config_chun[1][1]][1]) / 4)
    cv2.drawMarker(img, tuple(sp_chun1), color_chun1)
    color_chun = list(np.array(color_chun0 + np.array(color_chun1)) / 2)
    color_chun_rgb = [int(color_chun[2]), int(color_chun[1]), int(color_chun[0])]
    
    #TODO 眉毛区域中，只取和肤色差异较大的颜色的中位数作为眉毛颜色
    config_mei = sampleColorConfig["mei"]
    sp_mei0 = lms[config_mei[0]].astype(np.int64)
    sp_mei0[1] = sp_mei0[1] + 7
    color_mei0 = SampleRecColor(img, sp_mei0, 5, 5)
    cv2.drawMarker(img, tuple(sp_mei0), color_mei0)
    sp_mei1 = lms[config_mei[1]].astype(np.int64)
    sp_mei1[1] = sp_mei1[1] + 7
    color_mei1 = SampleRecColor(img, sp_mei1, 5, 5)
    cv2.drawMarker(img, tuple(sp_mei1), color_mei1)
    color_mei = list(np.array(color_mei0 + np.array(color_mei1)) / 2)
    color_mei_rgb = [int(color_mei[2]), int(color_mei[1]), int(color_mei[0])]

    if False:
        config_midMouth = sampleColorConfig["midMouth"]
        sp_midMouth0 = lms[config_midMouth[0]].astype(np.int64)
        sp_midMouth0[1] = sp_midMouth0[1] + 7
        color_midMouth0 = SampleRecColor(img, sp_midMouth0, 5, 5)
        cv2.drawMarker(img, tuple(sp_midMouth0), color_midMouth0)
        sp_midMouth1 = lms[config_midMouth[1]].astype(np.int64)
        sp_midMouth1[1] = sp_midMouth1[1] + 7
        color_midMouth1 = SampleRecColor(img, sp_midMouth1, 5, 5)
        cv2.drawMarker(img, tuple(sp_midMouth1), color_midMouth1)
        color_midMouth = list(np.array(color_midMouth0 + np.array(color_midMouth1)) / 2)
        color_midMouth_rgb = [int(color_midMouth[2]), int(color_midMouth[1]), int(color_midMouth[0])]
    else:
        color_midMouth_rgb = None
    #TODO提取瞳色，提取眼影颜色
    if (bShow):
        cv2.imshow("test", img)
        cv2.waitKey(0)
    #cv2.imwrite("/home/yxh/1.png", img)
    return color_face_rgb, color_chun_rgb, color_mei_rgb, color_midMouth_rgb


def beard_possibility(img, color_face, color_mei, *args):
    """以point(x,y)为中心，在图片img上，构建宽为w，高为h的矩形，
    计算img中矩形区域的不同于肤色的像素比例，并将该矩形区域颜色置为0，返回该比例
    
    Args:
        img (array): image
        point (list of int): (x,y)
        w (int): width 
        h (int): height

    Returns:
        list of float: 矩形区域的颜色均值
    """
    #import matplotlib.pyplot as plt
    if len(args[0]) == 2:
        w = args[0][1][0]
        h = args[0][1][1]
        point = args[0][0]
        w = max(int(w), 1)
        h = max(int(h), 1)
        roi = img[int(point[1] - h):int(point[1] + h), int(point[0] - w):int(point[0] + w), :]
        avg_color = np.mean(np.mean(roi, axis=0), axis=0)
        # img[int(point[1] - h):int(point[1] + h), int(point[0] - w):int(point[0] + w), :] = 0
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
    else:
        area = np.array(args[0])
        r1 = np.zeros([img.shape[0], img.shape[1], 1])
        r1 = cv2.fillConvexPoly(r1, area, 1)
        rect = np.array([(area[0] + area[1]) / 2, (area[0] + area[2]) / 2, 0.25 * area[1] + 0.75 * area[2], 0.75 * area[1] + 0.25 * area[2]]).astype('int')
        r1 = cv2.fillConvexPoly(r1, rect, 2)
        x = np.all((r1 == [1]), axis=2)
        roi = img[x]
        # img[x]=[255,0,0]
        avg_color = np.mean(roi, axis=0)
    avg_color = np.array([int(avg_color[2]), int(avg_color[1]), int(avg_color[0])])
    # color_face = cv2.cvtColor(np.array([[color_face]]).astype("uint8"),cv2.COLOR_RGB2HSV_FULL)
    # color_mei = cv2.cvtColor(np.array([[color_mei]]).astype("uint8"),cv2.COLOR_RGB2HSV_FULL)
    # avg_color = cv2.cvtColor(np.array([[avg_color]]).astype("uint8"),cv2.COLOR_RGB2HSV_FULL)
    c1 = np.array(color_face).astype("int32") - np.array(color_mei).astype("int32")
    c1 = np.where(c1 < 0, -c1, c1)
    ratio = (avg_color - np.array(color_mei)) / c1
    ratio = np.mean(ratio)
    return 1 - ratio, avg_color.tolist()


def GetBeardInfo(img, color_face, color_mei, lms, sampleColorConfig, bShow=False):
    config_beard = sampleColorConfig["beard"]
    # info = [[],[],[],[],0]#lip_up intensity, jaw intensity, cheek intensity, corner intensity, lip_down intensity
    info_list = []
    beard_colors = []
    #上唇,下巴,两脸颊
    for i, key in enumerate(['lip_up', "jaw", "cheek"]):
        for point in config_beard[key]:
            for n in range(2, len(point)):
                sp_beard0 = (((1 - point[n]) * lms[point[0]] + point[n] * lms[point[1]])).astype(np.int64)
                l = max(abs(lms[point[0]][1] - lms[point[1]][1]) / 8, abs(lms[point[0]][0] - lms[point[1]][0]) / 8)
                ratio, color = beard_possibility(img, color_face, color_mei, [sp_beard0, [l, l]])
                # info[i].append(ratio)
                info_list.append(ratio)
                beard_colors.append(color)

    for point in config_beard['corner']:
        for n in range(1, len(point)):
            z = 0.4 * (lms[point[0]] - lms[config_beard['corner'][0][0]]) + 0.4 * (lms[point[0]] - lms[config_beard['corner'][1][0]])
            sp_beard0 = ((lms[point[0]] + point[n] * (lms[config_beard['lip_down'][0][1]] - lms[config_beard['lip_down'][0][0]] + z))).astype(np.int64)
            l = max(abs(lms[config_beard['lip_down'][0][1]][1] - lms[config_beard['lip_down'][0][0]][1]) / 8, abs(lms[config_beard['lip_down'][0][1]][0] - lms[config_beard['lip_down'][0][0]][0]) / 4)
            ratio, color = beard_possibility(img, color_face, color_mei, [sp_beard0, [l / 2, l]])
            # info[i+1].append(ratio)
            info_list.append(ratio)
            beard_colors.append(color)
    #下唇
    triangle = []
    for point in config_beard['lip_down']:  #三个点，采样三角形内区域
        sp_beard0 = (((1 - point[2]) * lms[point[0]] + point[2] * lms[point[1]])).astype(np.int64)
        triangle.append(sp_beard0)
    ratio, color = beard_possibility(img, color_face, color_mei, triangle)
    beard_colors.append(color)
    # info[4] = ratio
    info_list.append(ratio)
    return info_list, beard_colors
