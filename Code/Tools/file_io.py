import sys
import struct
import copy
import pickle
import numpy as np
import os
from tqdm import tqdm
import ctypes
import logging
import json
import random
def readjson(file_name):
    with open(file_name,'r') as f:
        x = json.load(f)
    return x
def write_strand2obj(filepath, strands):
    # filepath = "/tmp/simple_obj.obj"
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        f.write(f"o group1\n")
        for strand in strands:
            for v in strand:
                f.write("v %.6f %.6f %.6f\n" % (v[0],v[1],v[2]))
        index = 1 ### start from index 1
        for strand in strands:
            for i in range(len(strand)-1):
                f.write(f"l {index} {index+1}\n")
                index += 1
            index+=1

TRESSFX_SIM_THREAD_GROUP_SIZE = 64 # from cn.unity.hairfx.core/Runtime/HairFXAsset.cs:177
class TressFXTFXFileHeader(ctypes.Structure):
	_fields_ = [('version', ctypes.c_float),
                ('numHairStrands', ctypes.c_uint),
                ('numVerticesPerStrand', ctypes.c_uint),
                ('offsetVertexPosition', ctypes.c_uint),
                ('offsetStrandUV', ctypes.c_uint),
                ('offsetVertexUV', ctypes.c_uint),
                ('offsetStrandThickness', ctypes.c_uint),
                ('offsetVertexColor', ctypes.c_uint),
                ('reserved', ctypes.c_uint * 32)]

class tressfx_float4(ctypes.Structure):
	_fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float),
                ('z', ctypes.c_float),
                ('w', ctypes.c_float)]

class tressfx_float2(ctypes.Structure):
	_fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float)]

def SaveTFXBinaryFile(filepath, curves, numVerticesPerStrand, meshShapedagPath="", sceneScale=1, curveSample=1, curveIndex_Offset=0, invertX=False,invertZ=False, useZUp=False, currentAxis='y',bothEndsImmovable=True):
    """_summary_

    Args:
        filepath (_type_): _description_
        curves (_type_): _description_
        numVerticesPerStrand (_type_): _description_
        meshShapedagPath (_type_): _description_
        sceneScale (int, optional): _description_. Defaults to 1.
        curveSample (int, optional): _description_. Defaults to 1.
        curveIndex_Offset (int, optional): _description_. Defaults to 0.
        invertZ (bool, optional): z坐标取反. Defaults to False.
        useZUp (bool, optional): 保存时向上坐标轴为z. Defaults to False.
        currentAxis (str, optional): 目前的向上坐标轴. Defaults to 'y'.
        bothEndsImmovable (bool, optional): 发丝两端点不能动. Defaults to True.
    """	
    # requirements from cn.unity.hairfx.core/Runtime/HairFXAsset.cs:177
    assert(numVerticesPerStrand > 2 and TRESSFX_SIM_THREAD_GROUP_SIZE%numVerticesPerStrand == 0 and numVerticesPerStrand<=TRESSFX_SIM_THREAD_GROUP_SIZE)
    assert(len(curves)*numVerticesPerStrand % 64 == 0)
    numCurves = len(curves)
	
	#check to see if we need to scale the points
    sceneScale = float(sceneScale)
    logging.info("TressFX: Saving TFX Binary:scene scale multiplier = %f" % sceneScale) #todo: make sure scaling doesn't affect uv issues (if there is an issue, that is)

    #sampling options
    curveSample = int(curveSample)
    #sanity check
    if curveIndex_Offset >= numCurves:
        logging.warning('TressFX: Curve Offset requested Greater or Equal to actual Number of Curves - Please Lower Offset Value')
        return

    bChangeUpToZ = False
    if (useZUp == True):
        #query current up axis in use
        if (currentAxis == 'y'):
            bChangeUpToZ = True
            logging.warning("TressFX: Maya currently using Y axis as UP, Z up requested: doing y/z swap")
        elif (currentAxis != 'z'):
            logging.warning("TressFX: Problem detected, attempt to detect Maya UP axis setting failed. No change to default UP axis (no y/z swap)")

    #useCurl = cmds.checkBox("useCurl",q = True, v = True)
    rootPositions = []

    tfxHeader = TressFXTFXFileHeader()
    tfxHeader.version = 4.0
    tfxHeader.numHairStrands = int(numCurves//curveSample) #number of curves may be a subset if we are sampling a subset only
    tfxHeader.numVerticesPerStrand = numVerticesPerStrand
    tfxHeader.offsetVertexPosition = ctypes.sizeof(TressFXTFXFileHeader)
    tfxHeader.offsetStrandUV = 0
    tfxHeader.offsetVertexUV = 0
    tfxHeader.offsetStrandThickness = 0
    tfxHeader.offsetVertexColor = 0

    #if sampling at a lower amount than the full curve set, div by sample over entire set, then mult by sample to jump
    #if offseting, subtract offset amount from full range of loop, so when it's added to the index
    #it won't overshoot the actual number of curves range	
    adjustedCurveRange = int(numCurves//curveSample) - curveIndex_Offset #floor division is //
    #sanity check 2
    if curveIndex_Offset >= adjustedCurveRange:
        logging.warning('TressFX: Curve Offset requested Greater or Equal to subset:Sampled Curves - Please Lower Offset Value')
        return

    f = open(filepath, "wb")
    f.write(tfxHeader)
    #if sampling at a lower amount than the full curve set, div by sample over entire set, then mult by sample to jump
    #if offseting, subtract offset amount from full range of loop, so when it's added to the index
    #it won't overshoot the actual number of curves range	
    #adjustedCurveRange = int(numCurves//curveSample) - curveIndex_Offset #floor division is //
    for i in tqdm(range(adjustedCurveRange)):
        curveFn = curves[(i*curveSample) + curveIndex_Offset] #adjusted curve range to accomodate sampling and offsetting into the curve set
        l=len(curveFn)
        sample_index = []
        if numVerticesPerStrand<l:
            for i in range(0, numVerticesPerStrand-1):
                sample_index.append(int(np.round(l/(numVerticesPerStrand-1)*i)))
            sample_index.append(l-1)
        else:
            for i in range(0, numVerticesPerStrand):
                if i>=len(curveFn):
                    sample_index.append(-1)
                    continue
                sample_index.append(i)
        for j in sample_index:
            pos = curveFn[j]
            p = tressfx_float4()
            p.x = pos[0]
            p.y = pos[1]

            if invertZ:
                p.z = -pos[2] # flip in z-axis
            else:
                p.z = pos[2]
            if invertX:
                p.x = -pos[0] # flip in z-axis
            else:
                p.x = pos[0]
            #if invertY: #no use case currently
            #	p.y = -pos.y
            #else:
            #	p.y = pos.y

            if (bChangeUpToZ == True): #Unreal uses Z as up (not Y), and Maya is currently using Y (so we need to swap values)
                temp = p.y
                p.y = p.z
                p.z = temp #not sure if can use p.yz = p.zy (or if supported on all possible python builds used with this exporter) so will use old fashioned way
            

            # w component is an inverse mass
            if j == 0 or j == 1: # the first two vertices are immovable always. 
                p.w = 0
            else:
                p.w = 1.0 
            
            if j == (numVerticesPerStrand-1) and bothEndsImmovable: #fix the last vertice of strand
                p.w = 0
            
            if (sceneScale != 1.0):
                #print('tfx scaling doing it...not 1.0')
                p.x = p.x * sceneScale
                p.y = p.y * sceneScale
                p.z = p.z * sceneScale
                
            f.write(p)
            
        rootPositions.append(curveFn[0])
    size=f.tell()
    print(size)
    # save uv coordinate
    # meshFn = None
    # meshIntersector = None
    # ignoreUVErrors = cmds.checkBox("ignoreUVErrorsCheckBox", q = True, v = True)
    #     # if meshShapedagPath is passed then let's get strand texture coords. To do this, we need MFnMesh and MMeshIntersector objects. 
    # if meshShapedagPath != None:
    #     meshFn = OpenMaya.MFnMesh(meshShapedagPath)
    #     meshIntersector = OpenMaya.MMeshIntersector()
    #     meshIntersector.create(meshShapedagPath.node())
    # #		tfxHeader.offsetStrandUV = tfxHeader.offsetVertexPosition + numCurves * numVerticesPerStrand * ctypes.sizeof(tressfx_float4)
    #     tfxHeader.offsetStrandUV = tfxHeader.offsetVertexPosition + adjustedCurveRange * numVerticesPerStrand * ctypes.sizeof(tressfx_float4)

    # bInvertYForUVs = cmds.checkBox("InvertYForUVs",q = True, v = True)
    # # if meshShapedagPath is passed then let's get strand texture coords by using raycasting to the mesh from each root position of hair strand.   
    # if meshFn != None: 
    #     #last known good u,v
    #     #in case we hit bad uv points, we can at least try to set them closer to the
    #     #last u,v set (versus 0,0)
    #     u_lkg = 0.0
    #     v_lkg = 0.0

    #     uMin = 0.0
    #     uMax = 1.0	
    #     vMin = 0.0
    #     vMax = 1.0	
    #     #query to see if we have a user defined custom uv bounding box
    #     bCustomUVRange = cmds.checkBox("useNonUniformUVRange",q = True, v = True)
    #     if (bCustomUVRange == True) and (bInvertYForUVs == True):
    #         #only need the min and max of v (currently)
    #         uMin = cmds.floatField("uMin", q = True, v = True)
    #         uMax = cmds.floatField("uMax", q = True, v = True)
    #         vMin = cmds.floatField("vMin", q = True, v = True)
    #         vMax = cmds.floatField("vMax", q = True, v = True)
    #         logging.warning("TressFX: Using custom UV range. This v range (%5.2f, %5.2f) will be used during DirectX y-flipping ('v' reflect) for uv coordinates." % (vMin, vMax))
    #     print ("TressFX: UV bounding box is: u[%5.2f, %5.2f], v[%5.2f, %5.2f]" % (uMin, uMax, vMin, vMax))

    #     for i in range(len(rootPositions)):
    #         rootPoint = rootPositions[i]

    #             # Find UV coordinates 
    #         util = OpenMaya.MScriptUtil()
    #         util.createFromList([0.0, 0.0], 2)
    #         uv_ptr = util.asFloat2Ptr()
            
    #         try: 
    #             meshFn.getUVAtPoint(rootPoint, uv_ptr)
    #             u = OpenMaya.MScriptUtil.getFloat2ArrayItem(uv_ptr, 0, 0)
    #             v = OpenMaya.MScriptUtil.getFloat2ArrayItem(uv_ptr, 0, 1)
    #             u_lkg = u
    #             v_lkg = v
    #         except:
    #             #if NOT strict mode' then ok to give point a default 0,0 uv point value, else kill the file
    #             #cmds.warning('Exception Hit! meshFn.getUVAtPoint failed for rootPoint')
    #             if ignoreUVErrors:
    #                 logging.warning('TressFX: UV point error Exception (strict mode off): UV point failed for rootPoint->Ignoring Exception, using last known good (lkg) as uv instead')
    #                 u = u_lkg
    #                 v = v_lkg
    #             else:
    #                 f.close()
    #                 logging.warning('TressFX: UV point error Exception (strict mode on): UV point failed for rootPoint->Failing to create TFX. Deleting the open TFX file: ' + filepath)
    #                 os.remove(filepath) #remove the damaged file
    #                 return
                    
    #         uv_coord = tressfx_float2()
    #         uv_coord.x = u
    #         uv_coord.y = v

    #         if (bInvertYForUVs == True):
    #             uv_coord.y = vMax - uv_coord.y + vMin #uv_coord.y = 1.0 - uv_coord.y # DirectX has it inverted, uniform means a typical bounding box of 0-1 u and v, so we would actually use 1- v + 0 to invert
                
    #         #print "uv:%g, %g\n" % (uv_coord.x, uv_coord.y)
    #         f.write(uv_coord)	

    f.close()
    return rootPositions

def ReadTFXBinaryFile(filepath):
    with open(filepath, "rb") as f:
        f.seek(0,2)                #指针移动到文件末尾
        size=f.tell()
        f.seek(0,0)  #指针移动到文件开头
        byte = f.read(160)
        header = struct.unpack("f 39I", byte)#!2B I 4H 24s I B
        strands = []
        weights = []
        for i in range(header[1]):
            strand = np.zeros((header[2],3)).astype('float')
            weight = np.zeros((header[2],1)).astype('float')
            for j in range(header[2]):
                byte = f.read(16)
                point = np.array(struct.unpack("4f", byte))
                strand[j] = point[:3]
                weight[j] = point[3]
            strands.append(strand)
            weights.append(weight)
        uv_s = []
        if f.tell()<size:
            for i in range(header[1]):
                byte = f.read(8)
                uv = np.array(struct.unpack("2f", byte))
                uv_s.append(uv)
	#check to see if we need to scale the points
    return strands, weights, uv_s

def readhair(file_name):
    with open(file_name, mode='rb')as f:
        num_strand = f.read(4)
        (num_strand,) = struct.unpack('I', num_strand)
        point_count = f.read(4)
        (point_count,) = struct.unpack('I', point_count)

        # print("num_strand:",num_strand)
        segments = f.read(2 * num_strand)
        segments = struct.unpack('H' * num_strand, segments)
        segments = list(segments)
        num_points = sum(segments)

        points = f.read(4 * num_points * 3)
        points = struct.unpack('f' * num_points * 3, points)
        segments = np.array(segments)
        points = np.array(points).reshape((-1,3))
    return segments,points

def writehair(file_name,points,segments):
    with open(file_name, 'wb')as f:
        f.write(struct.pack('I', len(segments)))
        f.write(struct.pack('I', len(points)))
        for num_every_strand in segments:
            f.write(struct.pack('H', num_every_strand ))

        for vec in points:
            f.write(struct.pack('f', vec[0]))
            f.write(struct.pack('f', vec[1]))
            f.write(struct.pack('f', vec[2]))

    f.close()
    
def get_data(file_name):
    with open(file_name, "rb") as f:
        byte = f.read(4)
        strands_num=int.from_bytes(byte, sys.byteorder)
        i,j=0,0
        strands = []
        # pickle.loads(pickle.dumps(origin_list))
        for i in range(0, strands_num):
            byte = f.read(4)
            v_num = int.from_bytes(byte, sys.byteorder)
            byte = f.read(4 * v_num * 3)
            points = np.array(struct.unpack('f' * v_num * 3, byte)).reshape((-1,3))     
            if v_num!=1:
                strands.append(points.tolist())
        return strands
# try:
#     from curvesAbc import CurvesAbc
# except:
#     from .curvesAbc import CurvesAbc
# def write_strand2abc(output_path, curves):
#     # create empty archive and put some objects in it
#     c=CurvesAbc()
#     c.CurvesExport(curves,output_path)
    
# def write_strand2abc1(output_path, segments, curves):
#     # create empty archive and put some objects in it
#     c=CurvesAbc()
#     c.CurvesExport1(segments,curves,output_path)
# def read_abc(input_path):
#     c=CurvesAbc()
#     segments,points = c.testCurvesImport(input_path)
#     return segments,points
