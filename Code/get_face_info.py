import cv2
import numpy as np

    
try:
    from .util import transform,trans_points2d,trans_points3d
except:
    from util import transform,trans_points2d,trans_points3d
import math

from scipy.spatial.transform import Rotation
import os
def angle2matrix(theta) :
    '''theta:rad
    '''
    R_x = np.array([[1, 0, 0 ],
    [0, math.cos(theta[0]), -math.sin(theta[0]) ],
    [0, math.sin(theta[0]), math.cos(theta[0]) ]
    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1]) ],
    [0, 1, 0 ],
    [-math.sin(theta[1]), 0, math.cos(theta[1]) ]
    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
    [math.sin(theta[2]), math.cos(theta[2]), 0],
    [0, 0, 1]
    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def drawLms(img, lms, color=(255, 0, 0),x="1"):
    img1 = img.copy()
    for i,lm in enumerate(lms):
        cv2.circle(img1, tuple(lm), 1, color, 2)
        # cv2.putText(img,'%d'%(i), (lm[0]-2, lm[1]-3),cv2.FONT_HERSHEY_DUPLEX,0.3,(0,0,255),1)
    # cv2.imshow(x,img1)
    # cv2.waitKey()
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
def barycentric(A, B, C, P):
    """
    A, B, C, P: Vector3, points
    return u: Vector3, barycentric coordinate of P
    """
    s1 = np.array([B[0] - A[0], C[0] - A[0], A[0] - P[0]])
    s2 = np.array([B[1] - A[1], C[1] - A[1], A[1] - P[1]])
    u = np.cross(s1,s2)
    u = u/np.linalg.norm(u)
    if abs(u[2]) > 1e-2: # means this works, we have the P inside triangle
        barycentric = np.array([1-(u[0]+u[1])/u[2], u[0]/u[2], u[1]/u[2]])
        P[2] = np.dot(barycentric,np.array([A[2],B[2],C[2]]))
        return barycentric
    return np.array((-1,1,1))
import math


class get_face_info:
    def __init__(self,path,bInsightFaceLocal) -> None:
        ctx=0
        det_size=640
        model_pack_name = 'buffalo_l'
        self.bInsightFaceLocal = bInsightFaceLocal
        if bInsightFaceLocal:
            try:
                # from .insightface.insightfacePackage import utils
                from .insightface.insightfacePackage.app import FaceAnalysis
                from .insightface.insightfacePackage.data import get_image as ins_get_image
            except:
                # from insightface.insightfacePackage import utils
                from insightface.insightfacePackage.app import FaceAnalysis
                from insightface.insightfacePackage.data import get_image as ins_get_image
            self.app = FaceAnalysis(name=model_pack_name, root = os.path.join(path,'insightface'),allowed_modules=['detection','landmark_3d_68','landmark_2d_106','recognition'],providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=ctx, det_size=(det_size,det_size))
        else:
            try:
                from .http_interface import insightFaceInterface
            except:
                from http_interface import insightFaceInterface
            self.app = insightFaceInterface(path)
    
    def get_faces(self,img, name=""):
        if 600-img.shape[0]<0 or 600-img.shape[1]<0:
            scale = 600/max(img.shape[0],img.shape[1])
            img = cv2.resize(img,(int(scale*img.shape[1]), int(scale*img.shape[0])))
        x = (640-img.shape[0])//2 if 640-img.shape[0]>0 else 0
        y = (640-img.shape[1])//2 if 640-img.shape[1]>0 else 0
        img = cv2.copyMakeBorder(img, x, x, y, y, cv2.BORDER_CONSTANT)
        if self.bInsightFaceLocal:
            self.faces = self.app.get(img)
        else:
            self.faces = self.app.request_insightFace(img,name)
        # self.img = img
        # self.x = self.app.draw_on(img, self.faces)
        if len(self.faces)>=1:
            face_id=0
            if len(self.faces)>1:
                #取box area最大的face
                area_max=0
                for i in range(len(self.faces)):
                    bbox=self.faces[i].bbox
                    w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                    area = w*h
                    if area>area_max:
                        face_id=i
                        area_max=area
            #先判断图片旋转角度过大，是则进行旋转，防止关键点检测误差过大
            face = self.faces[face_id]
            landmark3d_align = face["landmark_3d_68"].astype('float64')
            delta = landmark3d_align[16,:]-landmark3d_align[0,:]
            yaw = -np.arctan(delta[2]/delta[0])
            # M_res = angle2matrix([0, 0, 0])
            M1 = angle2matrix([0, yaw, 0])
            landmark3d_align = np.dot(landmark3d_align, M1)
            delta = landmark3d_align[16,:]-landmark3d_align[0,:]
            if landmark3d_align[16,0]<landmark3d_align[0,0]:
                roll = np.arctan(delta[1]/delta[0])+np.pi
            else:
                roll = np.arctan(delta[1]/delta[0])
            # roll = np.deg2rad(face["pose"][2])
            if roll <= np.deg2rad(-45) or roll >= np.deg2rad(45):
                bbox = face.bbox
                w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
                rotate = 0
                _scale = 640  / (max(w, h)*2)
                self.M,frame = transform(center, 640, _scale, -roll, img)
                # cv2.imshow("1",frame)
                # cv2.waitKey()
                if self.bInsightFaceLocal:
                    self.faces = self.app.get(frame)
                else:
                    self.faces = self.app.request_insightFace(frame,name)
                img = frame
            frames = []
            framesForGenders = []
            for face in self.faces:
                bbox = face.bbox
                w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
                rotate = 0
                _scale = 640  / (max(w, h)*2)
                self.M,frame = transform(center, 640, _scale, 0, img)
                _scale = 640  / (max(w, h)*1.5)
                _,frameForGender = transform(center, 640, _scale, 0, img)
                face.landmark_2d_106 = trans_points2d(face.landmark_2d_106, self.M).astype('int')
                face.landmark_3d_68 = trans_points3d(face.landmark_3d_68, self.M).astype('int')
                # drawLms(frame, face.landmark_2d_106)
                # drawLms(frameForGender,face.landmark_2d_106)
                frames.append(frame)
                framesForGenders.append(frameForGender)
                # cv2.imshow("1",frame)
                # cv2.waitKey()
            return self.faces,frames,framesForGenders
        else:
            return None,None,None
    def get_origin_faces(self,img, name=""):
        frames = []
        framesForGenders = []
        poses = []
        if self.bInsightFaceLocal:
            self.faces = self.app.get(img)
        else:
            self.faces = self.app.request_insightFace(img,name)
        frames.append(img)
        framesForGenders.append(img)
        poses.append([0,0,0,0,0,130])
        return frames, framesForGenders
    def get_recognition_feature(self,index):
        return self.faces[index].normed_embedding
    def get_pose(self,index):
        return self.faces[index].pose
    def get_lms_2d(self,index):
        return True,self.faces[index].landmark_2d_106
    def get_lms_3d(self,index):#以像素为单位的x,y,z坐标
        return True,self.faces[index].landmark_3d_68
    def get_lms_forRecon(self,index):
        self.recon_lms = self.faces[index].landmark_2d_106
        self.recon_lms = np.delete(self.recon_lms,np.s_[34,38,92,88],axis=0)
        self.recon_lms = self.recon_lms[33:]
        self.recon_lms = np.r_[self.faces[index].landmark_3d_68[:17,:2],self.recon_lms]
        
        # drawLms(self.img,self.recon_lms[:,:2].astype('int'),color = [0,0,255])
        return True,self.recon_lms[:,:2]
        # self.recon_lms
    def get_gender(self,index):
        if self.faces[index].gender==1:
            return 'male'
        else:
            return 'female'
    def align_face1(self,index):#only plane fit
        plane = Plane()
        face_plane = np.array([8,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])#17,18,19,20,21,22,23,24,25,26
        face_plane = np.concatenate((face_plane,np.arange(48,68)))
        mid_plane = np.array([27,28,29,30,33,51,62,66,57,8])
        # drawLms(self.img,self.faces[index].landmark_3d_68[:,:2].astype('int'))
        best_eq, best_inliers1 = plane.fit(self.faces[index]["landmark_3d_68"][face_plane,:], thresh=20)
        pitch = np.arctan(best_eq[1]/best_eq[2])
        M = angle2matrix([-pitch, 0, 0])
        landmark3d_align = np.dot(self.faces[index].landmark_3d_68, M)
        # drawLms(self.img,landmark3d_align[:,:2].astype('int'))
        best_eq, best_inliers2 = plane.fit(landmark3d_align[face_plane,:], thresh=20)
        yaw = np.arctan(best_eq[0]/best_eq[2])
        M = angle2matrix([0, yaw, 0])
        # best_eq, best_inliers = plane.fit(landmark3d_align[mid_plane,:], thresh=1)
        # yaw = np.arctan(best_eq[2]/best_eq[0])
        # M = angle2matrix([0, -yaw, 0])
        landmark3d_align = np.dot(landmark3d_align, M)
        landmark3d_align = self.faces[index].landmark_3d_68[28]-landmark3d_align[28]+landmark3d_align
        # drawLms(self.img,landmark3d_align[:,:2].astype('int'),x="1")
        best_eq, best_inliers3 = plane.fit(landmark3d_align[mid_plane,:], thresh=1,init_guess=[-1,0,0,0])#
        roll = np.arctan(best_eq[1]/best_eq[0])
        M = angle2matrix([0, 0, roll])
        # delta = self.faces[index]["landmark_3d_68"][16,:]-self.faces[index]["landmark_3d_68"][0,:]
        # roll = -np.arctan(delta[1]/delta[0])
        # M = angle2matrix([0, 0, roll])
        landmark3d_align = np.dot(landmark3d_align, M)
        landmark3d_align = self.faces[index].landmark_3d_68[28]-landmark3d_align[28]+landmark3d_align
        drawLms(self.img,landmark3d_align[:,:2].astype('int'),x="2")
        print(best_inliers1)
        print(best_inliers2)
        print(best_inliers3)
        return [yaw,pitch,roll]
    def align_face2(self,index):
        '''
        dir1 vector to dir2 vector's matrix
        '''
        plane = Plane()
        face_plane = np.array([8,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])#17,18,19,20,21,22,23,24,25,26
        face_plane = np.concatenate((face_plane,np.arange(48,68)))
        best_eq, best_inliers = plane.fit(self.faces[index]["landmark_3d_68"][face_plane,:], thresh=20)
        print(len(best_inliers))
        pitch = np.arctan(best_eq[1]/best_eq[2])
        x = best_eq[0]*best_eq[0]+best_eq[1]*best_eq[1]+best_eq[2]*best_eq[2]
        dir = np.array(best_eq[0:3]) if best_eq[2]<0 else -np.array(best_eq[0:3])
        # dir = dir.reshape((1,-1))
        dest = np.array([0,0,-1])
        M = rotation_matrix_from_vectors(dir,dest)
        # M = Rotation.align_vectors(dir,dest)[0].as_matrix()
        landmark3d_align = np.dot(M,np.transpose(self.faces[index].landmark_3d_68)).transpose()
        landmark3d_align = self.faces[index].landmark_3d_68[28]-landmark3d_align[28]+landmark3d_align
        drawLms(self.img,landmark3d_align[:,:2].astype('int'))
        
        yaw = np.arctan(best_eq[0]/best_eq[2])
    
        mid_plane = np.array([27,28,29,30,33,51,62,66,57,8])
        best_eq, best_inliers = plane.fit(landmark3d_align[mid_plane,:], thresh=1)
        print(len(best_inliers))
        roll = np.arctan(best_eq[1]/best_eq[0])
        dir = np.array(best_eq[0:3]) if best_eq[2]<0 else -np.array(best_eq[0:3])
        # dir = dir.reshape((1,-1))
        dest = np.array([-1,0,0])
        M = rotation_matrix_from_vectors(dir,dest)
       
        landmark3d_align = np.dot(M,np.transpose(landmark3d_align)).transpose()
        landmark3d_align = self.faces[index].landmark_3d_68[28]-landmark3d_align[28]+landmark3d_align
        drawLms(self.img,landmark3d_align[:,:2].astype('int'))
        return [yaw,pitch,roll]
    def align_face(self,index):#point fit and plane fit to rotate 3d landmark to front
        try:
            from .ransac import Plane
        except:
            from ransac import Plane
        plane = Plane()
        face_plane = np.array([8,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])#17,18,19,20,21,22,23,24,25,26
        face_plane = np.concatenate((face_plane,np.arange(48,68)))
        landmark3d_align = self.faces[index]["landmark_3d_68"].astype('float64')
        
        # drawLms(self.img,landmark3d_align[:,:2].astype('int'),color = [0,255,0])
        
        # drawLms(self.img,self.faces[index]["landmark_2d_106"][:,:2].astype('int'),color = [0,0,255])
        delta = landmark3d_align[16,:]-landmark3d_align[0,:]
        yaw = -np.arctan(delta[2]/delta[0])
        M_res = angle2matrix([0, 0, 0])
        M1 = angle2matrix([0, yaw, 0])
        landmark3d_align = np.dot(landmark3d_align, M1)
        M_res = np.dot(M_res,M1)
        # drawLms(self.img,landmark3d_align[:,:2].astype('int'))
        
        # mid_plane = np.array([27,28,29,30,33,51,62,66,57,8])
        # best_eq, best_inliers = plane.fit(landmark3d_align[mid_plane,:], thresh=1)
        # roll = np.arctan(best_eq[1]/best_eq[0])
        delta = landmark3d_align[16,:]-landmark3d_align[0,:]
        if landmark3d_align[16,0]<landmark3d_align[0,0]:
            roll = np.arctan(delta[1]/delta[0])+np.pi
        else:
            roll = np.arctan(delta[1]/delta[0])
        M2 = angle2matrix([0, 0, roll])
        landmark3d_align = np.dot(landmark3d_align, M2)
        M_res = np.dot(M_res,M2)
        ### ransac get pitch 
        # face_plane = np.array([8,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47])#17,18,19,20,21,22,23,24,25,26
        # face_plane = np.concatenate((face_plane,np.arange(48,68)))
        # best_eq, best_inliers1 = plane.fit(landmark3d_align[face_plane,:], thresh=20)
        face_plane = [31, 32, 33, 34, 35, 48, 49, 50 ,51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 ,64 ,65 ,66, 67,21,22,20,23]#36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 
        best_eq=plane.leastsquare_fit(landmark3d_align[face_plane,:])
        pitch = np.arctan(best_eq[1]/best_eq[2])
        M3 = angle2matrix([-pitch, 0, 0])
        M_res = np.dot(M_res,M3) 
        landmark3d_align = np.dot(landmark3d_align, M3)
    
        r = Rotation.from_matrix(M_res)
        euler = r.as_euler('xyz')
        euler[0]=-euler[0]
        euler[2]=-euler[2]
        # l1 = np.dot(landmark3d_align, np.linalg.inv(M_res))    [pitch，-roll,-yaw,]
        rot_matrix = np.linalg.inv(M_res)
        r = Rotation.from_matrix(rot_matrix)
        rotvec = r.as_rotvec()
        # euler = r.as_euler('xyz')
        # euler1 = r.as_euler('zyx')#返回[z,y,x],第一个值为z
        # rotvec[0]=-rotvec[0]
        rotvec[1]=-rotvec[1]
        # euler[1]=-euler[1]
        # euler1[1]=-euler1[1]
        landmark3d_align = self.faces[index].landmark_3d_68[28]-landmark3d_align[28]+landmark3d_align
        return rotvec, rot_matrix, euler, landmark3d_align[:,:2].astype('int')
    
    
if __name__ == "__main__":
    import os,sys,logging
    workingDir = os.path.split(sys.argv[0])[0]
    if (workingDir):
        workingDir = workingDir + "/"
    else:
        workingDir = "./"
    logging.basicConfig(
        #filename=workingDir + 'log.log',
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
        filemode='a',
        datefmt='%Y-%m-%d %A %H:%M:%S',
    )
    logging.info("test get_face_info start")
    insight_face_info = get_face_info(workingDir,False)
    frame = workingDir + "../test/data/000028.jpg"
    frame = cv2.imread(frame)
    faces, frames, framesForGender = insight_face_info.get_faces(frame,"000028")
    logging.info(f"test get_face_info end.detect {len(faces)} face ")
