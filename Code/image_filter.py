import cv2
from http_interface import *
try:
    from http_interface import *
except:
    from .http_interface import *
import pyfilter#have to install apt-get install libopencv-dev==4.2.0 and run in python 3.8.*
import math
class filter_crop:
    def __init__(self,rFolder) -> None:
        self.face_seg = faceParsingInterface(rFolder)
        try:
            from .get_face_info import get_face_info
        except:
            from get_face_info import get_face_info
        self.insight_face_info = get_face_info(rFolder,False)
    def pyfilter2neuralhd(self,img,image_name=""):
        #pyfilter 输出的： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（，向下）
        #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
        crop_image,mask = self.get_hair_seg(img,image_name)
        mask = cv2.resize(mask,(512,512))
        crop_image= cv2.resize(crop_image,(512,512))
        ori2D=pyfilter.GetImage(crop_image)#B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示(向下）

        ori2D[:,:,2]=0
        ori2D[:,:,1]=-ori2D[:,:,1]*0.5+0.5
        ori2D[:,:,0]=1-ori2D[:,:,0]
        ori2D=ori2D[:,:,[2,1,0]]
        ori2D=(ori2D*255).astype('uint8')
        ori2D[mask==127]=[127,127,127]
        ori2D[mask==0]=[0,0,0]
        cv2.imshow("2",ori2D)
        cv2.waitKey()
        return ori2D,mask
        
    def get_hair_seg(self,img,image_name):
        faces, frames, _ = self.insight_face_info.get_faces(img, image_name)
        frames[0] = img
        imgB64 = cvmat2base64(frames[0])
        detectedFacePart, parsing = self.face_seg.request_faceParsing(image_name, 'img', imgB64)
        parsing[(parsing!=17) & (parsing!=0)]=127
        parsing[parsing==17]=255
        # frames[0][parsing==127]=[127,127,127]
        # frames[0][parsing==0]=[0,0,0]
        # cv2.imshow("1",frames[0])
        # cv2.waitKey()
        return frames[0],parsing
if __name__=="__main__":
    img = cv2.imread("/home/yxh/Documents/company/NeuralHDHair/Code/image.jpg")
    fil = filter_crop("/home/yxh/Documents/company/NeuralHDHair/Code")
    fil.pyfilter2neuralhd(img)