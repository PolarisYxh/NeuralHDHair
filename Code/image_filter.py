import cv2
import pyfilter#have to install apt-get install libopencv-dev==4.2.0 and run in python 3.8.*
def pyfilter2neuralhd(image_path):
    #pyfilter 输出的： B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示（，向下）
    #neuralhd R:第三通道，（0,1）表示（向左，向右）；G:第二通道，（0,1）表示（向下，向上）
    image_path = "./image.jpg"
    # image_path = "/home/yxh/Documents/company/p2c-service/test/data/fix_test/celeb/Screenshot from 2023-03-15 15-29-51.png"
    crop_image=cv2.imread(image_path)
    crop_image= cv2.resize(crop_image,(512,512))
    # gt = cv2.imread("/home/yxh/Documents/company/p2c-service/strandhair/Ori.png")
    # gt= cv2.resize(gt,(512,512))
    # cv2.imshow("1",gt)
    # cv2.waitKey()
    orientation=pyfilter.GetImage(crop_image)#B:第1通道，（0,1）表示（向右，向左）；G:第二通道，（0,1）表示(向下）
    # orientation=(orientation*255).astype('uint8')
    # cv2.imshow("2",orientation)
    # cv2.waitKey()
    orientation[:,:,2]=0
    orientation[:,:,1]=-orientation[:,:,1]*0.5+0.5
    orientation[:,:,0]=1-orientation[:,:,0]
    orientation=orientation[:,:,[2,1,0]]
    orientation=(orientation*255).astype('uint8')
    cv2.imshow("2",orientation)
    cv2.waitKey()
pyfilter2neuralhd()