import numpy as np
import cv2
import os
import shutil
import glob
import argparse
#####################################à
######################################
###################################à
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_input", type=str)
    parser.add_argument("--path_output", type=str)
    args = parser.parse_args()

    path_input=args.path_input
    path_output=args.path_output
    #path_input='hand_recognition_dataset_050623/palmfront'
    #path_output='hand_recognition_dataset_050623/palmfront2'

    if os.path.exists(path_output):
        shutil.rmtree(path_output)
        
    os.mkdir(path_output)


    images=glob.glob(path_input+"/*.jpg")

    print(len(images))
    wmax=300
    for filename in images:
        basename=os.path.basename(filename)
        filename_out=os.path.join(path_output,basename)
        img=cv2.imread(filename)
        mask=np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
        mask[img[:,:,0]==0]=1
        kernel=np.ones((51,51),dtype='uint8')
        mask=cv2.erode(mask,kernel)
        mask=cv2.dilate(mask,kernel)
        dist = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
        max_dist=np.max(dist)
        dist_norm=dist/max_dist
        y,x=np.where(dist_norm==1)
        
       # cv2.imwrite('debug_dist.jpg',(dist_norm*255))
        
        yc=int(np.mean(y))
        xc=int(np.mean(x))
        w=int(0.6*max_dist)
        #yc=int(img.shape[0]/2)
        #xc=int(img.shape[1]/2)
        #w=min(int(dist[yc,xc]),wmax)
        print(max_dist,w)
        img_roi=img[ yc-w:yc+w,xc-w:xc+w,: ]
        cv2.imwrite(filename_out,img_roi)
        
    
    
    
    
