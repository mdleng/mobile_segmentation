#test_classificazione_finger2.py  questa verisone divide immagine in tre parti e li inserisce nei tre canali
#   miglioro anche il crop


import os
import shutil
import cv2
import glob 
import numpy as np

#path_input='H:/elance/hand/classificazione/hand_recognition_dataset_030723/fingersfrontScanleft_040723/kaggle/hand_recognition_dataset_010723/fingersfront'
#path_input='H:/elance/hand/classificazione/hand_recognition_dataset_030723/fingersfront'
path_input='H:\elance\hand\classificazione\dataset_260723_shape/fingersfront'

path_output=path_input+'_classes'
if os.path.exists(path_output):
    shutil.rmtree(path_output)
    
os.mkdir(path_output)


images=glob.glob(path_input+"/*.jpg")

print(len(images))
classes=[]
ctimg=0
for filename in images:
 #try:
    ctimg+=1
    print(ctimg/len(images),filename)
    basename=os.path.basename(filename)
    
    s=basename.split('_')
    id_user=s[0]
    hand_type=s[2]
    finger_type=s[3].split('.')[0]
    #print(basename, id_user,hand_type,finger_type)
    folder_out_finger=os.path.join( path_output ,finger_type )
    if not os.path.exists(folder_out_finger):
        os.mkdir(folder_out_finger)
    folder_out=os.path.join( folder_out_finger , id_user+'_'+hand_type)
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
        
        
    filename_out=os.path.join(folder_out,basename)
    
    img=cv2.imread(filename)
    h=img.shape[0]
    hroi=int(h/3)
    
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    mask=np.zeros_like(img_gray)
    mask[img_gray>5]=1
    
    kernelsize=31
    
    mask=cv2.erode(mask,np.ones((kernelsize,kernelsize),dtype='uint8'))
    mask=cv2.dilate(mask,np.ones((kernelsize,kernelsize),dtype='uint8'))
    
    [r,c]=np.where(mask>0)
    x1_crop=int(np.min(c))
    y1_crop=int(np.min(r))
    
    x2_crop=int(np.max(c))
    y2_crop=int(np.max(r))
    
    img_gray=img_gray[y1_crop:y2_crop,x1_crop:x2_crop]
    
    #cv2.imwrite('DEBUGimg.jpg',img)
    #cv2.imwrite('DEBUGmask.jpg',mask*255)
    
    img1=img_gray[0:hroi,:]
    img2=img_gray[hroi:2*hroi,:]
    img3=img_gray[2*hroi:h,:]
    
    img1=cv2.resize(img1,(224,244))
    img2=cv2.resize(img2,(224,244))
    img3=cv2.resize(img3,(224,244))
    img=cv2.resize(img,(224,244))
    
    img[:,:,0]=img1
    img[:,:,1]=img2
    img[:,:,2]=img3
    
    
    cv2.imwrite(filename_out,img)
    #shutil.copyfile(filename,filename_out )
# except :
#    print('error',filename)