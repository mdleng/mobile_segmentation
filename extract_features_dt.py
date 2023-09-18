import os
import numpy as np
import glob
import cv2
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        '--palm_shape',
        type=str,
        #default='dataset_260723_shape/dataset_ensamble_toy/shape/palmfront_classes'
        default='dataset_260723_shape/shape/palmfront_classes'
    )
    
    parser.add_argument(
        '--finger_shape',
        type=str,
        #default='dataset_260723_shape/dataset_ensamble_toy/shape/fingersfront_classes'
        default='dataset_260723_shape/shape/fingersfront_classes'
    )
    
    args = parser.parse_args()
    '''
    paths={'palm':'dataset_260723_shape/dataset_ensamble_toy/palmfront_classes',
          'finger':'dataset_260723_shape/dataset_ensamble_toy/fingersfront_classes',
          'palm_shape':'dataset_260723_shape/dataset_ensamble_toy/shape/palmfront_classes',
          'finger_shape':'dataset_260723_shape/dataset_ensamble_toy/shape/fingersfront_classes'}
    '''
    
    paths={'palm_shape':args.palm_shape,
          'finger_shape':args.finger_shape}



    paths_img_palm_shape = glob.glob(os.path.join(paths['palm_shape'], "*/*.jpg"))

    paths_img_index_shape = glob.glob(os.path.join(paths['finger_shape']+'/index', "*/*.jpg"))

    paths_img_middle_shape = glob.glob(os.path.join(paths['finger_shape']+'/middle', "*/*.jpg"))

    paths_img_ring_shape = glob.glob(os.path.join(paths['finger_shape']+'/ring', "*/*.jpg"))

    paths_img_pinky_shape = glob.glob(os.path.join(paths['finger_shape']+'/pinky', "*/*.jpg"))
    
    #### palm
    paths_img=paths_img_palm_shape
    
    for path_img in paths_img:
        img=cv2.imread(path_img,0)
        img=cv2.resize(img,(30,30)).reshape(-1,1).squeeze()
        img=img/255
        path_img_out=path_img.replace('.jpg', '_dist.txt')
        np.savetxt(path_img_out,img)
        print(path_img_out,img.shape,np.max(img))
        
    #### index
    paths_img=paths_img_index_shape
    
    for path_img in paths_img:
        img=cv2.imread(path_img,0)
        img=cv2.resize(img,(30,30)).reshape(-1,1).squeeze()
        img=img/255
        path_img_out=path_img.replace('.jpg', '_dist.txt')
        np.savetxt(path_img_out,img)
        print(path_img_out,img.shape,np.max(img))
        
    #### middle
    paths_img=paths_img_middle_shape
    
    for path_img in paths_img:
        img=cv2.imread(path_img,0)
        img=cv2.resize(img,(30,30)).reshape(-1,1).squeeze()
        img=img/255
        path_img_out=path_img.replace('.jpg', '_dist.txt')
        np.savetxt(path_img_out,img)
        print(path_img_out,img.shape,np.max(img))
                                            
    #### ring
    paths_img=paths_img_ring_shape
    
    for path_img in paths_img:
        img=cv2.imread(path_img,0)
        img=cv2.resize(img,(30,30)).reshape(-1,1).squeeze()
        img=img/255
        path_img_out=path_img.replace('.jpg', '_dist.txt')
        np.savetxt(path_img_out,img)
        print(path_img_out,img.shape,np.max(img))
        
        
    #### pinky
    paths_img=paths_img_pinky_shape
    
    for path_img in paths_img:
        img=cv2.imread(path_img,0)
        img=cv2.resize(img,(30,30)).reshape(-1,1).squeeze()
        img=img/255
        path_img_out=path_img.replace('.jpg', '_dist.txt')
        np.savetxt(path_img_out,img)
        print(path_img_out,img.shape,np.max(img))