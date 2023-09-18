import os
import shutil
import cv2
import glob

'''
datasets=['H:/elance/hand/classificazione/hand_recognition_dataset_030723/fingersfrontScanleft_040723/kaggle/hand_recognition_dataset_010723',
          'H:/elance/hand/classificazione/hand_recognition_dataset_030723/Han_fingersfront_040723/kaggle/working/hand_recognition_dataset'
          ]
datasets_name=['SL','HN']
path_out='H:/elance/hand/classificazione/hand_recognition_dataset_030723/fingersfront'
'''
####### datasets da combinare
datasets=['H:/elance/hand/classificazione/dataset_260723_shape/datasets/han_dataset',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/hid',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/righthandgreen',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/rightpass',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/scanleft',]
datasets_name=['HN','HID','RHG','RP','SL']


datasets=['H:/elance/hand/classificazione/dataset_260723_shape/datasets/birjand']
datasets_name=['BI']

path_out='H:/elance/hand/classificazione/dataset_260723_shape/fingersfront'
###########################


if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.mkdir(path_out)


for ct_dataset in range(len(datasets)):
    
    dataset=datasets[ct_dataset]
    dataset_name=datasets_name[ct_dataset]
    print(dataset_name)
    path_images=os.path.join(dataset,'fingersfront')
    images=glob.glob(path_images+"/*.jpg")
    print(dataset,len(images))
    nimages=len(images)
    for ct_image in range(nimages):
     # try:
        
        filename_src=images[ct_image]
        basename=os.path.basename(filename_src)
        basename_dst=dataset_name+'-'+basename
        filename_dst=os.path.join(path_out,basename_dst)
        #print(filename_src,filename_dst)
        #print(filename_src,filename_dst)
        shutil.copyfile(filename_src,filename_dst)
     # except:
        #print(filename_src,filename_dst)
        
        
        
    
    
    
    
