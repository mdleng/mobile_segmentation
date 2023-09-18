import os
import shutil
import cv2
import glob


'''
datasets=['H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmrighthandgreen_060723_0_300\kaggle\hand_recognition_datasetrighthandgreen',
          'H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmfrontScanleft_060723\kaggle\hand_recognition_dataset_010723',
          'H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmfrontRightPass_060723_0_300\kaggle\hand_recognition_datasetrightpass',
          'H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmdatabasehid_0_300\kaggle\hand_recognition_databasehid',
          'H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\Han_palmfront_060723\kaggle\working\hand_recognition_dataset',
          'H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmfrontScanleft_090723_part2\kaggle\hand_recognition_dataset_010723']


datasets_name=['GR','SL','RP','BH','HN','SL2']
path_out='H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmfront'
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






path_out='H:/elance/hand/classificazione/dataset_260723_shape/palmfront'
###########################



if os.path.exists(path_out):
    shutil.rmtree(path_out)
os.mkdir(path_out)


for ct_dataset in range(len(datasets)):
    
    dataset=datasets[ct_dataset]
    dataset_name=datasets_name[ct_dataset]
    print(dataset_name)
    path_images=os.path.join(dataset,'palmfront')
    images=glob.glob(path_images+"/*.jpg")
    print(dataset,len(images))
    nimages=len(images)
    for ct_image in range(nimages):
        filename_src=images[ct_image]
        basename=os.path.basename(filename_src)
        basename_dst=dataset_name+'-'+basename
        filename_dst=os.path.join(path_out,basename_dst)
        shutil.copyfile(filename_src,filename_dst)
        
        
        
        
    
    
    
    
