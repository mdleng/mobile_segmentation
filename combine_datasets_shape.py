import os
import shutil
import cv2
import glob

'''
datasets=['H:/elance/hand/classificazione/hand_recognition_dataset2_masks_010623/hand_recognition_dataset2/shape',
          'H:/elance/hand/classificazione/hand_recognition_dataset2_masks_010623/hand_recognition_dataset2bis/shape'
          ]
datasets_name=['SL','HN']

path_out_base='H:/elance/hand/classificazione/hand_recognition_dataset2_masks_010623'
'''

####### datasets da combinare
datasets=['H:/elance/hand/classificazione/dataset_260723_shape/datasets/han_dataset/shape',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/hid/shape',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/righthandgreen/shape',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/rightpass/shape',
          'H:/elance/hand/classificazione/dataset_260723_shape/datasets/scanleft/shape',]
datasets_name=['HN','HID','RHG','RP','SL']

datasets=['H:/elance/hand/classificazione/dataset_260723_shape/datasets/birjand']
datasets_name=['BI']

path_out_base='H:/elance/hand/classificazione/dataset_260723_shape'
###########################


path_out_base_shape=os.path.join(path_out_base,'shape')
print('path_out_base_shape',path_out_base_shape)


    
path_out=os.path.join(path_out_base_shape,'palmfront')

if os.path.exists(path_out_base_shape)==False:
    os.mkdir(path_out_base_shape)
    
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
        ### copy distance transform
        filename_src=images[ct_image]
        basename=os.path.basename(filename_src)
        basename_dst=dataset_name+'-'+basename
        filename_dst=os.path.join(path_out,basename_dst)
        #print(filename_src,filename_dst)
        shutil.copyfile(filename_src,filename_dst)
        ### copy shape filename
        filename_src=filename_src.replace('.jpg','.txt')
        
        filename_dst=filename_dst.replace('.jpg','.txt')
        
        shutil.copyfile(filename_src,filename_dst)
        
        
        
        
    
    
    
    
