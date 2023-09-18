import os
import shutil
import cv2
import glob 
#path_input='hand_recognition_dataset_050623\palmfront2'

#path_input='H:\elance\hand\classificazione\hand_recognition_dataset_030723\palm_training_data\palmfront'
path_input='H:\elance\hand\classificazione\dataset_260723_shape\palmfront'
path_output=path_input+'_classes'
if os.path.exists(path_output):
    shutil.rmtree(path_output)
    
os.mkdir(path_output)


images=glob.glob(path_input+"/*.jpg")

print(len(images))
classes=[]
for filename in images:
    basename=os.path.basename(filename)
    
    s=basename.split('_')
    id_user=s[0]
    hand_type=s[2].split('.')[0]
    print(basename, id_user,hand_type)
    folder_out=os.path.join( path_output , id_user+'_'+hand_type)
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)
        
        
    filename_out=os.path.join(folder_out,basename)  
    shutil.copyfile(filename,filename_out )