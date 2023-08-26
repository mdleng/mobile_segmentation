## versione con palmo ruotato

import os
import pickle
import glob
import cv2
import numpy as np
import argparse
import shutil
def MovePerpendicularCounterClockwiseImg(p,v,d): 
    x=p[0]
    y=p[1]
    vx=v[0]
    vy=v[1]
    return [int(x+vy*d),int(y-vx*d)]

def MovePerpendicularClockwiseImg(p,v,d):
    x=p[0]
    y=p[1]
    vx=v[0]
    vy=v[1]
    return [int(x-vy*d),int(y+vx*d)]

def MoveForward(p,v,d):
    x=p[0]
    y=p[1]
    vx=v[0]
    vy=v[1]
    return [int(x+d*vx),int(y+d*vy)]

def MoveBackward(p,v,d):
    x=p[0]
    y=p[1]
    vx=v[0]
    vy=v[1]
    return [int(x-d*vx),int(y-d*vy)]


def get_rotated_bb2 ( pBottom,pTop,v,w):
    pt=MoveForward(pTop,v,w)
    ptl=MovePerpendicularCounterClockwiseImg(pt,v,w)
    ptr=MovePerpendicularClockwiseImg(pt,v,w)
    pb=MoveBackward(pBottom,v,w)
    pbl=MovePerpendicularCounterClockwiseImg(pb,v,w)
    pbr=MovePerpendicularClockwiseImg(pb,v,w)
    
    return ptl, ptr,pbr,pbl,pt,pb


def get_rotated_bb3 ( pBottom,pTop,v,w,wforward):
    pt=MoveForward(pTop,v,wforward)
    ptl=MovePerpendicularCounterClockwiseImg(pt,v,w)
    ptr=MovePerpendicularClockwiseImg(pt,v,w)
    pb=MoveBackward(pBottom,v,wforward)
    pbl=MovePerpendicularCounterClockwiseImg(pb,v,w)
    pbr=MovePerpendicularClockwiseImg(pb,v,w)
    
    return ptl, ptr,pbr,pbl,pt,pb


def get_rotated_bb_palm ( pBottom,pTop,v,w):
    r=int(w/2)
    pt=pTop
    ptl=MovePerpendicularCounterClockwiseImg(pt,v,r)
    ptr=MovePerpendicularClockwiseImg(pt,v,r)
    pb=MoveBackward(pTop,v,w)
    pbl=MovePerpendicularCounterClockwiseImg(pb,v,r)
    pbr=MovePerpendicularClockwiseImg(pb,v,r)
    
    return ptl, ptr,pbr,pbl,pt,pb




## width roi = 2*w

def checkFinger(image,XY):
    if len(XY)>0:
        isgood=True
    else:
        isgood=False
        
    for ct in range(XY.shape[0]):
        x=int(XY[ct,0])
        y=int(XY[ct,1])
        if x<0 or x>image.shape[1] or y<0 or y>image.shape[0]:
            isgood=False
        else:
            if image[y,x,0]==0:
                isgood=False
    return isgood
                
            
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
    return newImage,M
       
       


def get_rotated_bb ( p,v,w):
    
    pt=MoveForward(p,v,w)
    ptl=MovePerpendicularCounterClockwiseImg(pt,v,w)
    ptr=MovePerpendicularClockwiseImg(pt,v,w)
    
    pb=MoveBackward(p,v,w)
    pbl=MovePerpendicularCounterClockwiseImg(pb,v,w)
    pbr=MovePerpendicularClockwiseImg(pb,v,w)
    
    return ptl, ptr,pbr,pbl,pt,pb
    
def get_finger_verson(XY):
    TIP=XY[-1,:]
    MCP=XY[0,:]
    vx= TIP[0]-MCP[0]
    vy= TIP[1]-MCP[1]
    n=np.sqrt(vx*vx+vy*vy)
    vx=vx/n
    vy=vy/n
    v=[vx,vy]
    return v


def get_rotated_roi(img,mask,label_finger,pBottom,pTop,w=100,wforward=100):
                #pBottom=XY_index_full[0,:]
                #pTop=XY_index_full[-1,:]
                vx= pTop[0]-pBottom[0]
                vy= pTop[1]-pBottom[1]
                n=np.sqrt(vx*vx+vy*vy)
                vx=vx/n
                vy=vy/n
                v=[vx,vy]  ### versor bottom - top  

                #w=100
                #print('p',p)
                #print('v',v)
               # ptl, ptr,pbr,pbl,pt,pb=get_rotated_bb2 ( pBottom,pTop,v,w)
                ptl, ptr,pbr,pbl,pt,pb=get_rotated_bb3 ( pBottom,pTop,v,w,wforward)
                pts=np.array([ptl, ptr,pbr,pbl])
            
                L= np.sqrt( (pt[0]-pb[0])*(pt[0]-pb[0])  + (pt[1]-pb[1])*(pt[1]-pb[1]) )*1.2
                xc=(pt[0]+pb[0])/2
                yc=(pt[1]+pb[1])/2

                x1=int(xc - L/2)
                y1=int(yc - L/2)
                x2=int(xc + L/2)
                y2=int(yc + L/2)

                bb_img_roi=[x1,y1,x2,y2] #x1,y1,x2,y2

                img_roi=img[y1:y2, x1:x2,:].copy()
                mask_roi=mask[y1:y2, x1:x2].copy()
                mask_roi_bin=np.zeros_like(mask_roi)
                mask_roi_bin[mask_roi==label_finger]=255

                offsetx=x1
                offsety=y1
                offset=np.array([x1,y1])
                ptl_roi=ptl-offset
                ptr_roi=ptr-offset
                pbr_roi=pbr-offset
                pbl_roi=pbl-offset
                pt_roi=pt-offset
                pb_roi=pb-offset


                # Define corresponding points
                rotated_roi=np.zeros_like(img_roi)

                xc_dest=int(rotated_roi.shape[1]/2)
                yc_dest=int(rotated_roi.shape[0]/2)

                dist_t_b=np.linalg.norm(np.array(pt) - np.array(pb))
                pt_roi_dest=[xc_dest, int(yc_dest -dist_t_b/2)]
                pb_roi_dest=[xc_dest, int(yc_dest +dist_t_b/2)]


                ptl_roi_dest=[pt_roi_dest[0]-w,pt_roi_dest[1]]
                ptr_roi_dest=[pt_roi_dest[0]+w,pt_roi_dest[1]]
                pbl_roi_dest=[pb_roi_dest[0]-w,pb_roi_dest[1]]
                pbr_roi_dest=[pb_roi_dest[0]+w,pb_roi_dest[1]]



                input_pts = np.float32([ptl_roi,ptr_roi,pbr_roi])
                output_pts = np.float32([ptl_roi_dest,ptr_roi_dest,pbr_roi_dest])
                M= cv2.getAffineTransform(input_pts , output_pts)
                # print(M)

                rotated_roi = cv2.warpAffine(img_roi,M, (img_roi.shape[1], img_roi.shape[0]))
                rotated_mask = cv2.warpAffine(mask_roi,M, (img_roi.shape[1], img_roi.shape[0]),flags=cv2.INTER_NEAREST)

                rotated_mask_bin=np.zeros_like(rotated_mask)
                rotated_mask_bin[rotated_mask==label_finger]=255

                r,c=np.where(rotated_mask_bin>0)
                x1_finger=np.min(c)
                y1_finger=np.min(r)
                x2_finger=np.max(c)
                y2_finger=np.max(r)


                final_roi=rotated_roi[y1_finger:y2_finger,x1_finger:x2_finger,:]
                final_roi_mask=rotated_mask_bin[y1_finger:y2_finger,x1_finger:x2_finger]
                final_roi_mask=cv2.erode(final_roi_mask,np.ones((5,5)))
                final_roi_mask=cv2.dilate(final_roi_mask,np.ones((5,5)))

                ch1=final_roi[:,:,0]
                ch2=final_roi[:,:,1]
                ch3=final_roi[:,:,2]

                ch1[final_roi_mask==0]=0
                ch2[final_roi_mask==0]=0
                ch3[final_roi_mask==0]=0

                final_roi[:,:,0]=ch1
                final_roi[:,:,1]=ch2
                final_roi[:,:,2]=ch3
                
                return final_roi

def get_rotated_roi_palm(img,mask,label_finger,pBottom,pTop,w):
               
                vx= pTop[0]-pBottom[0]
                vy= pTop[1]-pBottom[1]
                n=np.sqrt(vx*vx+vy*vy)
                vx=vx/n
                vy=vy/n
                v=[vx,vy]  ### versor bottom - top  

                ptl, ptr,pbr,pbl,pt,pb=get_rotated_bb_palm ( pBottom,pTop,v,w)
                pts=np.array([ptl, ptr,pbr,pbl])
            
                L= np.sqrt( (pt[0]-pb[0])*(pt[0]-pb[0])  + (pt[1]-pb[1])*(pt[1]-pb[1]) )*1.2
                xc=(pt[0]+pb[0])/2
                yc=(pt[1]+pb[1])/2

                x1=int(xc - L/2)
                y1=int(yc - L/2)
                x2=int(xc + L/2)
                y2=int(yc + L/2)

                bb_img_roi=[x1,y1,x2,y2] #x1,y1,x2,y2

                img_roi=img[y1:y2, x1:x2,:].copy()
                mask_roi=mask[y1:y2, x1:x2].copy()
                mask_roi_bin=np.zeros_like(mask_roi)
                mask_roi_bin[mask_roi==label_finger]=255

                offsetx=x1
                offsety=y1
                offset=np.array([x1,y1])
                ptl_roi=ptl-offset
                ptr_roi=ptr-offset
                pbr_roi=pbr-offset
                pbl_roi=pbl-offset
                pt_roi=pt-offset
                pb_roi=pb-offset


                # Define corresponding points
                rotated_roi=np.zeros_like(img_roi)

                xc_dest=int(rotated_roi.shape[1]/2)
                yc_dest=int(rotated_roi.shape[0]/2)

                dist_t_b=np.linalg.norm(np.array(pt) - np.array(pb))
                pt_roi_dest=[xc_dest, int(yc_dest -dist_t_b/2)]
                pb_roi_dest=[xc_dest, int(yc_dest +dist_t_b/2)]


                ptl_roi_dest=[pt_roi_dest[0]-w,pt_roi_dest[1]]
                ptr_roi_dest=[pt_roi_dest[0]+w,pt_roi_dest[1]]
                pbl_roi_dest=[pb_roi_dest[0]-w,pb_roi_dest[1]]
                pbr_roi_dest=[pb_roi_dest[0]+w,pb_roi_dest[1]]



                input_pts = np.float32([ptl_roi,ptr_roi,pbr_roi])
                output_pts = np.float32([ptl_roi_dest,ptr_roi_dest,pbr_roi_dest])
                M= cv2.getAffineTransform(input_pts , output_pts)
                # print(M)

                rotated_roi = cv2.warpAffine(img_roi,M, (img_roi.shape[1], img_roi.shape[0]))
                rotated_mask = cv2.warpAffine(mask_roi,M, (img_roi.shape[1], img_roi.shape[0]),flags=cv2.INTER_NEAREST)

                rotated_mask_bin=np.zeros_like(rotated_mask)
                rotated_mask_bin[rotated_mask==label_finger]=255

                r,c=np.where(rotated_mask_bin>0)
                x1_finger=np.min(c)
                y1_finger=np.min(r)
                x2_finger=np.max(c)
                y2_finger=np.max(r)


                final_roi=rotated_roi[y1_finger:y2_finger,x1_finger:x2_finger,:]
                final_roi_mask=rotated_mask_bin[y1_finger:y2_finger,x1_finger:x2_finger]
                final_roi_mask=cv2.erode(final_roi_mask,np.ones((5,5)))
                final_roi_mask=cv2.dilate(final_roi_mask,np.ones((5,5)))

                ch1=final_roi[:,:,0]
                ch2=final_roi[:,:,1]
                ch3=final_roi[:,:,2]

                ch1[final_roi_mask==0]=0
                ch2[final_roi_mask==0]=0
                ch3[final_roi_mask==0]=0

                final_roi[:,:,0]=ch1
                final_roi[:,:,1]=ch2
                final_roi[:,:,2]=ch3
                
                return final_roi

def get_rotated_bb3 ( pBottom,pTop,v,w,wforward):
    pt=MoveForward(pTop,v,wforward)
    ptl=MovePerpendicularCounterClockwiseImg(pt,v,w)
    ptr=MovePerpendicularClockwiseImg(pt,v,w)
    pb=MoveBackward(pBottom,v,wforward)
    pbl=MovePerpendicularCounterClockwiseImg(pb,v,w)
    pbr=MovePerpendicularClockwiseImg(pb,v,w)
    
    return ptl, ptr,pbr,pbl,pt,pb


def get_shape_features(roi,dist_ref):
    
    roi_bin=np.zeros((roi.shape[0],roi.shape[1]))
    roi_bin[roi[:,:,0]>0]=1
    features=[]
    for i in range(10,90,10):
        j=i/100
        r=int(j*roi.shape[0])
        c=int(j*roi.shape[1])
        prof_row=roi_bin[r,:]
        prof_col=roi_bin[:,c]
        feat_row=np.sum(prof_row)/dist_ref
        feat_col=np.sum(prof_col)/dist_ref
        features.append(feat_row)
        features.append(feat_col)
    return features
        
#####################################à
######################################
###################################à
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_data", type=str,default='H:\elance\hand\classificazione\hand_recognition_dataset2_masks_010623\hand_recognition_dataset2')
    parser.add_argument("--id", type=str,default='0')
    parser.add_argument("--clear", type=str,default='False')
    
    args = parser.parse_args()
    path_data=args.path_data
    id=args.id

    images=sorted(glob.glob(path_data+'/'+str(id)+"/*.JPG"))
    images.extend(glob.glob(path_data+'/'+str(id)+"/*.jpg"))
    images.extend(glob.glob(path_data+'/'+str(id)+"/*.png"))
    images.extend(glob.glob(path_data+'/'+str(id)+"/*.PNG"))
    images.extend(glob.glob(path_data+'/'+str(id)+"/*.bmp"))
    images.extend(glob.glob(path_data+'/'+str(id)+"/*.BMP"))
    
    print('n images',len(images))

    unlabel_paths=['fingersfront','palmfront','fingersback','palmback']
    for p in unlabel_paths:
        p_full=os.path.join(path_data,p)
        if os.path.exists(p_full) and args.clear=='True':
            shutil.rmtree(p_full)
        if not os.path.exists(p_full):
            os.mkdir(p_full)
            
    
    labels_fingers= {
      "index": 10,
      "middle":20,
      "ring": 30,
      "pinky": 40,
      #"thumb":50
    }
    #print(images[13])
    

    for id_img in range(len(images)):
     #sid_img=13
     try:
        filename_img=images[id_img]
        print(filename_img)

        basename=os.path.basename(filename_img)

        filename_img_pad=path_data+'/'+str(id)+"/masks/"+basename.lower()
        filename_mask=path_data+'/'+str(id)+"/masks/"+basename.lower()
        filename_mask=filename_mask.replace('jpg','png')
        filename_pkl=filename_mask.replace('png','pkl')
        
        
        if os.path.exists(filename_img_pad):
        
            img=cv2.imread(filename_img_pad)
            mask=cv2.imread(filename_mask,0)
            
            
            ### posso normalizzare immagine a dimensione standard
            ### 3100 x 3916 un pò gra,nde posso fare un resize ma faccio dopo per ottimizzazione
            img=cv2.resize(img, (3100 , 3916) )
            mask=cv2.resize(mask, (3100 , 3916),interpolation = cv2.INTER_NEAREST )
                
            ##############################
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        
            file = open(filename_pkl, 'rb')
            data_pkl= pickle.load(file)
            landmarks_coord = data_pkl['landmarks_coord']
            hand_type= data_pkl['hand_types'][0]
            
            file.close()

            XY_index_full=np.array(landmarks_coord["XY_index_full"]).squeeze()
            XY_middle_full=np.array(landmarks_coord["XY_middle_full"]).squeeze()
            XY_ring_full=np.array(landmarks_coord["XY_ring_full"]).squeeze()
            XY_thumb_full=np.array(landmarks_coord["XY_thumb_full"]).squeeze()
            XY_pinky_full=np.array(landmarks_coord["XY_pinky_full"]).squeeze()
            XY_wrist_full=np.array(landmarks_coord["XY_wrist_full"]).squeeze()


            image_height, image_width, _ = img.shape
            XY_index_full[:,0]*=image_width
            XY_middle_full[:,0]*=image_width
            XY_ring_full[:,0]*=image_width
            XY_thumb_full[:,0]*=image_width
            XY_pinky_full[:,0]*=image_width
            XY_wrist_full[0]*=image_width

            XY_index_full[:,1]*=image_height
            XY_middle_full[:,1]*=image_height
            XY_ring_full[:,1]*=image_height
            XY_thumb_full[:,1]*=image_height
            XY_pinky_full[:,1]*=image_height
            XY_wrist_full[1]*=image_height

            id_sort_thumb=[1,0,2,3]
            XY_thumb_full=XY_thumb_full[id_sort_thumb,:]

            id_sort_fing=[0,2,1,3]
            XY_index_full=XY_index_full[id_sort_fing,:]
            XY_middle_full=XY_middle_full[id_sort_fing,:]
            XY_ring_full=XY_ring_full[id_sort_fing,:]
            XY_pinky_full=XY_pinky_full[id_sort_fing,:]

            ##################
            ### extract finger roi ##
            ##################
            ### example INDEX , the other ar ethe same ###


            isgood=True#checkFinger(img,XY_index_full) ## modifier 250823
            #print('isgood',isgood)
            if isgood:


                #### PALM ####
            
                #mask_palm=np.zeros_like(mask)
                #mask_palm[mask==255]=1
                #rpalm,cpalm=np.where(mask_palm>0)
                #x1_palm=np.min(cpalm)
                #y1_palm=np.min(rpalm)
                #x2_palm=np.max(cpalm)
                #y2_palm=np.min(rpalm)
                #width_palm=x2_palm-x1_palm
                #height_palm=y2_palm-y1_palm
                #w_palm=int(np.max(width_palm,height_palm)/2)
                #pBottom=[int(np.mean(cpalm)), int(np.mean(rpalm))] #XY_wrist_full
                
                
                pBottom=XY_wrist_full
                pTop=XY_middle_full[0,:]
                
                pLeft=XY_index_full[0,:]
                pRight=XY_pinky_full[0,:]
                
                w_palm= np.linalg.norm(pRight-pLeft)
                print(pBottom,pTop)
                roi_palm_final=get_rotated_roi_palm(img,mask,255,pBottom,pTop,w=w_palm)
                
                offset_crop=int(roi_palm_final.shape[0]*0.07)
                roi_palm_final=roi_palm_final[offset_crop:roi_palm_final.shape[0]-offset_crop,offset_crop:roi_palm_final.shape[1]-offset_crop]
                
                roi_palm_final=cv2.resize(roi_palm_final,(512,512))
                
                dx=pBottom[0]-pTop[0]
                dy=pBottom[1]-pTop[1]
                dist_ref=np.sqrt(dx*dx + dy*dy)
                print('dist_ref',dist_ref)
                features_shape_palm=get_shape_features(roi_palm_final,dist_ref)
                
                ### write palm roi e features
                if hand_type[1]=='Palm':
                    path_out_palm=os.path.join(path_data,'palmfront')
                else:
                    path_out_palm=os.path.join(path_data,'palmback')
                    
                filename_palm_out=os.path.join(path_out_palm,str(id)+'_'+str(id_img)+'_'+hand_type[0]+'.jpg')
                filename_palm_out_shape=os.path.join(path_out_palm,str(id)+'_'+str(id_img)+'_'+hand_type[0]+'.txt')
                
                cv2.imwrite(filename_palm_out,roi_palm_final)
                np.savetxt(filename_palm_out_shape,features_shape_palm,fmt='%1.3f')
                    
                
                
                #### fingers
                
                for finger_type in sorted(list(labels_fingers.keys())):
                
                
                    label_finger=mask[int(XY_index_full[1,1]),int(XY_index_full[1,0])]
                    label_finger=labels_fingers[finger_type]
                    print(finger_type,'label_finger',label_finger)
                    if finger_type=='index':
                        pBottom=XY_index_full[0,:]
                        pTop=XY_index_full[-1,:]
                        isgoodIndex=checkFinger(img,XY_index_full)
                        
                    if finger_type=='middle':
                        pBottom=XY_middle_full[0,:]
                        pTop=XY_middle_full[-1,:]
                        isgoodMiddle=checkFinger(img,XY_middle_full)
                        
                    if finger_type=='ring':
                        pBottom=XY_ring_full[0,:]
                        pTop=XY_ring_full[-1,:]
                        isgoodRing=checkFinger(img,XY_ring_full)
                    
                    if finger_type=='thumb':
                        pBottom=XY_thumb_full[0,:]
                        pTop=XY_thumb_full[-1,:]
                        
                    if finger_type=='pinky':
                        pBottom=XY_pinky_full[0,:]
                        pTop=XY_pinky_full[-1,:]
                        isgoodPinky=checkFinger(img,XY_pinky_full)
                    
                    save_finger=False
                    if finger_type=='index' and  isgoodIndex==True:
                        save_finger=True
                    if finger_type=='middle' and  isgoodMiddle==True:
                        save_finger=True
                    if finger_type=='ring' and  isgoodRing==True:
                        save_finger=True
                    if finger_type=='pinky' and  isgoodPinky==True:
                        save_finger=True

                    if save_finger:
                        final_roi=get_rotated_roi(img,mask,label_finger,pBottom,pTop,w=100)
                        features_shape_finger=get_shape_features(final_roi,dist_ref)
                        
                        ### write finger roi e features 
                        if hand_type[1]=='Palm':
                            path_out_finger=os.path.join(path_data,'fingersfront')
                        else:
                            path_out_finger=os.path.join(path_data,'fingersback')
                        filename_finger_out=os.path.join(path_out_finger,str(id)+'_'+str(id_img)+'_'+hand_type[0]+'_'+finger_type+'.jpg')
                        filename_finger_out_shape=os.path.join(path_out_finger,str(id)+'_'+str(id_img)+'_'+hand_type[0]+'_'+finger_type+'.txt')  
                
                        print('saving ', filename_finger_out)
                        cv2.imwrite(filename_finger_out,final_roi)
                        np.savetxt(filename_finger_out_shape,features_shape_finger,fmt='%1.3f')
                    
                         
                   # isgoodIndex=checkFinger(img,XY_index_full)
                   # isgoodMiddle=checkFinger(img,XY_middle_full)
                   # isgoodRing=checkFinger(img,XY_ring_full)
                   # isgoodPinky=checkFinger(img,XY_pinky_full)
                    
                       

                    
                ###################################################################
     except Exception as e:
          print(e)
                
        
