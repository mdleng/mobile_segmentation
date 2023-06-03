import torch
import numpy as np
import cv2

import glob

import os
import argparse
import mediapipe as mp
#import skimage.measure
from timeit import default_timer as timer


#import glob
import shutil
import random


### test onnx floating vero 
import pickle


# Inference with ONNX Runtime
import onnxruntime
from onnx import numpy_helper
#########
from PIL import Image
def preprocess_image(image, height, width, channels=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # image = Image.open(image_path)
    image_data = cv2.resize(image,(width, height)).astype(np.float32)    
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHW
    mean = np.array([0, 0, 0]) + 0.5
    std = np.array([0, 0, 0]) + 0.5
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]
    image_data = np.expand_dims(image_data, 0)
    return image_data

def get_circular_mask(image_flip,index,middle,ring,thumb,pinky,wrist):
    image_view=np.zeros((image_flip.shape[0], image_flip.shape[1]), dtype='uint8')
    index=index.astype('int')
    middle=middle.astype('int')
    ring=ring.astype('int')
    thumb=thumb.astype('int')
    pinky=pinky.astype('int')
    wrist=wrist.astype('int')

    pt1=[thumb[0,0],thumb[0,1]]
    pt2=[index[0,0],index[0,1]]
    pt3=[middle[0,0],middle[0,1]]
    pt4=[ring[0,0],ring[0,1]]
    pt5=[pinky[0,0],pinky[0,1]]
    pt6=[wrist[0],wrist[1]]
    pt7=[thumb[1,0],thumb[1,1]]
    pt8=[thumb[0,0],thumb[0,1]]

    r=int(np.sqrt((pt6[0]-pt3[0])**2 + (pt6[1]-pt3[1])**2)*1.2)
    cv2.circle(image_view,( pt3[0], pt3[1]), r, (255), -1)

    return image_view

def get_skeleton(image_flip,index,middle,ring,thumb,pinky,wrist):
    line_thickness=30
    #image_view=image_flip.copy()
    image_view=np.zeros((image_flip.shape[0], image_flip.shape[1]), dtype='uint8')
    index=index.astype('int')
    middle=middle.astype('int')
    ring=ring.astype('int')
    thumb=thumb.astype('int')
    pinky=pinky.astype('int')
    wrist=wrist.astype('int')
 
    p1=0.2
    p2=1-p1
    pt0_index=(int((p1*index[0,0]+p2*index[2,0])),int((p1*index[0,1]+p2*index[2,1])))
    cv2.line(image_view, pt0_index, (index[2,0],index[2,1]), (10), thickness=line_thickness)
    cv2.line(image_view, (index[2,0],index[2,1]), (index[1,0],index[1,1]), (10), thickness=line_thickness)
    cv2.line(image_view, (index[1,0],index[1,1]), (index[3,0],index[3,1]), (10), thickness=line_thickness)

    pt0_middle=(int((p1*middle[0,0]+p2*middle[2,0])),int((p1*middle[0,1]+p2*middle[2,1])))       
    cv2.line(image_view, pt0_middle, (middle[2,0],middle[2,1]), (20), thickness=line_thickness)
    cv2.line(image_view, (middle[2,0],middle[2,1]), (middle[1,0],middle[1,1]), (20), thickness=line_thickness)
    cv2.line(image_view, (middle[1,0],middle[1,1]), (middle[3,0],middle[3,1]), (20), thickness=line_thickness)
    
    pt0_ring=(int((p1*ring[0,0]+p2*ring[2,0])),int((p1*ring[0,1]+p2*ring[2,1])))
    cv2.line(image_view, pt0_ring, (ring[2,0],ring[2,1]), (30), thickness=line_thickness)
    cv2.line(image_view, (ring[2,0],ring[2,1]), (ring[1,0],ring[1,1]), (30), thickness=line_thickness)
    cv2.line(image_view, (ring[1,0],ring[1,1]), (ring[3,0],ring[3,1]), (30), thickness=line_thickness)
    
    pt0_pinky=(int((p1*pinky[0,0]+p2*pinky[2,0])),int((p1*pinky[0,1]+p2*pinky[2,1])))
    cv2.line(image_view, pt0_pinky, (pinky[2,0],pinky[2,1]), (40), thickness=line_thickness)
    cv2.line(image_view, (pinky[2,0],pinky[2,1]), (pinky[1,0],pinky[1,1]), (40), thickness=line_thickness)
    cv2.line(image_view, (pinky[1,0],pinky[1,1]), (pinky[3,0],pinky[3,1]), (40), thickness=line_thickness)
    
    cv2.line(image_view, (thumb[0,0],thumb[0,1]), (thumb[2,0],thumb[2,1]), (50), thickness=line_thickness)
    cv2.line(image_view, (thumb[2,0],thumb[2,1]), (thumb[3,0],thumb[3,1]), (50), thickness=line_thickness)
    
    pt1=[thumb[0,0],thumb[0,1]]
    pt2=[index[0,0],index[0,1]]
    pt3=[middle[0,0],middle[0,1]]
    pt4=[ring[0,0],ring[0,1]]
    pt5=[pinky[0,0],pinky[0,1]]
    pt6=[wrist[0],wrist[1]]
    pt7=[thumb[1,0],thumb[1,1]]
    pt8=[thumb[0,0],thumb[0,1]]
    pts_palm=[pt1,pt2,pt3,pt4,pt5,pt6,pt7,pt8]
    pts_palm=np.array(pts_palm)
    
    pts_palm = pts_palm.reshape((-1,1,2))
    cv2.fillPoly(image_view, [pts_palm], color =(60))

    #################################    
    cv2.line(image_view, pt1, pt2, (70), thickness=2)
    cv2.line(image_view, pt2, pt3, (70), thickness=2)
    cv2.line(image_view, pt3, pt4, (70), thickness=2)
    cv2.line(image_view, pt4, pt5, (70), thickness=2)
    
    r=int(np.sqrt((pt6[0]-pt3[0])**2 + (pt6[1]-pt3[1])**2))
    cv2.circle(image_view,( pt3[0], pt3[1]), r, (80), 2)
    #################################
    

    return image_view



def finger_segmentation(img,img_seg,index,middle,pinky,thumb,ring,wrist):
    image_flip = cv2.flip(img, 1)
    output=cv2.flip(img_seg, 1)
    
    skelethon=get_skeleton(image_flip,index,middle,ring,thumb,pinky,wrist)
    mask=np.zeros_like(skelethon)
    mask[skelethon>0]=255
    mask[skelethon==80]=0
    mask_palm_border=np.zeros_like(skelethon)
    mask_palm_border[(skelethon==70) | (skelethon==80)]=255

    mask_palm=np.zeros_like(skelethon)
    mask_palm[skelethon==60]=255
    
    circ_mask=get_circular_mask(image_flip,index,middle,ring,thumb,pinky,wrist)
    ### added 201122######
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dmax=np.max(d)
    
    d2 = cv2.distanceTransform(255-mask, cv2.DIST_L2, 3)
    
   
    mask_good=np.zeros_like(skelethon)
    mask_good[d2<(dmax/2)]=1
    ######################

   
    dists=[]
    for ct in range(1,7):
        skelethon_finger=(skelethon==ct*10).astype('uint8')
        d = cv2.distanceTransform(1-skelethon_finger, cv2.DIST_L2, 3)
        
        dists.append(d)

    dists=np.array(dists)
    dmin=np.argmin(dists,axis=0)
    dmin=(dmin+1)*10
    dmin2=dmin.copy()
    
    #output = segment
    
    dmin2[output==0]=0
    finger=dmin2.copy()
    finger[dmin2==60]=0
    finger[mask_good==0]=0
    palm=np.zeros_like(finger)
    palm[dmin2==60]=255
    palm[circ_mask==0]=0
    return finger,palm


def get_landmarks_coord(image,results):
   
    ct_hand=0
    ##image_height, image_width, _ = image.shape
    ### no normalization
    image_height=1
    image_width=1

    XY_index_full=[]
    XY_middle_full=[]
    XY_ring_full=[]
    XY_thumb_full=[]  
    XY_pinky_full=[]
    XY_wrist_full=[]
    for hand_landmarks in results.multi_hand_landmarks:

          XY_index=[]
          XY_middle=[]
          XY_ring=[]
          XY_thumb=[]  
          XY_pinky=[]
          XY_wrist=[]

          ct_hand+=1
          ### 1 index
          x=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
          XY_index.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
          XY_index.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
          XY_index.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
          XY_index.append([x,y])

          
          
          ### 2 middle
          
          x=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
          XY_middle.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
          XY_middle.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
          XY_middle.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
          XY_middle.append([x,y])

          

          ### 3 ring
          
          x=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
          XY_ring.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
          XY_ring.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
          XY_ring.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)
          XY_ring.append([x,y])

          


          ### 4 pinky
          
          x=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
          XY_pinky.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
          XY_pinky.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
          XY_pinky.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)
          XY_pinky.append([x,y])

          

          ### 5 thumb
          
          x=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y *image_height)
          XY_thumb.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
          XY_thumb.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
          XY_thumb.append([x,y])

          x=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
          XY_thumb.append([x,y])

      

          ### 6 wrist
          
          x=(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)
          y=(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
          XY_wrist.append([x,y])
         


          XY_index_full.append(XY_index)
          XY_middle_full.append(XY_middle)
          XY_ring_full.append(XY_ring)
          XY_thumb_full.append(XY_thumb) 
          XY_pinky_full.append(XY_pinky)
          XY_wrist_full.append( XY_wrist)
         


    landmarks_coord = {
      "XY_index_full": XY_index_full,
      "XY_middle_full": XY_middle_full,
      "XY_ring_full": XY_ring_full,
      "XY_thumb_full": XY_thumb_full,
      "XY_pinky_full": XY_pinky_full,
      "XY_wrist_full": XY_wrist_full,
    }
    return landmarks_coord


def get_mp_mask(image,hands):
    
    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
   
    if not results.multi_hand_world_landmarks:
      return np.zeros((image.shape[0], image.shape[0]),dtype='uint8')
    
    image_hight, image_width, _ = image.shape
    annotated_image = cv2.flip(image.copy(), 1)
    annotated_image=np.zeros_like(annotated_image)
    
    for hand_landmarks in results.multi_hand_landmarks:
      
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

    out=((annotated_image[:,:,0]>0) | (annotated_image[:,:,1]>0) | (annotated_image[:,:,2]>0)).astype('uint8')
    out=cv2.flip(out,1)
    return out,results

def get_hand_type(results):
  types=[]
  ct=0
  for hand in results.multi_handedness:
      handType=hand.classification[0].label

      ### palm or back
      hand_landmarks=results.multi_hand_landmarks[ct]
      x_wrist=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
      y_wrist=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y

      x_thumbtip=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
      y_thumbtip=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

      x_midtip=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
      y_midtip=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y


      a=np.array([x_thumbtip-x_wrist,y_thumbtip-y_wrist],float)
      b=np.array([x_midtip-x_wrist,y_midtip-y_wrist],float)
      u=np.cross(a,b)
      if handType =='Left' and u <= 0: 
        handSide='Palm'
      if handType =='Left' and u > 0: 
        handSide='Back'
      if handType =='Right' and u <= 0: 
        handSide='Back'
      if handType =='Right' and u > 0: 
        handSide='Palm'
      ####################




      types.append([handType,handSide])
  return types

###################################à
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathname", type=str)
    parser.add_argument("--path_out", type=str,default='test_images_out')
    parser.add_argument("--size_img", type=int,default=640)
    parser.add_argument("--nmax", type=int,default=1000)
    parser.add_argument("--scale_out", type=float,default=1)
    parser.add_argument("--max_num_hands", type=int,default=1)
    parser.add_argument("--model_path", type=str,default="mobilenet_seg.onnx")
    
    args = parser.parse_args()
    
    pathname=args.pathname
    
    path_out=args.path_out
    if os.path.exists(path_out):
        shutil.rmtree(path_out)
    os.mkdir(path_out)
    size_img=args.size_img
    
    nmax=args.nmax
    scale_out=args.scale_out
    max_num_hands=args.max_num_hands
    
    
    #### mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands= mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_num_hands,
        min_detection_confidence=0.2)
    ###############
    
    ### onnx
    
    onnx_path=args.model_path
    
    if torch.cuda.is_available():
      session_fp32 = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    else:
      session_fp32 = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    
    
    files=glob.glob(pathname + '/*.*')
    print('total files', len(files))
    random.seed(4)
    random.shuffle(files)
    files=files[0:min(nmax,len(files))]
    padding=True
    for filename in files:
     try:
            basename=os.path.basename(filename)
            filename_out=os.path.join(path_out,basename)
            print(filename,filename_out)
            path_data=''
            frame=cv2.imread(filename)
            if padding == True:
                
                pad=int(frame.shape[0]/10)
                shape_before_pad=frame.shape
                print('padding ',pad)
                
                frame_big=np.zeros((frame.shape[0]+2*pad,frame.shape[1]+2*pad,3),dtype='uint8')
                frame_big[pad:pad+frame.shape[0],pad:pad+frame.shape[1],:]=frame
                frame=frame_big
                frame_to_save=frame.copy()
                
            frame_width_orig=frame.shape[1]
            frame_height_orig=frame.shape[0]
            if size_img==0:
              scale_inp=1
            else:
              scale_inp= size_img/np.max(frame.shape)
            frame=cv2.resize(frame,(int(frame.shape[1]*scale_inp),int(frame.shape[0]*scale_inp)))
            print('scaled ',size_img, scale_inp, frame.shape)

            img =frame.copy()


            shape_orig=img.shape
            start_time = timer()
            
            start_time_mp  = timer()
            mask_mp,results=get_mp_mask(frame,hands)  ### mediapipe su immagine a dimensione originale, posso anche fare opposto e scalare sopo results
            end_time_mp  = timer() 
            hand_types=get_hand_type(results)
            #print(basename,hand_types)
            ### finger_palm segmetation
            landmarks_coord=get_landmarks_coord(frame,results)
            ### normalizaiton
            #image_height_norm, image_width_norm, _ = frame.shape
            #landmarks_coord_norm=landmarks_coord.copy()
            #####
            finger_segm=np.zeros_like(mask_mp)
            palm_segm=np.zeros_like(mask_mp)
            XY_index_full=landmarks_coord["XY_index_full"]
            XY_middle_full=landmarks_coord["XY_middle_full"]
            XY_ring_full=landmarks_coord["XY_ring_full"]
            XY_thumb_full=landmarks_coord["XY_thumb_full"]
            XY_pinky_full=landmarks_coord["XY_pinky_full"]
            XY_wrist_full=landmarks_coord["XY_wrist_full"]
            
            
            start_time_seg  = timer()
            ### onnx hand segmentation
            input_arr=preprocess_image(frame, 224,224, channels=3)
            outputs = session_fp32.run([], {'input':input_arr})[0]
            frame_seg = np.argmax(outputs.squeeze(), axis=0)*255
            frame_seg=frame_seg.astype('uint8') 
            frame_seg = cv2.resize(frame_seg,(frame.shape[1], frame.shape[0]))

            #cv2.imwrite('DEBUG_frame_seg.jpg',frame_seg)
            #cv2.imwrite('DEBUG_frame.jpg',frame)
            #########################
            end_time_seg  = timer()
            
            for ct_hand in range(len(XY_index_full)):
                
                    XY_index=np.array(XY_index_full[ct_hand])
                    XY_middle=np.array(XY_middle_full[ct_hand])
                    XY_ring=np.array(XY_ring_full[ct_hand])
                    XY_thumb=np.array(XY_thumb_full[ct_hand])
                    XY_pinky=np.array(XY_pinky_full[ct_hand])
                    XY_wrist=np.array(XY_wrist_full[ct_hand]).squeeze()

                    image_height, image_width, _ = frame.shape
                    XY_index[:,0]*=image_width
                    XY_middle[:,0]*=image_width
                    XY_ring[:,0]*=image_width
                    XY_thumb[:,0]*=image_width
                    XY_pinky[:,0]*=image_width
                    XY_wrist[0]*=image_width

                    XY_index[:,1]*=image_height
                    XY_middle[:,1]*=image_height
                    XY_ring[:,1]*=image_height
                    XY_thumb[:,1]*=image_height
                    XY_pinky[:,1]*=image_height
                    XY_wrist[1]*=image_height

                    try:
                      finger,palm=finger_segmentation(frame,frame_seg,index=XY_index,middle=XY_middle,pinky=XY_pinky,thumb=XY_thumb,ring=XY_ring,wrist=XY_wrist)
                    except Exception as e :
                      print('error finger_segmentation ',XY_index.shape,str(e))
                      finger=np.zeros((frame.shape[0],frame.shape[1]),dtype='uint8')
                      palm=np.zeros((frame.shape[0],frame.shape[1]),dtype='uint8')
                   
                    finger_segm[finger==10]=10
                    finger_segm[finger==20]=20
                    finger_segm[finger==30]=30
                    finger_segm[finger==40]=40
                    finger_segm[finger==50]=50
                    palm_segm[palm>0]=1

            
            end_time  = timer()
            print('processing time (sec)',end_time - start_time, 'MP', end_time_mp - start_time_mp,'SEG', end_time_seg - start_time_seg )

            #######################
            #### draw image #######
            #######################
            palm_segm = cv2.flip(palm_segm, 1)
            finger_segm = cv2.flip(finger_segm, 1)

            kernel=np.ones((11,11),dtype='uint8')
            palm_segm=cv2.erode(palm_segm,kernel)
            palm_segm=cv2.dilate(palm_segm,kernel)

            finger_segm=cv2.erode(finger_segm,kernel)
            finger_segm=cv2.dilate(finger_segm,kernel)

            '''
            finger1_segm=np.zeros_like(finger_segm)
            finger2_segm=np.zeros_like(finger_segm)
            finger3_segm=np.zeros_like(finger_segm)
            finger4_segm=np.zeros_like(finger_segm)
            finger5_segm=np.zeros_like(finger_segm)

            finger1_segm[finger_segm==10]=1
            
            finger2_segm[finger_segm==20]=1
            finger3_segm[finger_segm==30]=1
            finger4_segm[finger_segm==40]=1
            finger5_segm[finger_segm==50]=1

            contours1 = cv2.findContours(finger1_segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            #print(' contours1 ',  contours1)
            contours2 = cv2.findContours(finger2_segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours3 = cv2.findContours(finger3_segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours4 = cv2.findContours(finger4_segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours5 = cv2.findContours(finger5_segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contours_palm = cv2.findContours(palm_segm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            contour_size=3

            frame_view2=frame.copy()
            cv2.drawContours(frame, contours1, -1, (0,0,255), contour_size)
            cv2.drawContours(frame, contours2, -1, (255,0,255), contour_size)
            cv2.drawContours(frame, contours3, -1, (0,255,255), contour_size)
            cv2.drawContours(frame, contours4, -1, (255,0,0), contour_size)
            cv2.drawContours(frame, contours5, -1, (255,255,0), contour_size)
            cv2.drawContours(frame, contours_palm, -1, (0,255,0), contour_size)

           
            r=frame_view2[:,:,2]
            g=frame_view2[:,:,1]
            b=frame_view2[:,:,0]

            r[finger_segm==10]=100

            g[finger_segm==20]=100

            b[finger_segm==30]=100

            r[finger_segm==40]=60
           
            g[finger_segm==50]=60
           
            b[palm_segm>0]=50
            frame_view2[:,:,2]=r
            frame_view2[:,:,1]=g
            frame_view2[:,:,0]=b
            contour_size=1
            cv2.drawContours(frame_view2, contours1, -1, (255,255,255), contour_size)
            cv2.drawContours(frame_view2, contours2, -1, (255,255,255), contour_size)
            cv2.drawContours(frame_view2, contours3, -1, (255,255,255), contour_size)
            cv2.drawContours(frame_view2, contours4, -1, (255,255,255), contour_size)
            cv2.drawContours(frame_view2, contours5, -1, (255,255,255), contour_size)
            cv2.drawContours(frame_view2, contours_palm, -1, (255,255,255), contour_size)
            '''
            finger_segm[palm_segm>0]=255
            frame_view2=finger_segm
            frame_view2=cv2.resize( frame_view2, (int(frame_width_orig),int(frame_height_orig)),interpolation = cv2.INTER_NEAREST)
            '''
            if padding == True:
                shape_frame_out=frame.shape
                
                ## resize to original size
                frame_view2=cv2.resize( frame_view2, (int(frame_width_orig),int(frame_height_orig)))
                
                frame_view2=frame_view2[pad:pad+shape_before_pad[0],pad:pad+shape_before_pad[1]]
                    
                frame_view2=cv2.resize( frame_view2, (shape_frame_out[1],shape_frame_out[0]))
                
                   
            '''
            cv2.imwrite(filename_out.lower(),frame_to_save)

            filename_out=filename_out.lower().replace('jpg','png')
            #frame_view2=cv2.resize( frame_view2, (int(frame_view2.shape[1]*scale_out),int(frame_view2.shape[0]*scale_out)))
            cv2.imwrite(filename_out,frame_view2)

            data_pkl = {
            "hand_types": hand_types,
            "landmarks_coord": landmarks_coord,
          
            }

            filename_out_pkl=filename_out.replace('png','pkl')
            file_pkl = open(filename_out_pkl, 'wb')
            pickle.dump(data_pkl, file_pkl)
            file_pkl.close()
     except Exception as e :
    #        print('error ',filename, str(e))
    print('done')
